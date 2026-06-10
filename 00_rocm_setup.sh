#!/bin/bash
# =============================================================================
# 00_rocm_setup.sh
# AMD MI300X / ROCm Environment Setup for IBM Granite Fine-Tuning
# =============================================================================
# Run ONCE before any training:  bash 00_rocm_setup.sh
# =============================================================================

set -e

echo "============================================================"
echo "  AMD MI300X ROCm Environment Setup"
echo "============================================================"

# ── 1. Verify ROCm / GPU visibility ──────────────────────────────────────────
echo ""
echo "[1/6] Verifying ROCm GPU visibility..."
rocm-smi --showproductname 2>/dev/null || echo "  WARNING: rocm-smi not found. Is ROCm installed?"
rocminfo 2>/dev/null | grep -E "Name|gfx" | head -20 || true
python3 -c "import torch; print('  PyTorch version :', torch.__version__); \
             print('  ROCm available  :', torch.cuda.is_available()); \
             print('  GPU count       :', torch.cuda.device_count()); \
             [print(f'  GPU {i}          : {torch.cuda.get_device_name(i)}') \
              for i in range(torch.cuda.device_count())]"

# ── 2. Install ROCm-compatible PyTorch (if not already) ──────────────────────
echo ""
echo "[2/6] Checking PyTorch ROCm build..."
ROCM_VERSION=$(rocm-smi --version 2>/dev/null | grep -oP 'ROCm-\K[0-9]+\.[0-9]+' | head -1 || echo "6.2")
echo "  Detected ROCm version: ${ROCM_VERSION}"
echo "  To reinstall PyTorch for ROCm, run:"
echo "    pip install torch torchvision torchaudio \\"
echo "      --index-url https://download.pytorch.org/whl/rocm${ROCM_VERSION}"

# ── 3. Install ROCm-specific bitsandbytes ────────────────────────────────────
echo ""
echo "[3/6] Installing bitsandbytes for ROCm (MI300X / gfx942)..."
echo "  Method A – Pre-built wheel (MI210/MI250/MI300A/MI300X and newer):"
pip install --no-deps --force-reinstall \
  'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl' \
  2>/dev/null && echo "  ✓ Pre-built wheel installed" || {
    echo "  Pre-built wheel failed. Trying build from source..."
    # Method B – Build from source with ROCm HIP backend
    if [ ! -d "/tmp/bitsandbytes" ]; then
      git clone -b multi-backend-refactor \
        https://github.com/bitsandbytes-foundation/bitsandbytes.git \
        /tmp/bitsandbytes
    fi
    cd /tmp/bitsandbytes
    apt-get install -y build-essential cmake 2>/dev/null || true
    # gfx942 = MI300X; add gfx90a for MI250 if needed
    cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx942" -S .
    make -j$(nproc)
    pip install -e .
    cd -
    echo "  ✓ Built from source"
  }

# ── 4. Install ROCm flash-attention (CK backend for MI300X) ──────────────────
echo ""
echo "[4/6] Installing Flash Attention for ROCm (Composable Kernel backend)..."
echo "  Note: Standard pip install flash-attn WILL FAIL on ROCm."
echo "  Using ROCm fork with CK backend for gfx942..."
pip install einops 2>/dev/null || true
# ROCm flash-attention via ROCm's fork
pip install flash-attn --no-build-isolation 2>/dev/null && \
  echo "  ✓ flash-attn installed" || {
    echo "  Standard flash-attn failed. Will use SDPA (torch built-in) fallback."
    echo "  This is acceptable: PyTorch SDPA on ROCm calls optimized Triton kernels."
    echo "  Set USE_FLASH_ATTN=false in 04_finetune_granite_rocm.py"
  }

# ── 5. Install remaining dependencies ────────────────────────────────────────
echo ""
echo "[5/6] Installing training dependencies..."
pip install \
  transformers>=4.47.0 \
  datasets>=2.21.0 \
  accelerate>=0.34.0 \
  peft>=0.12.0 \
  trl>=0.11.0 \
  deepspeed>=0.14.0 \
  pandas numpy scikit-learn imbalanced-learn pyarrow \
  matplotlib seaborn tensorboard kaggle

# ── 6. Export ROCm performance environment variables ─────────────────────────
echo ""
echo "[6/6] Writing ROCm performance environment to rocm_env.sh..."
cat > rocm_env.sh << 'ENVEOF'
#!/bin/bash
# Source this file before training: source rocm_env.sh

# ── GPU visibility (set which GPUs to use) ────────────────────────────────
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # all 8 × MI300X
# export HIP_VISIBLE_DEVICES=0                # single GPU mode

# ── ROCm / HIP core settings ─────────────────────────────────────────────
export HSA_ENABLE_SDMA=0           # Disable SDMA; recommended for MI300X
export HIP_LAUNCH_BLOCKING=0       # Async kernel launch (faster)
export ROCR_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}

# ── PyTorch TunableOp (auto-finds best GEMM kernels, ~1-2 min warmup) ────
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=1
export PYTORCH_TUNABLEOP_FILENAME="tunableop_results.csv"

# ── RCCL (collective comms for multi-GPU) ─────────────────────────────────
export NCCL_SOCKET_IFNAME=eth0          # adjust to your NIC
export RCCL_MSCCL_ENABLE=0             # disable MSCCL if instability
export NCCL_DEBUG=WARN                  # INFO for verbose collective logs

# ── Flash Attention backend selection ─────────────────────────────────────
# Options: "flash_attention_2" (requires rocm flash-attn build)
#          "sdpa"              (PyTorch Scaled Dot-Product Attention — safe default)
export ROCM_ATTN_BACKEND=sdpa

# ── hipBLASLt for optimized GEMM ──────────────────────────────────────────
export HIPBLASLT_ENABLED=1

# ── Memory allocator ──────────────────────────────────────────────────────
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8"

echo "ROCm environment loaded. GPU(s): ${HIP_VISIBLE_DEVICES}"
ENVEOF
chmod +x rocm_env.sh

echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "  Next steps:"
echo "    1. source rocm_env.sh         (load env vars)"
echo "    2. python 04_finetune_granite_rocm.py"
echo "============================================================"
