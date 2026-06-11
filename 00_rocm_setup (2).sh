#!/bin/bash
# =============================================================================
# 00_rocm_setup.sh  –  AMD MI300X / ROCm Environment Setup
# IBM Granite Fraud Detection Fine-Tuning Project
# =============================================================================
# Run once before any training:  bash 00_rocm_setup.sh
# Then before every session:     source rocm_env.sh
# =============================================================================
set -e
echo "============================================================"
echo "  AMD MI300X ROCm Environment Setup"
echo "============================================================"

# ── 1. Verify ROCm / GPU visibility ─────────────────────────────────────────
echo ""
echo "[1/6] Verifying ROCm GPU visibility..."
rocm-smi --showproductname 2>/dev/null || echo "  WARNING: rocm-smi not found. Is ROCm installed?"
python3 -c "
import torch
print('  PyTorch  :', torch.__version__)
print('  ROCm GPU :', torch.cuda.is_available())
print('  GPU count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'  GPU {i}    : {torch.cuda.get_device_name(i)}  ({mem:.0f} GB)')
"

# ── 2. Detect ROCm version ───────────────────────────────────────────────────
echo ""
echo "[2/6] Detecting ROCm version..."
ROCM_VERSION=$(rocm-smi --version 2>/dev/null | grep -oP 'ROCm-\K[0-9]+\.[0-9]+' | head -1 || echo "6.2")
echo "  Detected ROCm: ${ROCM_VERSION}"
echo "  To (re)install ROCm-enabled PyTorch:"
echo "    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm${ROCM_VERSION}"

# ── 3. Install ROCm bitsandbytes ─────────────────────────────────────────────
echo ""
echo "[3/6] Installing bitsandbytes for ROCm (MI300X / gfx942)..."
pip install --no-deps --force-reinstall \
  'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl' \
  2>/dev/null && echo "  Pre-built wheel OK" || {
    echo "  Pre-built failed – building from source (gfx942)..."
    [ ! -d /tmp/bitsandbytes ] && git clone -b multi-backend-refactor \
      https://github.com/bitsandbytes-foundation/bitsandbytes.git /tmp/bitsandbytes
    cd /tmp/bitsandbytes
    apt-get install -y build-essential cmake 2>/dev/null || true
    cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx942" -S .
    make -j$(nproc)
    pip install -e .
    cd -
    echo "  Built from source OK"
  }

# ── 4. Flash Attention for ROCm ──────────────────────────────────────────────
echo ""
echo "[4/6] Installing Flash Attention (ROCm CK backend for gfx942)..."
pip install einops 2>/dev/null || true
pip install flash-attn --no-build-isolation 2>/dev/null && \
  echo "  flash-attn OK (CK backend)" || \
  echo "  flash-attn not available – SDPA fallback will be used (safe default)"

# ── 5. Other dependencies ────────────────────────────────────────────────────
echo ""
echo "[5/6] Installing training & vLLM dependencies..."
pip install \
  "transformers>=4.47.0" \
  "datasets>=2.21.0" \
  "accelerate>=0.34.0" \
  "peft>=0.12.0" \
  "trl>=0.11.0" \
  "deepspeed>=0.14.0" \
  "vllm>=0.4.0" \
  pandas numpy scikit-learn imbalanced-learn pyarrow \
  matplotlib seaborn tensorboard kaggle openai

# ── 6. Write rocm_env.sh ─────────────────────────────────────────────────────
echo ""
echo "[6/6] Writing rocm_env.sh..."
cat > rocm_env.sh << 'ENVEOF'
#!/bin/bash
# Source before every training/inference session:  source rocm_env.sh

# GPU selection (all 8 × MI300X by default; change for single-GPU: "0")
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCR_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}

# MI300X core HIP settings
export HSA_ENABLE_SDMA=0              # Disable SDMA – recommended for MI300X
export HIP_FORCE_DEV_KERNARG=1        # Better kernel launch latency
export HIP_LAUNCH_BLOCKING=0          # Async launch (do NOT set to 1 in prod)

# GEMM / BLAS
export TORCH_BLAS_PREFER_HIPBLASLT=1  # Prefer hipBLASLt over hipBLAS
export HIPBLASLT_ENABLED=1
export SAFETENSORS_FAST_GPU=1         # GPU-accelerated weight loading

# TunableOp – finds optimal GEMM kernel (~1-2 min warmup, persists to CSV)
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=1
export PYTORCH_TUNABLEOP_FILENAME="tunableop_results.csv"

# Memory allocator
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8"

# RCCL multi-GPU collectives
export NCCL_MIN_NCHANNELS=112         # Optimal for MI300X fully-connected topology
export NCCL_SOCKET_IFNAME=eth0        # Set to your NIC name
export RCCL_MSCCL_ENABLE=0           # Disable MSCCL if you see instability
export NCCL_DEBUG=WARN

# vLLM-specific
export VLLM_USE_TRITON_FLASH_ATTN=0  # Use CK backend (faster on MI300X gfx942)

echo "ROCm env loaded. GPU(s): ${HIP_VISIBLE_DEVICES}"
ENVEOF
chmod +x rocm_env.sh

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  1. source rocm_env.sh"
echo "  2. python 04_finetune_granite_rocm.py"
echo "  3. python 05_vllm_inference_rocm.py"
echo "============================================================"
