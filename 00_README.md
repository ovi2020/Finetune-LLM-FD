# IBM Granite — Fraud Detection & Financial Crime Intelligence
## Fine-Tuning + ROCm + vLLM on AMD MI300X

---

## Project Structure

```
granite_fraud_finetuning/
│
├── 00_README.md                    ← This file
├── 00_rocm_setup.sh                ← One-time ROCm environment setup (generates rocm_env.sh)
│
├── 01_dataset_survey.py            ← Dataset registry, metadata, download helpers
├── 02_data_preprocessing.py        ← Full preprocessing pipeline for all 6 datasets
├── 03_instruction_dataset.py       ← Convert tabular rows → Granite chat-format JSONL
│
├── 04_finetune_granite.py          ← QLoRA fine-tuning (CUDA / NVIDIA path)
├── 04_finetune_granite_rocm.py     ← QLoRA fine-tuning (ROCm / AMD MI300X path)
│
├── 05_evaluate_model.py            ← Evaluation: F1, MCC, AUROC, confusion matrix
├── 05_vllm_inference_rocm.py       ← vLLM inference & serving (4 modes, MI300X)
│
├── accelerate_rocm_zero2.yaml      ← HuggingFace Accelerate config: ZeRO-2 (8B multi-GPU)
├── accelerate_rocm_zero3.yaml      ← HuggingFace Accelerate config: ZeRO-3 (70B multi-GPU)
│
├── requirements.txt                ← Dependencies for CUDA / NVIDIA path
└── requirements_rocm.txt           ← Dependencies for ROCm / AMD MI300X path
```

---

## Datasets

Six publicly available datasets covering credit card fraud, e-commerce fraud,
mobile payment fraud, anti-money laundering (AML), crypto forensics, and
retail banking fraud.

| # | Dataset | Type | Rows | Fraud % | Source |
|---|---------|------|------|---------|--------|
| 1 | Credit Card Fraud (ULB) | Credit card transactions | 284K | 0.17% | Kaggle |
| 2 | IEEE-CIS Fraud Detection | E-commerce + identity | 590K | 3.5% | Kaggle |
| 3 | PaySim Synthetic | Mobile money transfers | 6.3M | 0.13% | Kaggle |
| 4 | IBM AML Synthetic (HI-Small) | Inter-bank wire transfers | ~5M edges | ~2–10% | GitHub/IBM |
| 5 | Elliptic Bitcoin Dataset | On-chain BTC transactions | 203K nodes | 2% | Kaggle |
| 6 | BankSim | Retail bank transactions | 594K | 1.2% | Kaggle |

Download all datasets with:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
kaggle competitions download -c ieee-fraud-detection
kaggle datasets download -d ealaxi/paysim1
kaggle datasets download -d ellipticco/elliptic-data-set
kaggle datasets download -d ntnu-testimon/banksim1
git clone https://github.com/IBM/AMLSim   # IBM AML Synthetic
```

---

## Full Pipeline

```
Raw CSV Datasets  (./data/)
        │
        ▼
01_dataset_survey.py
  → Prints metadata, checks which files are present, exports registry JSON

        │
        ▼
02_data_preprocessing.py
  → Per-dataset:
      • Missing value imputation (median / mode)
      • Feature engineering (balance diffs, velocity counts, time features)
      • Categorical encoding (LabelEncoder / one-hot)
      • SMOTE oversampling + RandomUnderSampler (handles extreme class imbalance)
      • StandardScaler normalisation
      • 80 / 10 / 10 stratified train / val / test split
  → Output: ./data/processed/<dataset>_{train,val,test}.parquet

        │
        ▼
03_instruction_dataset.py
  → Converts each tabular row → Granite chat-format prompt/completion pair
  → System prompt: financial crime expert persona
  → User prompt:   transaction features serialised as natural language
  → Assistant:     structured JSON verdict (verdict, risk_score, risk_level,
                   fraud_type, aml_typology, risk_indicators, recommended_action)
  → Output: ./data/sft/{train,val,test}.jsonl  (Granite messages format)

        │
        ├──────────────────────────────────┐
        ▼                                  ▼
04_finetune_granite.py            04_finetune_granite_rocm.py
  (NVIDIA / CUDA path)              (AMD MI300X / ROCm path)
  QLoRA + paged_adamw_8bit          QLoRA + adamw_torch_fused
  flash_attention_2                 sdpa / CK flash-attn
  blocksize=128                     blocksize=64  ← CDNA3 warp-64
  → ./checkpoints/granite-fraud-qlora/final_adapter

        │
        ▼  (--merge flag)
  merge_adapter()
  → ./checkpoints/granite-fraud-merged/   ← standalone safetensors model
                                            required before vLLM

        │
        ├──────────────────────────────────┐
        ▼                                  ▼
05_evaluate_model.py              05_vllm_inference_rocm.py
  HuggingFace Transformers          vLLM (PagedAttention)
  F1, MCC, AUROC                    4 modes (offline / server /
  Per-dataset breakdown             client / benchmark)
  Confusion matrix PNG              OpenAI-compatible REST API
```

---

## GPU Requirements

### CUDA / NVIDIA Path (`04_finetune_granite.py`)

| Model | Minimum VRAM | Recommended |
|-------|-------------|-------------|
| Granite 8B (QLoRA) | 2× A100 40 GB | 2× A100 80 GB |
| Granite 70B (QLoRA) | 4× A100 80 GB | 8× A100 80 GB |

Cloud: AWS `p4d.24xlarge`, GCP A3, RunPod A100, Lambda Labs.

### ROCm / AMD Path (`04_finetune_granite_rocm.py`)

| Model | Minimum | Notes |
|-------|---------|-------|
| Granite 8B (QLoRA) | 1× MI300X (192 GB) | Fits comfortably in single GPU |
| Granite 70B (QLoRA) | 2× MI300X (384 GB) | ZeRO-2 sufficient |
| Granite 70B (full BF16) | 4× MI300X (768 GB) | ZeRO-3 recommended |

Each MI300X has 192 GB HBM3 and 5.3 TB/s bandwidth (gfx942 / CDNA3).

---

## Quick Start — NVIDIA / CUDA

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets → ./data/
python 01_dataset_survey.py

# 3. Preprocess all datasets
python 02_data_preprocessing.py

# 4. Build instruction-tuning JSONL
python 03_instruction_dataset.py

# 5. Fine-tune (8B default; add --model ibm-granite/granite-3.3-70b-instruct for 70B)
python 04_finetune_granite.py

# 6. Evaluate
python 05_evaluate_model.py
```

---

## Quick Start — AMD MI300X / ROCm

### Step 1 — Environment setup (run once)

```bash
bash 00_rocm_setup.sh
# Installs ROCm bitsandbytes, flash-attn (CK backend), and all deps.
# Generates rocm_env.sh with all required HIP / RCCL environment variables.
```

### Step 2 — Load environment (every session)

```bash
source rocm_env.sh
```

Key variables set by `rocm_env.sh`:

| Variable | Value | Purpose |
|----------|-------|---------|
| `HIP_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | Expose all 8× MI300X |
| `HSA_ENABLE_SDMA` | `0` | Disable SDMA (MI300X recommendation) |
| `HIP_FORCE_DEV_KERNARG` | `1` | Faster kernel argument passing |
| `TORCH_BLAS_PREFER_HIPBLASLT` | `1` | Use hipBLASLt for optimal GEMM |
| `SAFETENSORS_FAST_GPU` | `1` | GPU-accelerated weight loading |
| `PYTORCH_TUNABLEOP_ENABLED` | `1` | Auto-find best GEMM kernel (~1–2 min warmup) |
| `NCCL_MIN_NCHANNELS` | `112` | Optimal RCCL channels for MI300X topology |
| `VLLM_USE_TRITON_FLASH_ATTN` | `0` | Use CK attention backend (faster on gfx942) |

### Step 3 — Preprocess data (same as CUDA path)

```bash
python 01_dataset_survey.py
python 02_data_preprocessing.py
python 03_instruction_dataset.py
```

### Step 4 — Fine-tune on MI300X

```bash
# Single GPU (8B model)
python 04_finetune_granite_rocm.py

# Multi-GPU with DeepSpeed ZeRO-2 (8B, 4× MI300X)
accelerate launch --config_file accelerate_rocm_zero2.yaml \
    04_finetune_granite_rocm.py

# 70B model on 8× MI300X with DeepSpeed ZeRO-3
accelerate launch --config_file accelerate_rocm_zero3.yaml \
    04_finetune_granite_rocm.py \
    --model ibm-granite/granite-3.3-70b-instruct

# Merge LoRA adapter into base model after training (required for vLLM)
python 04_finetune_granite_rocm.py --merge
```

### Step 5 — Serve with vLLM

```bash
# Offline batch inference (Python API, highest throughput)
python 05_vllm_inference_rocm.py --mode offline

# OpenAI-compatible REST API server
python 05_vllm_inference_rocm.py --mode server

# Call the running server from Python
python 05_vllm_inference_rocm.py --mode client

# Throughput benchmark (100 synthetic transactions)
python 05_vllm_inference_rocm.py --mode benchmark

# Print the recommended Docker run command
python 05_vllm_inference_rocm.py --mode docker

# 4-GPU tensor parallel (for 70B)
python 05_vllm_inference_rocm.py --mode server --tp 4 \
    --model ./checkpoints/granite-fraud-merged-70b
```

---

## ROCm Code Changes vs CUDA Version

`04_finetune_granite_rocm.py` is a drop-in replacement for `04_finetune_granite.py`
with the following targeted changes (each marked `# ROCm CHANGE` in the source):

| # | Setting | CUDA value | ROCm value | Reason |
|---|---------|-----------|-----------|--------|
| 1 | `bnb_4bit_blocksize` | `128` (default) | **`64`** | MI300X wavefront = 64 threads (CDNA3 warp-64); 128 causes wrong dequantisation |
| 2 | `attn_implementation` | `flash_attention_2` | **`sdpa`** | Standard flash-attn PyPI package won't build on ROCm; PyTorch SDPA dispatches to AOTriton/CK automatically |
| 3 | `optim` | `paged_adamw_8bit` | **`adamw_torch_fused`** | Paged optimizer uses CUDA paging primitives not in HIP |
| 4 | `ddp_find_unused_parameters` | implicit | **`False` (explicit)** | ROCm DDP hangs when True |
| 5 | HIP env vars | — | Set at module load | `HSA_ENABLE_SDMA`, `HIP_FORCE_DEV_KERNARG`, `NCCL_MIN_NCHANNELS`, etc. |
| 6 | DeepSpeed configs | optional | Auto-generated | `ds_zero2_rocm.json` / `ds_zero3_rocm.json` written on first 70B run |
| 7 | `torch_dtype` | `auto` | **`bfloat16` (explicit)** | Prevents fp32 fallback on ROCm |

---

## vLLM Inference Modes (`05_vllm_inference_rocm.py`)

vLLM must be loaded with the **merged model** (not the raw PEFT adapter).
Run `python 04_finetune_granite_rocm.py --merge` first.

| Mode | Flag | Best for |
|------|------|---------|
| Offline batch | `--mode offline` | Bulk historical transaction analysis; highest throughput via PagedAttention |
| API server | `--mode server` | Real-time integration with fraud monitoring systems; OpenAI-compatible `/v1/chat/completions` |
| Client demo | `--mode client` | Testing the live server from Python using `openai` SDK |
| Benchmark | `--mode benchmark` | Measuring tokens/sec on your MI300X hardware |
| Docker command | `--mode docker` | Prints the optimal `docker run` command for the ROCm vLLM image |

### Recommended Docker approach (best MI300X performance)

AMD provides a pre-tuned Docker image with ROCm, PyTorch and vLLM already
compiled against Composable Kernel (CK) for gfx942:

```bash
# Disable NUMA auto-balancing on the host first
echo 0 | sudo tee /proc/sys/kernel/numa_balancing

# Pull the ROCm vLLM image
docker pull rocm/vllm:instinct_main

# Run the server (single GPU example)
docker run --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --shm-size 32G --network=host \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e VLLM_USE_TRITON_FLASH_ATTN=0 \
  -e NCCL_MIN_NCHANNELS=112 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -v $(pwd)/checkpoints:/models \
  rocm/vllm:instinct_main \
  vllm serve /models/granite-fraud-merged \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --max-num-seqs 256 \
    --num-scheduler-steps 10 \
    --enable-prefix-caching \
    --disable-log-requests \
    --host 0.0.0.0 --port 8000
```

---

## Preprocessing Summary

| Issue | Approach |
|-------|----------|
| Extreme class imbalance | SMOTE (minority → 10% of majority) + RandomUnderSampler |
| High-cardinality categoricals | `LabelEncoder` (IEEE-CIS), one-hot (PaySim `type`) |
| Missing values | Median for numerics, mode for categoricals |
| Feature leakage (PaySim balances) | Replaced with engineered `balance_diff_orig / dest` |
| Timestamp features (IBM AML) | Extracted `hour_of_day`, `day_of_week`, velocity count |
| Elliptic unlabelled nodes | Filtered out class-3 "unknown" nodes before splitting |
| Output format | Parquet (per-dataset, per-split) → JSONL (combined, chat-formatted) |

---

## Instruction Dataset Format

Each example in `./data/sft/*.jsonl` uses Granite's native messages format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a financial crime intelligence expert..."
    },
    {
      "role": "user",
      "content": "Analyze the following transaction...\n  - amount_scaled: 2.87\n  - type_TRANSFER: 1\n  ..."
    },
    {
      "role": "assistant",
      "content": "{\n  \"verdict\": \"FRAUD\",\n  \"risk_score\": 5,\n  \"risk_level\": \"HIGH\",\n  \"fraud_type\": \"rapid fund movement\",\n  \"aml_typology\": \"fan-out\",\n  \"risk_indicators\": [...],\n  \"recommended_action\": \"BLOCK and escalate...\",\n  \"confidence\": \"HIGH\"\n}"
    }
  ],
  "metadata": {
    "dataset": "ibm_aml",
    "split": "train",
    "label": 1,
    "is_fraud": true
  }
}
```

---

## Evaluation Metrics (`05_evaluate_model.py`)

| Metric | Why it matters for fraud |
|--------|--------------------------|
| **F1 Score** | Balances precision and recall on imbalanced classes |
| **MCC (Matthews)** | Most reliable single metric for heavily skewed fraud data |
| **AUROC** | Threshold-independent ranking quality |
| **Precision** | Cost of false alerts (analyst workload) |
| **Recall** | Cost of missed fraud (financial loss) |
| Per-dataset breakdown | Reveals which fraud types the model handles well/poorly |

---

## Dependencies

| Path | Install |
|------|---------|
| NVIDIA / CUDA | `pip install -r requirements.txt` |
| AMD MI300X / ROCm | `bash 00_rocm_setup.sh` then `pip install -r requirements_rocm.txt` |

Core stack: `transformers >= 4.47`, `peft >= 0.12`, `trl >= 0.11`,
`bitsandbytes` (ROCm fork for MI300X), `vllm >= 0.4`, `deepspeed >= 0.14`,
`accelerate >= 0.34`, `datasets >= 2.21`.
