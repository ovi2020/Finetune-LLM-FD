"""
05_vllm_inference_rocm.py
=========================
vLLM-based inference and serving for the fine-tuned IBM Granite fraud model
on AMD MI300X GPUs via ROCm.

WHAT IS vLLM AND WHY USE IT OVER RAW TRANSFORMERS
──────────────────────────────────────────────────
  - PagedAttention: manages KV-cache as non-contiguous pages → no OOM on long
    sequences, higher batch throughput vs HuggingFace generate()
  - Continuous batching: new requests join in-flight batches without waiting
  - Up to 24× higher throughput than HuggingFace Transformers for batch inference
  - OpenAI-compatible REST API server out-of-the-box

ROCm-SPECIFIC NOTES (MI300X / gfx942)
──────────────────────────────────────
  - Use the official ROCm vLLM Docker image (rocm/vllm:instinct_main) for
    the most reliable setup; it pre-tunes CK kernels for gfx942.
  - VLLM_USE_TRITON_FLASH_ATTN=0  → use Composable Kernel attention (faster)
  - NCCL_MIN_NCHANNELS=112        → optimal RCCL for MI300X 8-GPU topology
  - Disable NUMA auto-balancing on the host before running
  - vLLM needs the MERGED model (not PEFT adapter); run merge first.

PRE-REQUISITE
─────────────
  1. Merge the LoRA adapter into the base model:
       python 04_finetune_granite_rocm.py --merge
     This creates ./checkpoints/granite-fraud-merged/

  2. OR pull the ROCm vLLM Docker image for best performance:
       docker pull rocm/vllm:instinct_main

USAGE
─────
  source rocm_env.sh

  # Mode A – offline batch inference (Python API)
  python 05_vllm_inference_rocm.py --mode offline

  # Mode B – OpenAI-compatible REST API server
  python 05_vllm_inference_rocm.py --mode server

  # Mode C – Benchmark throughput
  python 05_vllm_inference_rocm.py --mode benchmark

  # Single-GPU
  python 05_vllm_inference_rocm.py --mode offline --tp 1

  # 4-GPU tensor parallel (for 70B)
  python 05_vllm_inference_rocm.py --mode server --tp 4 \\
      --model ./checkpoints/granite-fraud-merged-70b
"""

import os
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

# ── ROCm env vars BEFORE any torch / vLLM import ─────────────────────────────
os.environ.setdefault("HIP_VISIBLE_DEVICES",         "0,1,2,3,4,5,6,7")
os.environ.setdefault("HSA_ENABLE_SDMA",             "0")
os.environ.setdefault("HIP_FORCE_DEV_KERNARG",       "1")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
os.environ.setdefault("SAFETENSORS_FAST_GPU",        "1")
os.environ.setdefault("NCCL_MIN_NCHANNELS",          "112")   # MI300X RCCL tuning
os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN",  "0")    # CK > Triton on MI300X
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF",
                      "max_split_size_mb:512,garbage_collection_threshold:0.8")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Path to merged model (output of 04_finetune_granite_rocm.py --merge)
MERGED_MODEL_PATH = "./checkpoints/granite-fraud-merged"

# vLLM server settings
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

# Tensor-parallel size: number of GPUs to shard the model across
# 8B model:  tp=1 (fits in 1× MI300X 192 GB easily)
# 70B model: tp=2 or tp=4
TENSOR_PARALLEL_SIZE = 1

# Fraction of GPU HBM to give vLLM for KV-cache (0.0–1.0)
# MI300X has 192 GB; 0.90 reserves 10% for system / other processes
GPU_MEMORY_UTILIZATION = 0.90

# Max sequence length (prompt + generated tokens)
MAX_MODEL_LEN = 4096

# Max concurrent sequences in a batch
MAX_NUM_SEQS = 256

# Generation defaults
DEFAULT_MAX_TOKENS  = 512
DEFAULT_TEMPERATURE = 0.0   # greedy for deterministic fraud verdicts
DEFAULT_TOP_P       = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT  (identical to training-time prompt)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a financial crime intelligence expert. "
    "Analyze the transaction and respond ONLY in valid JSON format with keys: "
    "verdict (FRAUD|LEGITIMATE), risk_score (1-5), risk_level (LOW|MEDIUM|HIGH), "
    "fraud_type, aml_typology, risk_indicators (list), recommended_action, confidence."
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: format a transaction dict as a user message
# ══════════════════════════════════════════════════════════════════════════════

def format_transaction_prompt(transaction: dict, context: str = "Financial transaction") -> str:
    lines = "\n".join(f"  - {k}: {v}" for k, v in transaction.items())
    return (
        f"Analyze the following transaction for potential fraud or AML risk.\n\n"
        f"Context: {context}\n\n"
        f"Transaction Features:\n{lines}\n\n"
        f"Provide a structured JSON risk assessment."
    )


def build_chat_prompt(user_message: str, tokenizer) -> str:
    """Apply the Granite chat template to produce a formatted prompt string."""
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_message},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODE A: OFFLINE BATCH INFERENCE  (vLLM Python API)
# ══════════════════════════════════════════════════════════════════════════════

def run_offline(model_path: str, tp: int):
    """
    Use the vLLM LLM class for high-throughput offline (batch) inference.
    This is ideal for processing large volumes of historical transactions.

    ROCm-specific settings:
      - dtype="bfloat16"        explicit BF16 for MI300X native path
      - enforce_eager=False     allow CUDA/HIP graph capture for speed
      - disable_log_stats=True  reduce overhead on ROCm
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print("=" * 68)
    print("vLLM OFFLINE BATCH INFERENCE  (ROCm / MI300X)")
    print("=" * 68)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ── Load model into vLLM ─────────────────────────────────────────────────
    print(f"\nLoading model: {model_path}")
    print(f"Tensor parallel: {tp}  |  GPU mem util: {GPU_MEMORY_UTILIZATION}")
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",                        # MI300X native dtype
        tensor_parallel_size=tp,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        trust_remote_code=True,
        enforce_eager=False,                     # HIP graph capture enabled
        disable_log_stats=True,
        # ROCm-specific: use Composable Kernel attention backend
        # (set via env var VLLM_USE_TRITON_FLASH_ATTN=0 above)
    )
    print("Model loaded into vLLM.")

    # ── Sampling parameters ──────────────────────────────────────────────────
    sampling_params = SamplingParams(
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        max_tokens=DEFAULT_MAX_TOKENS,
        stop=["<|eot_id|>", "</s>"],             # Granite EOS tokens
    )

    # ── Sample batch of transactions ─────────────────────────────────────────
    sample_transactions = [
        {
            "amount_scaled":             2.87,
            "type_TRANSFER":             1,
            "balance_diff_orig":        -2.87,
            "balance_diff_dest":         2.85,
            "amount_to_balance_ratio":   0.998,
            "step":                      183,
            "txn_count_24h":             12,
        },
        {
            "V1": -1.36, "V2": -0.07, "V3":  2.54,
            "V4":  1.38, "V14": -0.31, "V17": -0.46,
            "Amount_scaled": 3.21, "Time_scaled": 0.15,
        },
        {
            "amount_paid":       0.42,
            "payment_format":    2,
            "payment_currency":  0,
            "hour_of_day":       2,
            "day_of_week":       4,
            "txn_count_24h":     34,
        },
    ]

    contexts = [
        "Mobile payment system (PaySim)",
        "Credit card transaction (ULB dataset)",
        "Inter-bank wire transfer (IBM AML)",
    ]

    # Build formatted prompts
    prompts = [
        build_chat_prompt(format_transaction_prompt(txn, ctx), tokenizer)
        for txn, ctx in zip(sample_transactions, contexts)
    ]

    # ── Run batch inference ───────────────────────────────────────────────────
    print(f"\nRunning batch inference on {len(prompts)} transactions...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0

    print(f"Inference time: {elapsed:.2f}s  "
          f"({len(prompts)/elapsed:.1f} transactions/sec)\n")

    results = []
    for i, (out, txn, ctx) in enumerate(zip(outputs, sample_transactions, contexts)):
        raw_text = out.outputs[0].text.strip()
        try:
            verdict = json.loads(raw_text)
        except json.JSONDecodeError:
            verdict = {"raw": raw_text, "parse_error": True}

        print(f"Transaction {i+1} [{ctx}]")
        print(f"  Verdict : {verdict.get('verdict', 'PARSE_ERROR')}")
        print(f"  Risk    : {verdict.get('risk_score', '?')}/5  "
              f"({verdict.get('risk_level', '?')})")
        print(f"  Action  : {verdict.get('recommended_action', '?')}")
        print()
        results.append({"transaction": txn, "context": ctx, "result": verdict})

    out_path = "./results/vllm_offline_results.json"
    os.makedirs("./results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE B: OPENAI-COMPATIBLE REST API SERVER
# ══════════════════════════════════════════════════════════════════════════════

def run_server(model_path: str, tp: int):
    """
    Launch a vLLM OpenAI-compatible API server.

    ROCm Docker equivalent:
      docker run --device=/dev/kfd --device=/dev/dri \\
        --security-opt seccomp=unconfined --shm-size 32G --network=host \\
        -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
        -e VLLM_USE_TRITON_FLASH_ATTN=0 \\
        -e NCCL_MIN_NCHANNELS=112 \\
        -v $(pwd)/checkpoints:/models \\
        rocm/vllm:instinct_main \\
        vllm serve /models/granite-fraud-merged \\
          --dtype bfloat16 \\
          --tensor-parallel-size 1 \\
          --gpu-memory-utilization 0.90 \\
          --max-model-len 4096 \\
          --max-num-seqs 256 \\
          --disable-log-requests \\
          --num-scheduler-steps 10 \\
          --host 0.0.0.0 --port 8000

    After the server starts, call it with the Python client below
    or with any OpenAI-compatible tool (curl, LangChain, etc.).
    """
    import subprocess, sys

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                   model_path,
        "--tokenizer",               model_path,
        "--dtype",                   "bfloat16",
        "--tensor-parallel-size",    str(tp),
        "--gpu-memory-utilization",  str(GPU_MEMORY_UTILIZATION),
        "--max-model-len",           str(MAX_MODEL_LEN),
        "--max-num-seqs",            str(MAX_NUM_SEQS),
        "--host",                    SERVER_HOST,
        "--port",                    str(SERVER_PORT),
        "--trust-remote-code",
        "--disable-log-requests",
        "--num-scheduler-steps",     "10",   # MI300X best-practice: 10-15
        "--enable-prefix-caching",           # reuse system-prompt KV cache
    ]
    print("Starting vLLM server...")
    print("  " + " ".join(cmd))
    print(f"\nServer will be available at http://localhost:{SERVER_PORT}")
    print("Send requests to http://localhost:{PORT}/v1/chat/completions\n")
    subprocess.run(cmd)


# ══════════════════════════════════════════════════════════════════════════════
# MODE C: CLIENT  (call the running server)
# ══════════════════════════════════════════════════════════════════════════════

def call_server_api(
    transaction: dict,
    context: str = "Financial transaction",
    base_url: str = f"http://localhost:{SERVER_PORT}/v1",
    model_name: str = MERGED_MODEL_PATH,
) -> dict:
    """
    Call the running vLLM OpenAI-compatible server with a single transaction.
    Uses the standard openai Python client.

    Install:  pip install openai
    """
    from openai import OpenAI

    client = OpenAI(api_key="not-needed", base_url=base_url)

    user_message = format_transaction_prompt(transaction, context)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_message},
        ],
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    raw_text = response.choices[0].message.content.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {"raw_response": raw_text, "parse_error": True}


def demo_server_client():
    """Quick demo: send one transaction to the running server."""
    print("\nCalling vLLM server API...")
    result = call_server_api(
        transaction={
            "amount_scaled":  3.5,
            "type_TRANSFER":  1,
            "balance_diff_orig": -3.5,
            "txn_count_24h":  18,
        },
        context="Mobile payment system (PaySim)",
    )
    print(json.dumps(result, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# MODE D: THROUGHPUT BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(model_path: str, tp: int, n_requests: int = 100):
    """
    Measure tokens/sec throughput on MI300X.
    Generates n_requests synthetic fraud-analysis prompts and times the batch.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print("=" * 68)
    print(f"THROUGHPUT BENCHMARK  ({n_requests} requests, tp={tp})")
    print("=" * 68)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=tp,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        trust_remote_code=True,
        enforce_eager=False,
        disable_log_stats=True,
    )

    import random, numpy as np
    rng = random.Random(42)

    # Generate synthetic prompts
    prompts = []
    for _ in range(n_requests):
        txn = {
            "amount_scaled":           round(rng.gauss(0, 1), 3),
            "type_TRANSFER":           rng.choice([0, 1]),
            "balance_diff_orig":       round(rng.gauss(0, 1), 3),
            "amount_to_balance_ratio": round(abs(rng.gauss(0, 0.5)), 3),
            "step":                    rng.randint(1, 743),
            "txn_count_24h":           rng.randint(1, 50),
        }
        msg = format_transaction_prompt(txn, "Synthetic benchmark transaction")
        prompts.append(build_chat_prompt(msg, tokenizer))

    sp = SamplingParams(temperature=0.0, max_tokens=256, stop=["<|eot_id|>", "</s>"])

    # Warmup
    print("Warming up (10 requests)...")
    llm.generate(prompts[:10], sp)

    # Timed run
    print(f"Running {n_requests} requests...")
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0

    total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    total_in_tokens  = sum(len(tokenizer.encode(p)) for p in prompts)

    print(f"\nResults on AMD MI300X (tp={tp}):")
    print(f"  Requests       : {n_requests}")
    print(f"  Total time     : {elapsed:.2f} s")
    print(f"  Throughput     : {n_requests / elapsed:.1f} req/s")
    print(f"  Input tokens   : {total_in_tokens:,}")
    print(f"  Output tokens  : {total_out_tokens:,}")
    print(f"  Output tok/sec : {total_out_tokens / elapsed:,.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# DOCKER HELPER  (prints the recommended Docker run command)
# ══════════════════════════════════════════════════════════════════════════════

def print_docker_command(model_path: str, tp: int):
    abs_path = os.path.abspath(os.path.dirname(model_path))
    model_dir = os.path.basename(model_path)
    print("\n# ── Recommended ROCm vLLM Docker command ───────────────────────")
    print("# Disable NUMA balancing first (host):")
    print("echo 0 | sudo tee /proc/sys/kernel/numa_balancing\n")
    print("docker run --device=/dev/kfd --device=/dev/dri \\")
    print("  --security-opt seccomp=unconfined --shm-size 32G --network=host \\")
    print(f"  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\")
    print(f"  -e VLLM_USE_TRITON_FLASH_ATTN=0 \\")
    print(f"  -e NCCL_MIN_NCHANNELS=112 \\")
    print(f"  -e HIP_FORCE_DEV_KERNARG=1 \\")
    print(f"  -e SAFETENSORS_FAST_GPU=1 \\")
    print(f"  -v {abs_path}:/models \\")
    print(f"  rocm/vllm:instinct_main \\")
    print(f"  vllm serve /models/{model_dir} \\")
    print(f"    --dtype bfloat16 \\")
    print(f"    --tensor-parallel-size {tp} \\")
    print(f"    --gpu-memory-utilization {GPU_MEMORY_UTILIZATION} \\")
    print(f"    --max-model-len {MAX_MODEL_LEN} \\")
    print(f"    --max-num-seqs {MAX_NUM_SEQS} \\")
    print(f"    --num-scheduler-steps 10 \\")
    print(f"    --enable-prefix-caching \\")
    print(f"    --disable-log-requests \\")
    print(f"    --host 0.0.0.0 --port {SERVER_PORT}")
    print("# ────────────────────────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vLLM inference for fine-tuned Granite on AMD MI300X"
    )
    parser.add_argument("--mode",   default="offline",
                        choices=["offline", "server", "client", "benchmark", "docker"])
    parser.add_argument("--model",  default=MERGED_MODEL_PATH,
                        help="Path to merged model directory")
    parser.add_argument("--tp",     type=int, default=TENSOR_PARALLEL_SIZE,
                        help="Tensor parallel size (GPUs)")
    parser.add_argument("--n",      type=int, default=100,
                        help="Number of requests for benchmark mode")
    args = parser.parse_args()

    if args.mode == "offline":
        run_offline(args.model, args.tp)
    elif args.mode == "server":
        run_server(args.model, args.tp)
    elif args.mode == "client":
        demo_server_client()
    elif args.mode == "benchmark":
        run_benchmark(args.model, args.tp, args.n)
    elif args.mode == "docker":
        print_docker_command(args.model, args.tp)
