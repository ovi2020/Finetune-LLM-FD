"""
04_finetune_granite_rocm.py
===========================
ROCm / AMD MI300X adaptation of 04_finetune_granite.py

Key differences from CUDA version:
  1. bitsandbytes ROCm fork required (gfx942 backend)
  2. flash_attention_2 replaced with SDPA / AOTriton (torch built-in)
  3. paged_adamw_8bit optimizer replaced with adamw_torch (more stable on ROCm)
  4. DeepSpeed ZeRO-3 config provided for 70B multi-GPU on 8× MI300X
  5. HIP environment variables set at startup
  6. BF16 forced (MI300X has no TF32, BF16 is the fast path)
  7. blocksize=64 for 4-bit quantization (required for ROCm CDNA warp-64)
  8. RCCL-aware DDP settings

MI300X specs (per GPU):
  - 192 GB HBM3 memory
  - 5.3 TB/s memory bandwidth
  - gfx942 architecture (CDNA3)
  - 8× MI300X per node = 1.5 TB total HBM

Usage:
  # Single GPU
  source rocm_env.sh && python 04_finetune_granite_rocm.py

  # Multi-GPU (4× MI300X) with DeepSpeed ZeRO-2
  source rocm_env.sh
  accelerate launch --config_file accelerate_rocm_zero2.yaml \\
      04_finetune_granite_rocm.py

  # 70B on 8× MI300X with DeepSpeed ZeRO-3
  source rocm_env.sh
  accelerate launch --config_file accelerate_rocm_zero3.yaml \\
      04_finetune_granite_rocm.py --model ibm-granite/granite-3.3-70b-instruct
"""

import os
import sys
import json
import math
import warnings

warnings.filterwarnings("ignore")

# ── ROCm / HIP environment – set BEFORE importing torch ──────────────────────
# These mirror what rocm_env.sh exports; set here for programmatic control too.
os.environ.setdefault("HIP_VISIBLE_DEVICES",             "0,1,2,3,4,5,6,7")
os.environ.setdefault("HSA_ENABLE_SDMA",                 "0")
os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED",        "1")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF",
                      "max_split_size_mb:512,garbage_collection_threshold:0.8")
os.environ.setdefault("HIPBLASLT_ENABLED",               "1")

import torch
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# ROCm Compatibility Check
# ──────────────────────────────────────────────────────────────────────────────

def check_rocm_environment():
    """Verify we are on ROCm and print hardware summary."""
    print("=" * 70)
    print("ROCm / AMD MI300X ENVIRONMENT CHECK")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No ROCm-capable GPU detected. "
            "Ensure ROCm-enabled PyTorch is installed:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2"
        )

    n_gpus = torch.cuda.device_count()
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  ROCm available  : True")
    print(f"  GPU count       : {n_gpus}")

    for i in range(n_gpus):
        name   = torch.cuda.get_device_name(i)
        mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}           : {name}  ({mem_gb:.0f} GB)")
        if "MI300X" not in name and "MI300" not in name:
            print(f"  NOTE: Expected MI300X, got {name}. "
                  "Settings are tuned for gfx942 CDNA3 architecture.")

    # Check BF16 support (MI300X always supports it)
    bf16_ok = torch.cuda.is_bf16_supported()
    print(f"  BF16 support    : {bf16_ok}")
    if not bf16_ok:
        print("  WARNING: BF16 not supported. Falling back to FP16.")

    print("=" * 70)
    return n_gpus, bf16_ok


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ROCmTrainingConfig:
    # ── Model ─────────────────────────────────────────────────────────────────
    # Use 8B for single / 2-GPU; 70B for 4-8 GPU with ZeRO-3
    model_name: str = "ibm-granite/granite-3.3-8b-instruct"
    model_revision: str = "main"

    # ── Data ──────────────────────────────────────────────────────────────────
    train_jsonl: str = "./data/sft/train.jsonl"
    val_jsonl:   str = "./data/sft/val.jsonl"
    output_dir:  str = "./checkpoints/granite-fraud-rocm"

    # ── ROCm Quantization ─────────────────────────────────────────────────────
    # CHANGE: blocksize=64 is REQUIRED for ROCm CDNA (warp-64) GPUs
    # On CUDA: blocksize defaults to 128 (warp-32); on ROCm MI300X: must be 64
    load_in_4bit: bool   = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"   # BF16 = fast path on MI300X
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_blocksize: int = 64                # ← ROCm CHANGE (was 128 on CUDA)

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_r: int          = 64
    lora_alpha: int      = 128
    lora_dropout: float  = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_bias: str = "none"

    # ── Attention backend ─────────────────────────────────────────────────────
    # CHANGE: "flash_attention_2" → "sdpa" (safe default on ROCm)
    # If ROCm flash-attn is successfully installed: use "flash_attention_2"
    attn_implementation: str = "sdpa"           # ← ROCm CHANGE

    # ── Training ──────────────────────────────────────────────────────────────
    num_train_epochs: int   = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int  = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01

    # CHANGE: "paged_adamw_8bit" → "adamw_torch" for ROCm stability
    # paged_adamw_8bit uses CUDA-specific paging that has issues on ROCm.
    # adamw_torch_fused is the fastest pure-PyTorch option on ROCm.
    optim: str = "adamw_torch_fused"            # ← ROCm CHANGE

    fp16: bool = False
    bf16: bool = True                           # MI300X BF16 native
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3

    # ── Logging & Saving ──────────────────────────────────────────────────────
    logging_steps: int        = 25
    eval_steps: int           = 200
    save_steps: int           = 200
    save_total_limit: int     = 3
    load_best_model_at_end: bool = True
    report_to: str = "tensorboard"

    # ── ROCm-specific ─────────────────────────────────────────────────────────
    # CHANGE: Disable unused param detection (causes hangs with ROCm DDP)
    ddp_find_unused_parameters: bool = False    # ← ROCm CHANGE

    seed: int = 42
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False

    # ── 70B flag (switches to DeepSpeed ZeRO-3 config) ───────────────────────
    use_deepspeed: bool = False                 # set True for 70B multi-GPU


CONFIG = ROCmTrainingConfig()


# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset, DatasetDict


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Loading (unchanged from CUDA version)
# ──────────────────────────────────────────────────────────────────────────────

def load_sft_dataset(cfg: ROCmTrainingConfig) -> DatasetDict:
    ds = load_dataset(
        "json",
        data_files={"train": cfg.train_jsonl, "validation": cfg.val_jsonl},
    )
    print(f"Dataset loaded: {ds}")
    return ds


def apply_chat_template(examples, tokenizer):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}


# ──────────────────────────────────────────────────────────────────────────────
# ROCm-Specific Model Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_tokenizer(cfg: ROCmTrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, revision=cfg.model_revision,
        trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def build_bnb_config_rocm(cfg: ROCmTrainingConfig) -> BitsAndBytesConfig:
    """
    ROCm-specific BitsAndBytesConfig.

    Critical difference: blocksize=64 is required for ROCm CDNA GPUs (MI300X).
    CDNA3 uses 64-thread wavefronts (vs CUDA's 32-thread warps).
    Using blocksize=128 (CUDA default) causes incorrect quantization on gfx942.

    Ref: https://github.com/bitsandbytes-foundation/bitsandbytes/releases
         PR #1856: "Add blocksize=64 4-bit quantization support for ROCm CDNA"
    """
    compute_dtype = (
        torch.bfloat16 if cfg.bnb_4bit_compute_dtype == "bfloat16"
        else torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_blocksize=cfg.bnb_4bit_blocksize,     # ← 64 for ROCm
    )


def load_quantized_model_rocm(cfg: ROCmTrainingConfig):
    """
    Load IBM Granite in 4-bit NF4 for ROCm.

    ROCm Changes vs CUDA version:
      - attn_implementation: "sdpa" instead of "flash_attention_2"
      - BnB config uses blocksize=64
      - No torch.compile() (Triton backend on ROCm has lower stability for training)
    """
    bnb_config = build_bnb_config_rocm(cfg)

    # Check if ROCm flash-attn is available
    attn_impl = cfg.attn_implementation
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa
            print("  flash_attention_2 available (ROCm CK backend)")
        except ImportError:
            attn_impl = "sdpa"
            print("  flash_attn not available → falling back to sdpa (safe)")

    compute_dtype = (
        torch.bfloat16 if cfg.bnb_4bit_compute_dtype == "bfloat16"
        else torch.float16
    )

    print(f"  Loading model  : {cfg.model_name}")
    print(f"  Attention impl : {attn_impl}")
    print(f"  Compute dtype  : {compute_dtype}")
    print(f"  BnB blocksize  : {cfg.bnb_4bit_blocksize} (ROCm CDNA3 warp-64)")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        revision=cfg.model_revision,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        attn_implementation=attn_impl,      # ← sdpa on ROCm
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.gradient_checkpointing,
    )
    model.config.use_cache = False

    # ROCm-specific: enable TunableOp for GEMM kernel auto-tuning
    # This runs a brief warmup but finds optimal hipBLASLt GEMM kernels
    if os.environ.get("PYTORCH_TUNABLEOP_ENABLED", "0") == "1":
        print("  PyTorch TunableOp enabled (GEMM auto-tuning, ~1-2 min warmup)")

    return model


def attach_lora(model, cfg: ROCmTrainingConfig):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias=cfg.lora_bias,
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)

    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"  Trainable params: {trainable:,}  ({100*trainable/total:.2f}%)")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# ROCm Training Arguments
# ──────────────────────────────────────────────────────────────────────────────

def build_training_arguments_rocm(cfg: ROCmTrainingConfig) -> TrainingArguments:
    """
    ROCm-specific TrainingArguments.

    Key changes vs CUDA:
      - optim: adamw_torch_fused  (paged_adamw_8bit unstable on ROCm)
      - ddp_find_unused_parameters: False (prevents ROCm DDP hangs)
      - deepspeed: ZeRO-2 or ZeRO-3 config path (for multi-GPU)
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    deepspeed_cfg = None
    if cfg.use_deepspeed:
        # For 70B: use ZeRO-3; for 8B multi-GPU: ZeRO-2 is sufficient
        if "70b" in cfg.model_name.lower():
            deepspeed_cfg = "./ds_zero3_rocm.json"
        else:
            deepspeed_cfg = "./ds_zero2_rocm.json"
        print(f"  DeepSpeed config: {deepspeed_cfg}")

    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        optim=cfg.optim,                                 # ← adamw_torch_fused
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=cfg.report_to,
        seed=cfg.seed,
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=cfg.remove_unused_columns,
        group_by_length=True,
        ddp_find_unused_parameters=cfg.ddp_find_unused_parameters,  # ← False
        deepspeed=deepspeed_cfg,
        run_name="granite-fraud-rocm",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: ROCmTrainingConfig = CONFIG):
    print("=" * 70)
    print("IBM GRANITE QLoRA FINE-TUNING — AMD MI300X / ROCm")
    print("=" * 70)

    n_gpus, bf16_ok = check_rocm_environment()

    # If BF16 unavailable (very old ROCm), fall back to FP16
    if not bf16_ok:
        cfg.bf16 = False
        cfg.fp16 = True
        cfg.bnb_4bit_compute_dtype = "float16"
        print("  Switched to FP16 (BF16 not supported on this GPU)")

    # Auto-enable DeepSpeed for 70B
    if "70b" in cfg.model_name.lower() and not cfg.use_deepspeed:
        cfg.use_deepspeed = True
        print(f"  Auto-enabled DeepSpeed ZeRO-3 for 70B model on {n_gpus} GPUs")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("\n[1/5] Loading tokenizer...")
    tokenizer = load_tokenizer(cfg)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\n[2/5] Loading and formatting dataset...")
    raw_ds      = load_sft_dataset(cfg)
    formatted_ds = raw_ds.map(
        lambda ex: apply_chat_template(ex, tokenizer),
        batched=True,
        remove_columns=raw_ds["train"].column_names,
        desc="Applying Granite chat template",
    )
    print(f"  Train: {len(formatted_ds['train']):,} | Val: {len(formatted_ds['validation']):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[3/5] Loading quantized model (ROCm QLoRA)...")
    model = load_quantized_model_rocm(cfg)
    model = attach_lora(model, cfg)

    # ── Training args ─────────────────────────────────────────────────────────
    print("\n[4/5] Configuring training arguments (ROCm)...")
    training_args = build_training_arguments_rocm(cfg)

    # Response-only loss masking (same as CUDA version)
    response_template = "<|start_header_id|>assistant<|end_header_id|>"
    try:
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
            mlm=False,
        )
    except Exception:
        collator = None

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[5/5] Starting training on MI300X...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_ds["train"],
        eval_dataset=formatted_ds["validation"],
        args=training_args,
        data_collator=collator,
        max_seq_length=cfg.max_seq_length,
        dataset_text_field="text",
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    adapter_path = os.path.join(cfg.output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nLoRA adapter saved: {adapter_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=CONFIG.model_name)
    parser.add_argument("--epochs",     type=int,   default=CONFIG.num_train_epochs)
    parser.add_argument("--lr",         type=float, default=CONFIG.learning_rate)
    parser.add_argument("--lora_r",     type=int,   default=CONFIG.lora_r)
    parser.add_argument("--output_dir", default=CONFIG.output_dir)
    parser.add_argument("--attn",       default="sdpa",
                        choices=["sdpa", "flash_attention_2"],
                        help="Attention backend. Use sdpa (safe) or flash_attention_2 "
                             "if ROCm CK flash-attn is installed.")
    parser.add_argument("--deepspeed",  action="store_true",
                        help="Enable DeepSpeed (required for 70B)")
    args = parser.parse_args()

    CONFIG.model_name         = args.model
    CONFIG.num_train_epochs   = args.epochs
    CONFIG.learning_rate      = args.lr
    CONFIG.lora_r             = args.lora_r
    CONFIG.lora_alpha         = args.lora_r * 2
    CONFIG.output_dir         = args.output_dir
    CONFIG.attn_implementation = args.attn
    CONFIG.use_deepspeed      = args.deepspeed

    train(CONFIG)
