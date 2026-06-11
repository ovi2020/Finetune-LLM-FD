"""
04_finetune_granite_rocm.py
===========================
ROCm / AMD MI300X adaptation of 04_finetune_granite.py for QLoRA fine-tuning
of IBM Granite on fraud detection & financial crime datasets.

WHAT CHANGED VS THE CUDA VERSION (04_finetune_granite.py)
──────────────────────────────────────────────────────────
| # | Setting / Line                  | CUDA value            | ROCm value             | Why                                          |
|---|----------------------------------|----------------------|------------------------|----------------------------------------------|
| 1 | bnb_4bit_blocksize               | 128 (default)        | 64                     | MI300X wavefront = 64 threads (warp-64)      |
| 2 | attn_implementation              | flash_attention_2    | sdpa (or ck_flash)     | std flash-attn pkg won't build on ROCm       |
| 3 | optim                            | paged_adamw_8bit     | adamw_torch_fused      | paged optimizer uses CUDA paging primitives  |
| 4 | ddp_find_unused_parameters       | False                | False (explicit)        | ROCm DDP can hang with True                  |
| 5 | HIP env vars                     | —                    | set at module load      | HSA_ENABLE_SDMA, PYTORCH_TUNABLEOP, etc.     |
| 6 | DeepSpeed config                 | optional             | recommended for 70B    | ZeRO-3 + RCCL needed at scale               |
| 7 | bf16 / fp16                      | bf16=True            | bf16=True (confirmed)  | MI300X has native BF16 support               |
| 8 | torch_dtype                      | auto / bfloat16      | bfloat16 (explicit)    | avoids fp32 fallback on ROCm                 |

GPU REQUIREMENTS
────────────────
  8B  model: 1× MI300X (192 GB)  — QLoRA fits easily
  70B model: 2× MI300X (384 GB)  — QLoRA + DeepSpeed ZeRO-2

USAGE
─────
  # Prerequisite
  source rocm_env.sh

  # Single GPU, 8B model
  python 04_finetune_granite_rocm.py

  # Multi-GPU, 8B model with ZeRO-2
  accelerate launch --config_file accelerate_rocm_zero2.yaml \\
      04_finetune_granite_rocm.py

  # 70B model on 4+ GPUs with ZeRO-3
  accelerate launch --config_file accelerate_rocm_zero3.yaml \\
      04_finetune_granite_rocm.py --model ibm-granite/granite-3.3-70b-instruct
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

# ── Set ROCm / HIP env vars BEFORE importing torch ────────────────────────────
# (These duplicate rocm_env.sh so the script is self-contained.)
os.environ.setdefault("HIP_VISIBLE_DEVICES",             "0,1,2,3,4,5,6,7")
os.environ.setdefault("HSA_ENABLE_SDMA",                 "0")
os.environ.setdefault("HIP_FORCE_DEV_KERNARG",           "1")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT",     "1")
os.environ.setdefault("SAFETENSORS_FAST_GPU",            "1")
os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED",       "1")
os.environ.setdefault("PYTORCH_TUNABLEOP_TUNING",        "1")
os.environ.setdefault("NCCL_MIN_NCHANNELS",              "112")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF",
                      "max_split_size_mb:512,garbage_collection_threshold:0.8")
os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN",      "0")  # use CK backend

import torch
from dataclasses import dataclass, field
from typing import List


# ══════════════════════════════════════════════════════════════════════════════
# 1. ENVIRONMENT CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_rocm():
    """Print GPU summary and return (n_gpus, bf16_supported)."""
    print("=" * 68)
    print("AMD MI300X / ROCm ENVIRONMENT")
    print("=" * 68)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No ROCm GPU found. Install ROCm-enabled PyTorch:\n"
            "  pip install torch --index-url "
            "https://download.pytorch.org/whl/rocm6.2"
        )
    n = torch.cuda.device_count()
    for i in range(n):
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({mem:.0f} GB)")
    bf16 = torch.cuda.is_bf16_supported()
    print(f"  BF16: {bf16}  |  GPUs: {n}  |  PyTorch: {torch.__version__}")
    print("=" * 68)
    return n, bf16


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION  (diff-marked with  # ROCm CHANGE)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ROCmConfig:
    # Model ─────────────────────────────────────────────────────────────────
    model_name: str = "ibm-granite/granite-3.3-8b-instruct"

    # Data ──────────────────────────────────────────────────────────────────
    train_jsonl: str = "./data/sft/train.jsonl"
    val_jsonl:   str = "./data/sft/val.jsonl"
    output_dir:  str = "./checkpoints/granite-fraud-rocm"

    # Quantisation ──────────────────────────────────────────────────────────
    load_in_4bit:              bool  = True
    bnb_4bit_quant_type:       str   = "nf4"
    bnb_4bit_compute_dtype:    str   = "bfloat16"
    bnb_4bit_use_double_quant: bool  = True
    bnb_4bit_blocksize:        int   = 64    # ROCm CHANGE: 128→64  (CDNA3 warp-64)

    # Attention ─────────────────────────────────────────────────────────────
    attn_implementation: str = "sdpa"        # ROCm CHANGE: flash_attention_2→sdpa
    # Set to "flash_attention_2" only if ROCm CK flash-attn is installed.

    # LoRA ──────────────────────────────────────────────────────────────────
    lora_r:              int   = 64
    lora_alpha:          int   = 128
    lora_dropout:        float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training ──────────────────────────────────────────────────────────────
    num_train_epochs:            int   = 3
    per_device_train_batch_size: int   = 2
    per_device_eval_batch_size:  int   = 2
    gradient_accumulation_steps: int   = 8
    max_seq_length:              int   = 2048
    learning_rate:               float = 2e-4
    lr_scheduler_type:           str   = "cosine"
    warmup_ratio:                float = 0.05
    weight_decay:                float = 0.01
    optim:                       str   = "adamw_torch_fused"  # ROCm CHANGE: paged_adamw_8bit→adamw_torch_fused
    fp16:                        bool  = False
    bf16:                        bool  = True
    gradient_checkpointing:      bool  = True
    max_grad_norm:               float = 0.3
    ddp_find_unused_parameters:  bool  = False  # ROCm CHANGE: explicit False prevents DDP hangs

    # Logging ───────────────────────────────────────────────────────────────
    logging_steps:           int  = 25
    eval_steps:              int  = 200
    save_steps:              int  = 200
    save_total_limit:        int  = 3
    load_best_model_at_end:  bool = True
    report_to:               str  = "tensorboard"

    # Multi-GPU / DeepSpeed ─────────────────────────────────────────────────
    use_deepspeed: bool = False  # auto-enabled for 70B
    seed:          int  = 42
    dataloader_num_workers: int = 4


CFG = ROCmConfig()


# ══════════════════════════════════════════════════════════════════════════════
# 3. IMPORTS  (after env vars are set)
# ══════════════════════════════════════════════════════════════════════════════

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, BitsAndBytesConfig, EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATASET  (unchanged from CUDA version)
# ══════════════════════════════════════════════════════════════════════════════

def load_and_format_dataset(cfg: ROCmConfig, tokenizer):
    ds = load_dataset("json", data_files={
        "train":      cfg.train_jsonl,
        "validation": cfg.val_jsonl,
    })
    def _apply_template(examples):
        return {"text": [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in examples["messages"]
        ]}
    ds = ds.map(_apply_template, batched=True,
                remove_columns=ds["train"].column_names,
                desc="Chat template")
    print(f"  Train: {len(ds['train']):,}  |  Val: {len(ds['validation']):,}")
    return ds


# ══════════════════════════════════════════════════════════════════════════════
# 5. MODEL LOADING  (ROCm-specific changes annotated)
# ══════════════════════════════════════════════════════════════════════════════

def load_tokenizer(cfg: ROCmConfig):
    tok = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=True, padding_side="right"
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_bnb_config(cfg: ROCmConfig) -> BitsAndBytesConfig:
    """
    ROCm CHANGE: blocksize=64 is mandatory for CDNA3 (MI300X / gfx942).

    The MI300X wavefront is 64 threads wide.  bitsandbytes' default
    blocksize=128 was designed for CUDA warp-32.  Using blocksize=128 on
    ROCm produces incorrect dequantisation and accuracy loss.

    bitsandbytes PR #1856 adds the blocksize parameter and sets 64 as the
    ROCm default, but specifying it explicitly makes the intent clear.
    """
    compute_dtype = torch.bfloat16 if cfg.bnb_4bit_compute_dtype == "bfloat16" \
                    else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_blocksize=cfg.bnb_4bit_blocksize,   # ← ROCm CHANGE
    )


def load_model(cfg: ROCmConfig):
    """
    ROCm CHANGES in this function:
      - attn_implementation = "sdpa"  instead of "flash_attention_2"
        PyTorch SDPA on ROCm dispatches to AOTriton / Composable Kernel
        automatically; no separate install needed.
        Use "flash_attention_2" only if you have separately installed
        ROCm/flash-attention (CK build for gfx942).
      - torch_dtype is set explicitly to bfloat16 to avoid fp32 fallback.
    """
    bnb_cfg = build_bnb_config(cfg)
    compute_dtype = torch.bfloat16 if cfg.bnb_4bit_compute_dtype == "bfloat16" \
                    else torch.float16

    # Check if ROCm flash-attn is actually available; fall back gracefully
    attn_impl = cfg.attn_implementation
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa
            print("  Attention: flash_attention_2 (ROCm CK backend)")
        except ImportError:
            attn_impl = "sdpa"
            print("  flash-attn not installed → falling back to sdpa (safe)")
    else:
        print("  Attention: sdpa (PyTorch built-in, dispatches to AOTriton/CK)")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,       # explicit bfloat16
        attn_implementation=attn_impl,   # ROCm CHANGE
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.gradient_checkpointing
    )
    model.config.use_cache = False
    return model


def attach_lora(model, cfg: ROCmConfig):
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none", inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  LoRA trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING ARGUMENTS  (ROCm-specific changes annotated)
# ══════════════════════════════════════════════════════════════════════════════

def build_training_args(cfg: ROCmConfig) -> TrainingArguments:
    """
    ROCm CHANGES:
      optim="adamw_torch_fused"
        paged_adamw_8bit uses CUDA-specific paging primitives that are not
        available in ROCm's HIP runtime.  adamw_torch_fused is the fastest
        pure-PyTorch option and is fully ROCm-compatible.

      ddp_find_unused_parameters=False
        With ROCm DDP, setting this to True can cause inter-process hangs
        during gradient synchronisation. Always keep False.

      deepspeed (optional)
        For 70B or aggressive multi-GPU use, point to one of the JSON configs
        generated by this script (ds_zero2_rocm.json / ds_zero3_rocm.json).
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    ds_cfg = None
    if cfg.use_deepspeed:
        ds_cfg = ("./ds_zero3_rocm.json"
                  if "70b" in cfg.model_name.lower()
                  else "./ds_zero2_rocm.json")
        print(f"  DeepSpeed config: {ds_cfg}")

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
        optim=cfg.optim,                            # ROCm CHANGE
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
        remove_unused_columns=False,
        group_by_length=True,
        ddp_find_unused_parameters=cfg.ddp_find_unused_parameters,  # ROCm CHANGE
        deepspeed=ds_cfg,
        run_name="granite-fraud-rocm",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7. DEEPSPEED CONFIG GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def write_deepspeed_configs():
    """Write ZeRO-2 and ZeRO-3 JSON configs optimised for MI300X + RCCL."""

    zero2 = {
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 0.3,
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "steps_per_print": 50,
        "wall_clock_breakdown": False,
        # RCCL tuning for MI300X fully-connected topology
        "comms_logger": {"enabled": False},
    }

    zero3 = {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "none"},
            "offload_param":     {"device": "none"},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 0.3,
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "steps_per_print": 50,
        "wall_clock_breakdown": False,
    }

    for name, cfg_obj in [("ds_zero2_rocm.json", zero2),
                           ("ds_zero3_rocm.json", zero3)]:
        with open(name, "w") as f:
            json.dump(cfg_obj, f, indent=2)
        print(f"  Wrote {name}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MERGE ADAPTER (post-training, required before vLLM)
# ══════════════════════════════════════════════════════════════════════════════

def merge_adapter(
    base_model_name: str,
    adapter_path: str,
    merged_output: str = "./checkpoints/granite-fraud-merged",
):
    """
    Merge LoRA adapter into the base model and save as a standalone model.
    This is REQUIRED before loading with vLLM (vLLM does not load PEFT adapters
    directly; it needs merged full-precision weights in safetensors format).
    """
    from peft import PeftModel
    print(f"\nMerging adapter into base model...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, device_map="cpu",
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    os.makedirs(merged_output, exist_ok=True)
    model.save_pretrained(merged_output, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(adapter_path)
    tok.save_pretrained(merged_output)
    print(f"Merged model saved → {merged_output}")
    print("You can now run: python 05_vllm_inference_rocm.py")


# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(cfg: ROCmConfig = CFG):
    print("=" * 68)
    print("IBM GRANITE QLoRA — AMD MI300X / ROCm")
    print("=" * 68)

    n_gpus, bf16_ok = check_rocm()
    if not bf16_ok:
        cfg.bf16 = False
        cfg.fp16 = True
        cfg.bnb_4bit_compute_dtype = "float16"
        print("  BF16 unavailable → switched to FP16")
    if "70b" in cfg.model_name.lower() and not cfg.use_deepspeed:
        cfg.use_deepspeed = True
        print(f"  Auto-enabled DeepSpeed ZeRO-3 for 70B on {n_gpus} GPUs")
        write_deepspeed_configs()

    print("\n[1/5] Tokenizer...")
    tokenizer = load_tokenizer(cfg)

    print("\n[2/5] Dataset...")
    ds = load_and_format_dataset(cfg, tokenizer)

    print("\n[3/5] Model (ROCm QLoRA)...")
    model = load_model(cfg)
    model = attach_lora(model, cfg)

    print("\n[4/5] Training arguments...")
    training_args = build_training_args(cfg)

    # Response-only loss masking (same as CUDA version)
    response_template = "<|start_header_id|>assistant<|end_header_id|>"
    try:
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template, tokenizer=tokenizer, mlm=False,
        )
    except Exception:
        collator = None

    print("\n[5/5] Training on MI300X...")
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        args=training_args, data_collator=collator,
        max_seq_length=cfg.max_seq_length, dataset_text_field="text",
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    adapter_path = os.path.join(cfg.output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nAdapter saved → {adapter_path}")
    print("Next: merge adapter for vLLM, or run: python 05_vllm_inference_rocm.py")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default=CFG.model_name)
    p.add_argument("--epochs",     type=int,   default=CFG.num_train_epochs)
    p.add_argument("--lr",         type=float, default=CFG.learning_rate)
    p.add_argument("--lora_r",     type=int,   default=CFG.lora_r)
    p.add_argument("--output_dir", default=CFG.output_dir)
    p.add_argument("--attn",       default="sdpa",
                   choices=["sdpa", "flash_attention_2"])
    p.add_argument("--deepspeed",  action="store_true")
    p.add_argument("--merge",      action="store_true",
                   help="Merge adapter after training (needed for vLLM)")
    a = p.parse_args()

    CFG.model_name          = a.model
    CFG.num_train_epochs    = a.epochs
    CFG.learning_rate       = a.lr
    CFG.lora_r              = a.lora_r
    CFG.lora_alpha          = a.lora_r * 2
    CFG.output_dir          = a.output_dir
    CFG.attn_implementation = a.attn
    CFG.use_deepspeed       = a.deepspeed

    train(CFG)

    if a.merge:
        merge_adapter(
            base_model_name=CFG.model_name,
            adapter_path=os.path.join(CFG.output_dir, "final_adapter"),
        )
