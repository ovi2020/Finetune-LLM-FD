"""
04_finetune_granite.py
======================
QLoRA fine-tuning of IBM Granite for fraud detection & financial crime intelligence.

Supports:
  - ibm-granite/granite-3.3-8b-instruct   (recommended: 2× A100 40GB)
  - ibm-granite/granite-3.1-8b-instruct   (alternative 8B)
  - ibm-granite/granite-3.3-70b-instruct  (requires 4× A100 80GB + DeepSpeed ZeRO-3)

Technique: QLoRA (4-bit NF4 quantization + LoRA adapters)
Framework: HuggingFace Transformers + PEFT + TRL SFTTrainer + BitsAndBytes
"""

import os
import json
import math
import torch
import warnings
from dataclasses import dataclass, field
from typing import Optional, List

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # ── Model ─────────────────────────────────────────────────────────────────
    # 8B recommended. Change to "ibm-granite/granite-3.3-70b-instruct" for 70B
    model_name: str = "ibm-granite/granite-3.3-8b-instruct"
    model_revision: str = "main"

    # ── Data ──────────────────────────────────────────────────────────────────
    train_jsonl: str = "./data/sft/train.jsonl"
    val_jsonl:   str = "./data/sft/val.jsonl"
    output_dir:  str = "./checkpoints/granite-fraud-qlora"

    # ── QLoRA / Quantization ──────────────────────────────────────────────────
    load_in_4bit: bool   = True
    bnb_4bit_quant_type: str = "nf4"           # nf4 or fp4
    bnb_4bit_compute_dtype: str = "bfloat16"   # bfloat16 or float16
    bnb_4bit_use_double_quant: bool = True

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_r: int          = 64       # rank (16 for quick test, 64-128 for production)
    lora_alpha: int      = 128      # typically 2× lora_r
    lora_dropout: float  = 0.05
    # Target the attention + MLP projection layers (covers Granite's architecture)
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_bias: str = "none"

    # ── Training ──────────────────────────────────────────────────────────────
    num_train_epochs: int   = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int  = 2
    gradient_accumulation_steps: int = 8     # effective batch = 2×8 = 16
    max_seq_length: int = 2048
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    optim: str = "paged_adamw_8bit"           # memory-efficient optimizer
    fp16: bool = False
    bf16: bool = True                         # A100/H100; use fp16 for V100
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3

    # ── Logging & Saving ──────────────────────────────────────────────────────
    logging_steps: int        = 25
    eval_steps: int           = 200
    save_steps: int           = 200
    save_total_limit: int     = 3
    load_best_model_at_end: bool = True
    report_to: str = "tensorboard"            # "wandb" if W&B is configured

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = 42
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False


CONFIG = TrainingConfig()


# ──────────────────────────────────────────────────────────────────────────────
# Imports (after config so they can be conditional)
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
import bitsandbytes as bnb


# ──────────────────────────────────────────────────────────────────────────────
# Load & Format Dataset
# ──────────────────────────────────────────────────────────────────────────────

def load_sft_dataset(cfg: TrainingConfig) -> DatasetDict:
    """Load JSONL files into HuggingFace Dataset with chat template applied."""
    ds = load_dataset(
        "json",
        data_files={
            "train": cfg.train_jsonl,
            "validation": cfg.val_jsonl,
        },
    )
    print(f"Loaded dataset: {ds}")
    return ds


def apply_chat_template(examples, tokenizer, cfg: TrainingConfig):
    """
    Apply Granite's chat template to the messages list.
    Produces a single 'text' column containing the full formatted conversation.
    """
    texts = []
    for messages in examples["messages"]:
        # apply_chat_template handles system/user/assistant formatting for Granite
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,   # False for training (include full response)
        )
        texts.append(text)
    return {"text": texts}


# ──────────────────────────────────────────────────────────────────────────────
# Model & Tokenizer Setup
# ──────────────────────────────────────────────────────────────────────────────

def load_tokenizer(cfg: TrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        revision=cfg.model_revision,
        trust_remote_code=True,
        padding_side="right",   # right-padding for decoder-only training
    )
    # Granite tokenizer may not have a pad token – use EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_quantized_model(cfg: TrainingConfig):
    """Load IBM Granite in 4-bit NF4 quantization (QLoRA)."""
    compute_dtype = (
        torch.bfloat16 if cfg.bnb_4bit_compute_dtype == "bfloat16"
        else torch.float16
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        revision=cfg.model_revision,
        quantization_config=bnb_config,
        device_map="auto",          # automatically shard across GPUs
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2",  # faster attention (A100+)
    )

    # Prepare for k-bit (QLoRA) training: cast norms to fp32, disable cache
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.gradient_checkpointing,
    )
    model.config.use_cache = False   # required for gradient checkpointing

    print(f"Model loaded: {cfg.model_name}")
    print(f"Model dtype : {model.dtype}")
    print(f"Device map  : {model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}")
    return model


def attach_lora(model, cfg: TrainingConfig):
    """Attach LoRA adapters to the quantized base model."""
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

    # Print trainable parameter summary
    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100 * trainable / total
    print(f"\nLoRA adapter attached:")
    print(f"  Trainable params : {trainable:,}  ({pct:.2f}% of total)")
    print(f"  Total params     : {total:,}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def build_training_arguments(cfg: TrainingConfig) -> TrainingArguments:
    os.makedirs(cfg.output_dir, exist_ok=True)
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
        optim=cfg.optim,
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
        group_by_length=True,       # pack similar-length sequences → efficiency
        ddp_find_unused_parameters=False,
        run_name="granite-fraud-qlora",
    )


def train(cfg: TrainingConfig = CONFIG):
    print("=" * 70)
    print("IBM GRANITE QLoRA FINE-TUNING: FRAUD DETECTION")
    print("=" * 70)

    # ── Load tokenizer ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading tokenizer...")
    tokenizer = load_tokenizer(cfg)

    # ── Load & format dataset ──────────────────────────────────────────────────
    print("\n[2/5] Loading and formatting dataset...")
    raw_ds = load_sft_dataset(cfg)
    formatted_ds = raw_ds.map(
        lambda ex: apply_chat_template(ex, tokenizer, cfg),
        batched=True,
        remove_columns=raw_ds["train"].column_names,
        desc="Applying chat template",
    )
    print(f"  Train: {len(formatted_ds['train']):,} examples")
    print(f"  Val  : {len(formatted_ds['validation']):,} examples")

    # Quick sanity check
    sample = formatted_ds["train"][0]["text"]
    token_len = len(tokenizer.encode(sample))
    print(f"  Sample token length: {token_len}")
    if token_len > cfg.max_seq_length:
        print(f"  WARNING: Sample exceeds max_seq_length={cfg.max_seq_length}. "
              "Consider reducing feature verbosity in 03_instruction_dataset.py")

    # ── Load quantized model ───────────────────────────────────────────────────
    print("\n[3/5] Loading quantized model (QLoRA)...")
    model = load_quantized_model(cfg)
    model = attach_lora(model, cfg)

    # ── Build training args ────────────────────────────────────────────────────
    print("\n[4/5] Configuring training arguments...")
    training_args = build_training_arguments(cfg)

    # Response-only masking: only compute loss on assistant turns
    # Find the assistant header token(s) for Granite
    # Granite uses the same format as LLaMA-3: <|start_header_id|>assistant<|end_header_id|>
    response_template = "<|start_header_id|>assistant<|end_header_id|>"
    try:
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
            mlm=False,
        )
    except Exception:
        # Fall back to standard collator if template not found
        collator = None
        print("  Note: Completion-only collator not configured. "
              "Full sequence loss will be used.")

    # ── Train ──────────────────────────────────────────────────────────────────
    print("\n[5/5] Starting training...")
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

    # ── Save adapter ───────────────────────────────────────────────────────────
    adapter_path = os.path.join(cfg.output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nLoRA adapter saved to: {adapter_path}")

    # Optional: merge adapter into base model (for deployment)
    print("\nTo merge adapter into base model for deployment, run:")
    print(f"  python -c \"from peft import PeftModel; "
          f"from transformers import AutoModelForCausalLM; ...")


# ──────────────────────────────────────────────────────────────────────────────
# Merge & Export (run after training)
# ──────────────────────────────────────────────────────────────────────────────

def merge_and_save(
    base_model_name: str = CONFIG.model_name,
    adapter_path: str = "./checkpoints/granite-fraud-qlora/final_adapter",
    merged_output: str = "./checkpoints/granite-fraud-merged",
):
    """
    Merge LoRA adapters into the base model weights.
    Produces a standalone model that can be loaded without PEFT.
    Requires enough CPU/GPU RAM to hold the full unquantized model.
    """
    print(f"Loading base model: {base_model_name}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    from peft import PeftModel
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)
    print("Merging and unloading adapter weights...")
    model = model.merge_and_unload()
    model.save_pretrained(merged_output)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(merged_output)
    print(f"Merged model saved to: {merged_output}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune IBM Granite for fraud detection")
    parser.add_argument("--model", default=CONFIG.model_name,
                        help="HuggingFace model ID for IBM Granite")
    parser.add_argument("--epochs", type=int, default=CONFIG.num_train_epochs)
    parser.add_argument("--lr", type=float, default=CONFIG.learning_rate)
    parser.add_argument("--lora_r", type=int, default=CONFIG.lora_r)
    parser.add_argument("--output_dir", default=CONFIG.output_dir)
    parser.add_argument("--merge", action="store_true",
                        help="Merge adapter into base model after training")
    args = parser.parse_args()

    CONFIG.model_name       = args.model
    CONFIG.num_train_epochs = args.epochs
    CONFIG.learning_rate    = args.lr
    CONFIG.lora_r           = args.lora_r
    CONFIG.lora_alpha       = args.lora_r * 2
    CONFIG.output_dir       = args.output_dir

    train(CONFIG)

    if args.merge:
        merge_and_save(
            base_model_name=CONFIG.model_name,
            adapter_path=os.path.join(CONFIG.output_dir, "final_adapter"),
        )
