"""
05_evaluate_model.py
====================
Evaluation and inference for the fine-tuned IBM Granite fraud detection model.

Covers:
  - Batch evaluation on held-out test JSONL
  - Metrics: F1, Precision, Recall, MCC, AUROC
  - Confusion matrix and classification report
  - Per-dataset performance breakdown
  - Interactive single-transaction inference function
"""

import os
import json
import re
import torch
import numpy as np
import pandas as pd
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BASE_MODEL_NAME = "ibm-granite/granite-3.3-8b-instruct"
ADAPTER_PATH    = "./checkpoints/granite-fraud-qlora/final_adapter"
MERGED_PATH     = "./checkpoints/granite-fraud-merged"    # if merged
TEST_JSONL      = "./data/sft/test.jsonl"
RESULTS_DIR     = "./results"
MAX_NEW_TOKENS  = 512
BATCH_SIZE      = 4

os.makedirs(RESULTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model_for_inference(
    use_merged: bool = False,
    use_8bit: bool = True,
):
    """
    Load the fine-tuned model for inference.
    use_merged: load the merged (adapter + base) model
    use_8bit:   load in 8-bit for inference (saves memory, slightly slower)
    """
    model_path = MERGED_PATH if use_merged else BASE_MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH if not use_merged else MERGED_PATH,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if use_8bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    if not use_merged:
        print(f"Loading LoRA adapter: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model = model.merge_and_unload()  # merge for faster inference

    model.eval()
    return tokenizer, model


# ──────────────────────────────────────────────────────────────────────────────
# Inference Utilities
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a financial crime intelligence expert. "
    "Analyze the transaction and respond ONLY in valid JSON format with keys: "
    "verdict (FRAUD|LEGITIMATE), risk_score (1-5), risk_level (LOW|MEDIUM|HIGH), "
    "fraud_type, aml_typology, risk_indicators (list), recommended_action, confidence."
)


def build_inference_prompt(tokenizer, user_message: str) -> str:
    """Format a single inference prompt using Granite's chat template."""
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_message},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   # True for inference
    )


def generate_response(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate a single response from the model."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy for deterministic evaluation
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (exclude the prompt)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_verdict(response_text: str) -> Optional[int]:
    """
    Extract binary fraud label from JSON response.
    Returns 1 (fraud) or 0 (legit), or None if parsing fails.
    """
    try:
        # Try direct JSON parse
        data = json.loads(response_text)
        verdict = data.get("verdict", "").upper()
        return 1 if verdict == "FRAUD" else 0
    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction
    match = re.search(r'"verdict"\s*:\s*"(FRAUD|LEGITIMATE)"', response_text, re.IGNORECASE)
    if match:
        return 1 if match.group(1).upper() == "FRAUD" else 0

    # Last resort: keyword search
    if "FRAUD" in response_text.upper() and "LEGITIMATE" not in response_text.upper():
        return 1
    if "LEGITIMATE" in response_text.upper():
        return 0

    return None   # parsing failed


def parse_risk_score(response_text: str) -> Optional[int]:
    """Extract risk_score (1-5) from response."""
    try:
        data = json.loads(response_text)
        return int(data.get("risk_score", -1))
    except Exception:
        match = re.search(r'"risk_score"\s*:\s*([1-5])', response_text)
        return int(match.group(1)) if match else None


# ──────────────────────────────────────────────────────────────────────────────
# Batch Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_on_test_set(
    tokenizer,
    model,
    test_jsonl: str = TEST_JSONL,
    max_samples: int = 500,
):
    """
    Run the fine-tuned model on the test set and compute metrics.
    max_samples: cap to avoid long runtime (set None for full eval)
    """
    print(f"\nLoading test set: {test_jsonl}")
    examples = []
    with open(test_jsonl) as f:
        for line in f:
            examples.append(json.loads(line))
    if max_samples:
        # Stratified sampling to maintain class ratio
        fraud  = [e for e in examples if e.get("metadata", {}).get("is_fraud")]
        legit  = [e for e in examples if not e.get("metadata", {}).get("is_fraud")]
        n_f    = min(max_samples // 2, len(fraud))
        n_l    = min(max_samples - n_f, len(legit))
        examples = fraud[:n_f] + legit[:n_l]
        import random; random.shuffle(examples)

    print(f"  Evaluating on {len(examples)} examples...")

    y_true, y_pred, y_scores, datasets, raw_responses = [], [], [], [], []

    for i, ex in enumerate(examples):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(examples)}")

        messages  = ex.get("messages", [])
        user_msg  = next((m["content"] for m in messages if m["role"] == "user"), "")
        true_label = int(ex.get("metadata", {}).get("is_fraud", 0))
        dataset   = ex.get("metadata", {}).get("dataset", "unknown")

        prompt   = build_inference_prompt(tokenizer, user_msg)
        response = generate_response(tokenizer, model, prompt)

        pred_label = parse_verdict(response)
        risk_score = parse_risk_score(response) or 2

        y_true.append(true_label)
        y_pred.append(pred_label if pred_label is not None else 0)
        # Use risk_score as proxy for fraud probability
        y_scores.append((risk_score - 1) / 4.0)   # normalize to [0, 1]
        datasets.append(dataset)
        raw_responses.append(response)

    return y_true, y_pred, y_scores, datasets, raw_responses


def compute_and_print_metrics(
    y_true, y_pred, y_scores, datasets, output_prefix: str = ""
):
    """Compute and display all evaluation metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Overall metrics
    f1        = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    mcc       = matthews_corrcoef(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = float("nan")

    print(f"\nOverall Metrics:")
    print(f"  F1 Score          : {f1:.4f}")
    print(f"  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall:.4f}")
    print(f"  MCC               : {mcc:.4f}")
    print(f"  AUROC             : {auc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Legitimate", "Fraud"],
        digits=4,
    ))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    # Per-dataset breakdown
    df_results = pd.DataFrame({
        "dataset": datasets,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    print("\nPer-Dataset Breakdown:")
    for ds, grp in df_results.groupby("dataset"):
        ds_f1  = f1_score(grp["y_true"], grp["y_pred"], zero_division=0)
        ds_rec = recall_score(grp["y_true"], grp["y_pred"], zero_division=0)
        n_f    = grp["y_true"].sum()
        print(f"  {ds:12s}: F1={ds_f1:.4f}  Recall={ds_rec:.4f}  "
              f"({n_f}/{len(grp)} fraud samples)")

    # Save metrics to JSON
    results = {
        "f1": f1, "precision": precision, "recall": recall,
        "mcc": mcc, "auroc": auc,
        "confusion_matrix": cm.tolist(),
    }
    out_path = os.path.join(RESULTS_DIR, f"{output_prefix}metrics.json")
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nMetrics saved to: {out_path}")

    # Plot and save confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Fraud"]).plot(ax=ax)
    ax.set_title(f"IBM Granite Fraud Detector\nF1={f1:.4f}  MCC={mcc:.4f}")
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, f"{output_prefix}confusion_matrix.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Confusion matrix plot saved to: {fig_path}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Interactive Single-Transaction Inference
# ──────────────────────────────────────────────────────────────────────────────

def analyze_transaction(
    tokenizer,
    model,
    transaction_dict: dict,
    dataset_context: str = "Financial transaction",
) -> dict:
    """
    Analyze a single transaction dictionary.

    Example:
        analyze_transaction(tokenizer, model, {
            "amount": 4521.00,
            "transaction_type": "TRANSFER",
            "time_step": 3,
            "balance_diff_orig": -4521.00,
        })
    """
    # Serialize transaction to natural language
    feature_lines = "\n".join(
        f"  - {k}: {v}" for k, v in transaction_dict.items()
    )
    user_message = (
        f"Analyze the following transaction for potential fraud or AML risk.\n\n"
        f"Context: {dataset_context}\n\n"
        f"Transaction Features:\n{feature_lines}\n\n"
        f"Provide a structured JSON risk assessment."
    )

    prompt   = build_inference_prompt(tokenizer, user_message)
    response = generate_response(tokenizer, model, prompt)

    # Parse response
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        result = {
            "raw_response": response,
            "verdict": "PARSE_ERROR",
            "note": "Model output was not valid JSON",
        }

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Granite fraud model")
    parser.add_argument("--merged", action="store_true",
                        help="Use merged model instead of adapter")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--demo", action="store_true",
                        help="Run a demo inference on a sample transaction")
    args = parser.parse_args()

    print("Loading model for evaluation...")
    tokenizer, model = load_model_for_inference(use_merged=args.merged)

    if args.demo:
        # Demo: analyze a suspicious transaction
        print("\n" + "=" * 60)
        print("DEMO: Analyzing a suspicious transaction")
        print("=" * 60)
        sample_txn = {
            "amount_scaled":             2.87,
            "type_TRANSFER":             1,
            "type_CASH_OUT":             0,
            "balance_diff_orig":        -2.87,
            "balance_diff_dest":         2.85,
            "amount_to_balance_ratio":   0.998,
            "step":                      183,
            "txn_count_24h":             12,
        }
        result = analyze_transaction(tokenizer, model, sample_txn,
                                     "Mobile payment system transaction")
        print(json.dumps(result, indent=2))
    else:
        # Full evaluation
        y_true, y_pred, y_scores, datasets, responses = evaluate_on_test_set(
            tokenizer, model,
            max_samples=args.max_samples,
        )
        compute_and_print_metrics(y_true, y_pred, y_scores, datasets)
