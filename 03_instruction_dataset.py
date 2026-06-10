"""
03_instruction_dataset.py
=========================
Converts preprocessed tabular fraud data into instruction-tuning format
suitable for fine-tuning IBM Granite (and other causal LLMs).

Each tabular row becomes a structured prompt-completion pair:
  - System prompt: domain expert framing
  - User prompt:   serialized transaction features as natural language
  - Assistant:     structured JSON response with verdict + reasoning

Output: JSONL files (train.jsonl, val.jsonl, test.jsonl) in ./data/sft/
Compatible with: HuggingFace TRL SFTTrainer, LLaMA-Factory, Axolotl
"""

import os
import json
import random
import textwrap
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path

PROCESSED_DIR = "./data/processed"
SFT_DIR       = "./data/sft"
RANDOM_SEED   = 42
MAX_ROWS_PER_DATASET = 15_000   # cap per dataset to keep training set manageable

os.makedirs(SFT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are a financial crime intelligence expert with deep expertise in:
- Transaction fraud detection (credit card, mobile payments, e-commerce)
- Anti-money laundering (AML) and financial crime typology analysis
- Suspicious Activity Report (SAR) generation
- Regulatory compliance (FATF, BSA, AML/CFT frameworks)

When presented with a financial transaction or account activity profile,
you analyze the features and provide a structured risk assessment.
Always respond in valid JSON format as specified.
""").strip()


def make_user_prompt(dataset_alias: str, row: pd.Series) -> str:
    """Serialize a transaction row into a human-readable prompt."""
    feature_lines = []
    for col, val in row.items():
        if col == "label":
            continue
        # Format numerics nicely
        if isinstance(val, float):
            feature_lines.append(f"  - {col}: {val:.4f}")
        else:
            feature_lines.append(f"  - {col}: {val}")

    features_str = "\n".join(feature_lines)

    dataset_context = {
        "ccf":      "European cardholder credit card transaction (Sept 2013)",
        "ieee_cis": "E-commerce transaction with identity signals (Vesta Corp dataset)",
        "paysim":   "Mobile money transfer transaction (West African mobile service)",
        "ibm_aml":  "Inter-bank wire transfer (IBM AML synthetic financial network)",
        "elliptic": "Bitcoin blockchain transaction (Elliptic financial forensics dataset)",
        "banksim":  "Retail bank transaction (Spanish bank customer activity)",
    }

    context = dataset_context.get(dataset_alias, "Financial transaction")

    return textwrap.dedent(f"""
Analyze the following financial transaction for potential fraud or money laundering activity.

Dataset Context: {context}

Transaction Features:
{features_str}

Provide a comprehensive financial crime risk assessment in the JSON format specified.
""").strip()


def make_assistant_response(label: int, dataset_alias: str,
                             row: pd.Series) -> str:
    """
    Generate a structured JSON response for the given label.
    For fine-tuning: positive examples get detailed fraud reasoning,
    negative examples get a clean-bill-of-health with normal behavior notes.
    """
    is_fraud = bool(label == 1)

    # Risk score: fraud → 4-5, legit → 1-2 (with some variation)
    if is_fraud:
        risk_score = random.choice([4, 5])
    else:
        risk_score = random.choice([1, 2])

    # Dataset-specific fraud type context
    fraud_types = {
        "ccf":      ["card-not-present fraud", "account takeover", "stolen card"],
        "ieee_cis": ["identity theft", "device spoofing", "card testing",
                     "friendly fraud", "synthetic identity"],
        "paysim":   ["cash-out fraud", "transfer layering", "mule account activity",
                     "rapid fund movement"],
        "ibm_aml":  ["layering", "fan-out structuring", "scatter-gather pattern",
                     "round-trip transactions", "shell account activity"],
        "elliptic": ["ransomware payment", "darknet marketplace", "scam",
                     "Ponzi scheme", "terrorist financing", "malware"],
        "banksim":  ["merchant fraud", "account takeover", "unusual purchase pattern"],
    }

    aml_typologies = {
        "ibm_aml":  ["fan-in", "fan-out", "bipartite", "cycle", "scatter-gather",
                     "gather-scatter", "random", "stack"],
    }

    indicators = []
    typology   = None

    if is_fraud:
        fraud_type = random.choice(fraud_types.get(dataset_alias, ["unspecified fraud"]))
        if dataset_alias == "ibm_aml":
            typology = random.choice(aml_typologies["ibm_aml"])

        # Generate plausible indicators based on feature values
        amount_features = [col for col in row.index if "amount" in col.lower()]
        if amount_features:
            val = row[amount_features[0]]
            if abs(val) > 1.5:   # scaled value
                indicators.append("Unusually high transaction amount compared to account profile")
            elif abs(val) < -1.0:
                indicators.append("Suspiciously low transaction amount (possible structuring below threshold)")

        balance_features = [col for col in row.index if "balance_diff" in col.lower()]
        if balance_features:
            val = row[balance_features[0]]
            if abs(val) > 1.5:
                indicators.append("Significant balance change inconsistent with normal activity")

        indicators += random.sample([
            "Transaction pattern deviates from established customer baseline",
            "Velocity anomaly: high frequency of transactions in short window",
            "Geographic/IP mismatch with cardholder profile",
            "Transaction amount near reporting threshold (structuring indicator)",
            "High-risk merchant category code",
            "New account with immediate high-value activity",
            "Destination account has no prior transaction history",
            "Funds rapidly moved onward within minutes of receipt",
        ], min(3, 8 - len(indicators)))
    else:
        fraud_type = None
        indicators = random.sample([
            "Transaction within normal amount range for this customer segment",
            "Consistent with established spending patterns",
            "Merchant category aligns with customer profile",
            "No velocity anomalies detected",
            "Device and location consistent with prior activity",
        ], 2)

    response_obj = {
        "verdict":       "FRAUD" if is_fraud else "LEGITIMATE",
        "risk_score":    risk_score,
        "risk_level":    "HIGH" if risk_score >= 4 else ("MEDIUM" if risk_score == 3 else "LOW"),
        "fraud_type":    fraud_type,
        "aml_typology":  typology,
        "risk_indicators": indicators,
        "recommended_action": (
            "BLOCK and escalate to fraud analyst. File SAR if laundering confirmed."
            if risk_score == 5 else
            "HOLD for manual review. Request additional verification."
            if risk_score == 4 else
            "MONITOR. Flag for enhanced due diligence on next transaction."
            if risk_score == 3 else
            "APPROVE. No suspicious indicators detected."
        ),
        "confidence":    "HIGH" if random.random() > 0.3 else "MEDIUM",
    }

    return json.dumps(response_obj, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Instruction Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_instruction_examples(
    alias: str,
    split: str,
    max_rows: Optional[int] = MAX_ROWS_PER_DATASET,
) -> list:
    """
    Load a preprocessed parquet split, convert rows to instruction examples.
    Returns a list of dicts: {system, user, assistant}
    """
    path = os.path.join(PROCESSED_DIR, f"{alias}_{split}.parquet")
    if not os.path.exists(path):
        print(f"  [SKIP] {alias}/{split} – {path} not found")
        return []

    df = pd.read_parquet(path)

    # Stratified sampling to honour the class ratio but cap total rows
    if max_rows and len(df) > max_rows:
        fraud_df   = df[df["label"] == 1]
        legit_df   = df[df["label"] == 0]
        fraud_ratio = len(fraud_df) / len(df)
        n_fraud     = int(max_rows * fraud_ratio)
        n_legit     = max_rows - n_fraud
        df = pd.concat([
            fraud_df.sample(min(n_fraud, len(fraud_df)), random_state=RANDOM_SEED),
            legit_df.sample(min(n_legit, len(legit_df)), random_state=RANDOM_SEED),
        ]).sample(frac=1, random_state=RANDOM_SEED)   # shuffle

    examples = []
    for _, row in df.iterrows():
        label = int(row["label"])
        example = {
            "system":    SYSTEM_PROMPT,
            "user":      make_user_prompt(alias, row),
            "assistant": make_assistant_response(label, alias, row),
            "metadata": {
                "dataset":   alias,
                "split":     split,
                "label":     label,
                "is_fraud":  label == 1,
            },
        }
        examples.append(example)

    n_fraud = sum(1 for e in examples if e["metadata"]["is_fraud"])
    print(f"  {alias}/{split}: {len(examples):,} examples ({n_fraud:,} fraud)")
    return examples


# ──────────────────────────────────────────────────────────────────────────────
# Granite Chat Format Converter
# ──────────────────────────────────────────────────────────────────────────────

def to_granite_chat_format(example: dict) -> dict:
    """
    Convert a {system, user, assistant} dict to Granite 3.x chat format.
    Granite uses the standard OpenAI messages format:
      [{"role": "system", "content": ...},
       {"role": "user",   "content": ...},
       {"role": "assistant", "content": ...}]
    TRL SFTTrainer with apply_chat_template handles the rest.
    """
    return {
        "messages": [
            {"role": "system",    "content": example["system"]},
            {"role": "user",      "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
        ],
        "metadata": example.get("metadata", {}),
    }


def to_alpaca_format(example: dict) -> dict:
    """
    Alternative: Alpaca-style format (instruction / output).
    Some fine-tuning frameworks prefer this.
    """
    return {
        "instruction": example["system"] + "\n\n" + example["user"],
        "output":      example["assistant"],
        "metadata":    example.get("metadata", {}),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Build & Save Combined JSONL
# ──────────────────────────────────────────────────────────────────────────────

DATASET_ALIASES = ["ccf", "ieee_cis", "paysim", "ibm_aml", "elliptic", "banksim"]


def build_and_save(output_format: str = "granite_chat"):
    """
    Build train/val/test JSONL files combining all datasets.
    output_format: 'granite_chat' | 'alpaca'
    """
    assert output_format in ("granite_chat", "alpaca"), \
        "output_format must be 'granite_chat' or 'alpaca'"

    converter = to_granite_chat_format if output_format == "granite_chat" \
                else to_alpaca_format

    for split in ["train", "val", "test"]:
        print(f"\n─── Building {split.upper()} split ───")
        combined = []

        for alias in DATASET_ALIASES:
            examples = build_instruction_examples(alias, split)
            combined.extend(examples)

        # Shuffle combined set
        random.shuffle(combined)

        # Convert to target format
        formatted = [converter(e) for e in combined]

        # Save as JSONL
        out_path = os.path.join(SFT_DIR, f"{split}.jsonl")
        with open(out_path, "w") as f:
            for item in formatted:
                f.write(json.dumps(item) + "\n")

        n_fraud = sum(1 for e in combined if e["metadata"]["is_fraud"])
        print(f"  ✓ {split}: {len(formatted):,} total examples "
              f"({n_fraud:,} fraud, {len(formatted)-n_fraud:,} legit) → {out_path}")

    print(f"\nAll JSONL files saved to: {SFT_DIR}")


def preview_examples(n: int = 2):
    """Print a few formatted examples for inspection."""
    train_path = os.path.join(SFT_DIR, "train.jsonl")
    if not os.path.exists(train_path):
        print("No training JSONL found – run build_and_save() first.")
        return

    examples = []
    with open(train_path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            examples.append(json.loads(line))

    for i, ex in enumerate(examples, 1):
        print(f"\n{'='*70}")
        print(f"EXAMPLE {i}")
        print(f"{'='*70}")
        if "messages" in ex:
            for msg in ex["messages"]:
                print(f"[{msg['role'].upper()}]")
                print(msg["content"][:500] + ("..." if len(msg["content"]) > 500 else ""))
                print()
        elif "instruction" in ex:
            print("[INSTRUCTION]")
            print(ex["instruction"][:400])
            print("\n[OUTPUT]")
            print(ex["output"][:400])


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("BUILDING INSTRUCTION-TUNING DATASET FOR IBM GRANITE")
    print(f"Output directory: {SFT_DIR}")
    print(f"Output format: granite_chat (messages list)")
    print("=" * 70)

    build_and_save(output_format="granite_chat")
    preview_examples(n=2)

    print("\n" + "=" * 70)
    print("INSTRUCTION DATASET READY")
    print("Next step: python 04_finetune_granite.py")
    print("=" * 70)
