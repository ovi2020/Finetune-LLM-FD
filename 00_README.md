# IBM Granite Fine-Tuning for Fraud Detection & Financial Crime Intelligence

## Project Structure

```
granite_fraud_finetuning/
├── 00_README.md                  ← This file
├── 01_dataset_survey.py          ← Dataset overview and download helpers
├── 02_data_preprocessing.py      ← Full preprocessing pipeline for all datasets
├── 03_instruction_dataset.py     ← Convert tabular data → instruction-tuning format
├── 04_finetune_granite.py        ← QLoRA fine-tuning with IBM Granite
├── 05_evaluate_model.py          ← Evaluation and inference
└── requirements.txt              ← All dependencies
```

## Datasets Covered

| # | Dataset | Type | Size | Source |
|---|---------|------|------|--------|
| 1 | Credit Card Fraud (ULB) | Credit card transactions | 284K rows | Kaggle |
| 2 | IEEE-CIS Fraud Detection | E-commerce transactions | 590K rows | Kaggle |
| 3 | PaySim Synthetic | Mobile money | 6.3M rows | Kaggle |
| 4 | IBM AML Synthetic (HI-Small) | Anti-money laundering | ~5M edges | GitHub/IBM |
| 5 | Elliptic Bitcoin Dataset | Crypto / AML | 203K nodes | Kaggle |
| 6 | BankSim | Bank transactions | ~600K rows | Kaggle |

## Pipeline Overview

```
Raw Datasets
    ↓
02_data_preprocessing.py   (clean, normalize, handle imbalance)
    ↓
03_instruction_dataset.py  (convert rows → prompt/completion pairs)
    ↓
04_finetune_granite.py     (QLoRA fine-tune ibm-granite/granite-3.3-8b-instruct)
    ↓
05_evaluate_model.py       (F1, Precision, Recall, MCC on held-out set)
```

## GPU Requirements

- Minimum: 2× A100 40GB (or 4× V100 32GB) for 8B model with QLoRA
- For 70B: 4× A100 80GB with QLoRA + DeepSpeed ZeRO-3
- Cloud options: AWS p4d.24xlarge, GCP A100 × 8, RunPod, Lambda Labs

## Quick Start

```bash
pip install -r requirements.txt
python 01_dataset_survey.py        # Review & download datasets
python 02_data_preprocessing.py    # Preprocess all datasets
python 03_instruction_dataset.py   # Build instruction-tuning JSONL
python 04_finetune_granite.py      # Fine-tune (set MODEL_SIZE below)
python 05_evaluate_model.py        # Evaluate the fine-tuned model
```
