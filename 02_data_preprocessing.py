"""
02_data_preprocessing.py
========================
Full preprocessing pipeline for all fraud detection & AML datasets.
Handles:
  - Missing value imputation
  - Feature engineering (balance diffs, velocity features, time features)
  - Categorical encoding
  - Class imbalance (SMOTE, undersampling, class weights)
  - Train/Val/Test stratified split
  - Per-dataset StandardScaler normalization
  - Saving cleaned Parquet files for downstream instruction-tuning
"""

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

DATA_DIR     = "./data"
OUTPUT_DIR   = "./data/processed"
RANDOM_SEED  = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_splits(alias: str, X_train, X_val, X_test, y_train, y_val, y_test):
    """Concatenate features + labels and save as Parquet."""
    for split, X, y in [("train", X_train, y_train),
                        ("val",   X_val,   y_val),
                        ("test",  X_test,  y_test)]:
        df = X.copy()
        df["label"] = y.values if hasattr(y, "values") else y
        path = os.path.join(OUTPUT_DIR, f"{alias}_{split}.parquet")
        df.to_parquet(path, index=False)
        n_fraud = int(df["label"].sum())
        print(f"  Saved {split:5s}: {len(df):>8,} rows  ({n_fraud:,} fraud) → {path}")


def report(alias: str, df: pd.DataFrame, label_col: str):
    n_fraud = df[label_col].sum()
    ratio   = n_fraud / len(df) * 100
    print(f"\n[{alias.upper()}] {len(df):,} rows | {n_fraud:,} fraud ({ratio:.3f}%)")


def compute_weights(y_train) -> dict:
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return dict(zip(classes, weights))


def stratified_split(df: pd.DataFrame, label_col: str):
    """80/10/10 stratified train/val/test split."""
    y   = df[label_col]
    X   = df.drop(columns=[label_col])
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.1111, stratify=y_tv, random_state=RANDOM_SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote_undersample(X_train, y_train,
                             sampling_strategy_over=0.1,
                             sampling_strategy_under=0.5):
    """
    Two-step resampling for extreme class imbalance:
      Step 1 – SMOTE: oversample minority to `sampling_strategy_over` ratio
               (minority / majority). Skipped if minority is already at or
               above the target ratio — avoids the imblearn ValueError.
      Step 2 – RandomUnderSampler: undersample majority so minority reaches
               `sampling_strategy_under` ratio. Skipped if already there.
    Returns resampled X, y as DataFrames/Series.
    """
    cols = X_train.columns.tolist()
    counts = y_train.value_counts()
    n_majority = counts[0]
    n_minority = counts[1]
    current_ratio = n_minority / n_majority  # minority / majority

    steps = []

    # Only SMOTE if minority is genuinely below the target ratio
    if current_ratio < sampling_strategy_over:
        steps.append(("over", SMOTE(sampling_strategy=sampling_strategy_over,
                                    random_state=RANDOM_SEED, k_neighbors=5)))
    else:
        print(f"  [SMOTE skipped] minority ratio {current_ratio:.3f} already "
              f">= target {sampling_strategy_over}")

    # Only undersample if majority still needs trimming
    # After possible SMOTE, recompute expected ratio to validate under step
    expected_minority = max(n_minority, int(n_majority * sampling_strategy_over))
    expected_ratio_after_over = expected_minority / n_majority
    if expected_ratio_after_over < sampling_strategy_under:
        steps.append(("under", RandomUnderSampler(
            sampling_strategy=sampling_strategy_under,
            random_state=RANDOM_SEED)))
    else:
        print(f"  [UnderSampler skipped] ratio {expected_ratio_after_over:.3f} "
              f"already >= target {sampling_strategy_under}")

    if not steps:
        print("  [Resampling skipped] data already meets both ratio targets.")
        return X_train.copy(), y_train.copy()

    pipe = ImbPipeline(steps)
    X_res, y_res = pipe.fit_resample(X_train, y_train)
    return pd.DataFrame(X_res, columns=cols), pd.Series(y_res, name=y_train.name)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Credit Card Fraud (ULB)
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_ccf():
    path = os.path.join(DATA_DIR, "creditcard.csv")
    if not os.path.exists(path):
        print(f"[CCF] Skipped – file not found: {path}")
        return

    df = pd.read_csv(path)
    report("ccf", df, "Class")

    # Normalize Time (seconds → hour of day proxy) and Amount
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])
    df.drop(columns=["Amount", "Time"], inplace=True)

    # Rename label
    df.rename(columns={"Class": "label"}, inplace=True)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df, "label")

    # Resample training set (SMOTE + undersampling)
    print("  Applying SMOTE + undersampling...")
    X_train, y_train = apply_smote_undersample(X_train, y_train)

    print(f"  Class weights: {compute_weights(y_train)}")
    save_splits("ccf", X_train, X_val, X_test, y_train, y_val, y_test)


# ──────────────────────────────────────────────────────────────────────────────
# 2. IEEE-CIS Fraud Detection
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_ieee_cis():
    txn_path = os.path.join(DATA_DIR, "train_transaction.csv")
    id_path  = os.path.join(DATA_DIR, "train_identity.csv")

    if not os.path.exists(txn_path):
        print(f"[IEEE-CIS] Skipped – file not found: {txn_path}")
        return

    print("\n[IEEE-CIS] Loading and merging tables...")
    txn = pd.read_csv(txn_path)
    if os.path.exists(id_path):
        identity = pd.read_csv(id_path)
        df = txn.merge(identity, on="TransactionID", how="left")
    else:
        df = txn

    report("ieee_cis", df, "isFraud")

    # Drop high-null columns (>50% missing)
    null_pct    = df.isnull().mean()
    drop_cols   = null_pct[null_pct > 0.50].index.tolist()
    df.drop(columns=drop_cols, inplace=True)
    print(f"  Dropped {len(drop_cols)} high-null columns")

    # Drop identifiers
    df.drop(columns=["TransactionID"], errors="ignore", inplace=True)

    # Impute numerics with median, categoricals with mode
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(["isFraud"])
    cat_cols = df.select_dtypes(include=["object"]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Label-encode all categoricals
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Scale TransactionAmt
    scaler = StandardScaler()
    df["TransactionAmt"] = scaler.fit_transform(df[["TransactionAmt"]])

    df.rename(columns={"isFraud": "label"}, inplace=True)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df, "label")
    print("  Applying SMOTE + undersampling...")
    X_train, y_train = apply_smote_undersample(X_train, y_train)

    print(f"  Class weights: {compute_weights(y_train)}")
    save_splits("ieee_cis", X_train, X_val, X_test, y_train, y_val, y_test)


# ──────────────────────────────────────────────────────────────────────────────
# 3. PaySim Synthetic
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_paysim():
    path = os.path.join(DATA_DIR, "PS_20174392719_1491204439457_log.csv")
    if not os.path.exists(path):
        print(f"[PaySim] Skipped – file not found: {path}")
        return

    df = pd.read_csv(path)
    report("paysim", df, "isFraud")

    # Feature engineering
    df["balance_diff_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1e-9)

    # One-hot encode transaction type
    df = pd.get_dummies(df, columns=["type"], prefix="type", drop_first=False)

    # Drop potentially leaky and high-cardinality columns
    drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud",
                 "oldbalanceOrg", "newbalanceOrig",
                 "oldbalanceDest", "newbalanceDest"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Scale amount
    scaler = StandardScaler()
    df["amount"] = scaler.fit_transform(df[["amount"]])

    df.rename(columns={"isFraud": "label"}, inplace=True)

    # Sample 10% for manageable training (still 600K+ rows)
    df = df.sample(frac=0.10, random_state=RANDOM_SEED)
    print(f"  Sampled to {len(df):,} rows (10%) for manageability")

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df, "label")
    print("  Applying SMOTE + undersampling...")
    X_train, y_train = apply_smote_undersample(X_train, y_train)

    save_splits("paysim", X_train, X_val, X_test, y_train, y_val, y_test)


# ──────────────────────────────────────────────────────────────────────────────
# 4. IBM AML Synthetic (HI-Small)
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_ibm_aml():
    path = os.path.join(DATA_DIR, "HI-Small_Trans.csv")
    if not os.path.exists(path):
        print(f"[IBM-AML] Skipped – file not found: {path}")
        return

    df = pd.read_csv(path)
    report("ibm_aml", df, "Is Laundering")

    # Rename columns to snake_case
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour_of_day"]  = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day

    # Velocity features per sending account
    df.sort_values(["account", "timestamp"], inplace=True)
    df["txn_count_24h"] = (
        df.groupby("account")["timestamp"]
          .transform(lambda x: x.expanding().count())
    )

    # Encode categoricals
    for col in ["payment_currency", "receiving_currency",
                "payment_format", "account", "account.1"]:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Drop raw timestamp
    df.drop(columns=["timestamp"], errors="ignore", inplace=True)

    # Scale amount
    scaler = StandardScaler()
    df["amount_paid"] = scaler.fit_transform(df[["amount_paid"]])

    # Preserve laundering_type as additional context before renaming
    if "laundering_type" in df.columns:
        df["laundering_type_label"] = df["laundering_type"]

    df.rename(columns={"is_laundering": "label"}, inplace=True)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df, "label")
    print("  Applying SMOTE + undersampling...")
    X_train, y_train = apply_smote_undersample(X_train, y_train)

    save_splits("ibm_aml", X_train, X_val, X_test, y_train, y_val, y_test)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Elliptic Bitcoin Dataset
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_elliptic():
    features_path = os.path.join(DATA_DIR, "elliptic_txs_features.csv")
    classes_path  = os.path.join(DATA_DIR, "elliptic_txs_classes.csv")

    if not os.path.exists(features_path):
        print(f"[Elliptic] Skipped – file not found: {features_path}")
        return

    print("\n[Elliptic] Loading transaction features and class labels...")
    feat_cols = ["txId", "time_step"] + [f"f{i}" for i in range(1, 166)]
    features  = pd.read_csv(features_path, header=None, names=feat_cols)
    classes   = pd.read_csv(classes_path)

    df = features.merge(classes, on="txId", how="inner")
    df = df[df["class"] != "unknown"]  # drop unlabelled nodes

    # Remap: 1 (illicit) → 1 (fraud), 2 (licit) → 0 (legit)
    df["label"] = df["class"].map({"1": 1, "2": 0, 1: 1, 2: 0}).astype(int)
    df.drop(columns=["txId", "class"], inplace=True)

    report("elliptic", df, "label")

    # Normalize all features
    feature_cols = [c for c in df.columns if c not in ["label", "time_step"]]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df, "label")
    print("  Applying SMOTE + undersampling...")
    X_train, y_train = apply_smote_undersample(X_train, y_train)

    save_splits("elliptic", X_train, X_val, X_test, y_train, y_val, y_test)


# ──────────────────────────────────────────────────────────────────────────────
# 6. BankSim
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_banksim():
    path = os.path.join(DATA_DIR, "bs140513_032310.csv")
    if not os.path.exists(path):
        print(f"[BankSim] Skipped – file not found: {path}")
        return

    df = pd.read_csv(path)
    report("banksim", df, "fraud")

    # Drop quotes from string fields
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip("'\"")

    # Age bucket → numeric
    age_map = {"U": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "O": 7}
    df["age"] = df["age"].map(age_map).fillna(0).astype(int)

    # Encode gender
    df["gender"] = (df["gender"] == "M").astype(int)

    # Label-encode merchant and category
    le = LabelEncoder()
    df["merchant"]  = le.fit_transform(df["merchant"].astype(str))
    df["category"]  = le.fit_transform(df["category"].astype(str))
    df["customer"]  = le.fit_transform(df["customer"].astype(str))

    # Scale amount
    scaler = StandardScaler()
    df["amount"] = scaler.fit_transform(df[["amount"]])

    df.rename(columns={"fraud": "label"}, inplace=True)
    df["label"] = df["label"].astype(int)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df, "label")
    print("  Applying SMOTE + undersampling...")
    X_train, y_train = apply_smote_undersample(X_train, y_train)

    save_splits("banksim", X_train, X_val, X_test, y_train, y_val, y_test)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING PREPROCESSING PIPELINE FOR ALL DATASETS")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    preprocess_ccf()
    preprocess_ieee_cis()
    preprocess_paysim()
    preprocess_ibm_aml()
    preprocess_elliptic()
    preprocess_banksim()

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print(f"All splits saved to: {OUTPUT_DIR}")
    print("Next step: python 03_instruction_dataset.py")
    print("=" * 70)
