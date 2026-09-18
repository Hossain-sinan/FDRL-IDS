"""
Day 21 — Evaluate XGBoost Baseline

Purpose:
Evaluate the saved Day 20 centralized XGBoost model using
the same CICIDS2017 preprocessing and stratified test split.

Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- False Positive Rate (FPR)

FPR is calculated:
1. Per class using one-vs-rest
2. Macro FPR across all 8 classes
3. Binary attack-vs-benign FPR for IDS interpretation
"""

import os
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

VALIDATION_SIZE = 0.10
TEST_SIZE = 0.10

NUM_CLASSES = 8

CLASS_NAMES = [
    "BENIGN",
    "DoS",
    "DDoS",
    "Brute Force",
    "Web Attack",
    "Infiltration",
    "Botnet",
    "Port Scan"
]


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FEATURE_FILE = os.path.join(
    PROJECT_ROOT,
    "docs",
    "day16_features_20.csv"
)

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "models",
    "xgboost_baseline.json"
)

DOCS_DIR = os.path.join(
    PROJECT_ROOT,
    "docs"
)

RESULT_FILE = os.path.join(
    DOCS_DIR,
    "day21_baseline_results.csv"
)

PER_CLASS_FILE = os.path.join(
    DOCS_DIR,
    "day21_per_class_metrics.csv"
)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

sys.path.insert(
    0,
    os.path.join(PROJECT_ROOT, "src")
)

from data_loader import load_cicids2017
from preprocessing import clean_data
from label_processing import process_labels


# ============================================================
# FPR FUNCTION
# ============================================================

def calculate_per_class_fpr(y_true, y_pred):
    """
    Calculate one-vs-rest FPR for each class.

    FPR = FP / (FP + TN)
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    total = cm.sum()

    results = []

    for class_id, class_name in enumerate(CLASS_NAMES):

        tp = cm[class_id, class_id]

        fp = cm[:, class_id].sum() - tp

        fn = cm[class_id, :].sum() - tp

        tn = total - tp - fp - fn

        denominator = fp + tn

        if denominator == 0:
            fpr = 0.0
        else:
            fpr = fp / denominator

        results.append({
            "label_id": class_id,
            "class": class_name,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "FPR": fpr
        })

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DAY 21 — EVALUATE XGBOOST BASELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1 — CHECK MODEL
    # --------------------------------------------------------

    print("\n[STEP 1] Checking saved Day 20 model")
    print("-" * 70)

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"XGBoost model not found:\n{MODEL_FILE}"
        )

    print(f"Model found:")
    print(MODEL_FILE)

    # --------------------------------------------------------
    # STEP 2 — LOAD DATA
    # --------------------------------------------------------

    print("\n[STEP 2] Loading CICIDS2017 dataset")
    print("-" * 70)

    df = load_cicids2017()

    print(f"Raw dataset shape: {df.shape}")

    # --------------------------------------------------------
    # STEP 3 — CLEAN DATA
    # --------------------------------------------------------

    print("\n[STEP 3] Cleaning data")
    print("-" * 70)

    df = clean_data(df)

    print(f"Cleaned dataset shape: {df.shape}")

    # --------------------------------------------------------
    # STEP 4 — PROCESS LABELS
    # --------------------------------------------------------

    print("\n[STEP 4] Processing labels")
    print("-" * 70)

    df = process_labels(df)

    print(f"Dataset after label processing: {df.shape}")

    # --------------------------------------------------------
    # STEP 5 — LOAD SELECTED FEATURES
    # --------------------------------------------------------

    print("\n[STEP 5] Loading 20 selected features")
    print("-" * 70)

    feature_df = pd.read_csv(FEATURE_FILE)

    selected_features = feature_df["feature"].tolist()

    print(f"Selected feature count: {len(selected_features)}")

    # --------------------------------------------------------
    # STEP 6 — CREATE X AND y
    # --------------------------------------------------------

    print("\n[STEP 6] Creating X and y")
    print("-" * 70)

    X = df[selected_features].copy()
    y = df["label_id"].copy()

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # --------------------------------------------------------
    # STEP 7 — RECREATE DAY 20 SPLIT
    # --------------------------------------------------------

    print("\n[STEP 7] Recreating Day 20 stratified split")
    print("-" * 70)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(VALIDATION_SIZE + TEST_SIZE),
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    print(f"Training samples:   {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples:       {len(X_test):,}")

    # --------------------------------------------------------
    # STEP 8 — LOAD SAVED XGBOOST MODEL
    # --------------------------------------------------------

    print("\n[STEP 8] Loading Day 20 XGBoost model")
    print("-" * 70)

    model = XGBClassifier()

    model.load_model(MODEL_FILE)

    print("Saved XGBoost model loaded successfully.")

    # --------------------------------------------------------
    # STEP 9 — PREDICT TEST DATA
    # --------------------------------------------------------

    print("\n[STEP 9] Generating test predictions")
    print("-" * 70)

    y_pred = model.predict(X_test)

    print("Prediction completed.")

    # --------------------------------------------------------
    # STEP 10 — OVERALL METRICS
    # --------------------------------------------------------

    print("\n[STEP 10] Calculating overall metrics")
    print("-" * 70)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    weighted_recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy:           {accuracy:.6f}")
    print(f"Macro Precision:    {macro_precision:.6f}")
    print(f"Macro Recall:       {macro_recall:.6f}")
    print(f"Macro F1:           {macro_f1:.6f}")
    print(f"Weighted Precision: {weighted_precision:.6f}")
    print(f"Weighted Recall:    {weighted_recall:.6f}")
    print(f"Weighted F1:        {weighted_f1:.6f}")

    # --------------------------------------------------------
    # STEP 11 — PER-CLASS FPR
    # --------------------------------------------------------

    print("\n[STEP 11] Calculating per-class FPR")
    print("-" * 70)

    per_class_fpr = calculate_per_class_fpr(
        y_test,
        y_pred
    )

    print(
        per_class_fpr[
            ["label_id", "class", "TP", "FP", "TN", "FN", "FPR"]
        ].to_string(index=False)
    )

    macro_fpr = per_class_fpr["FPR"].mean()

    print(f"\nMacro FPR: {macro_fpr:.6f}")

    # --------------------------------------------------------
    # STEP 12 — BINARY ATTACK VS BENIGN FPR
    # --------------------------------------------------------

    print("\n[STEP 12] Calculating binary attack-vs-benign FPR")
    print("-" * 70)

    # BENIGN = 0
    #
    # Binary interpretation:
    # 0 = BENIGN
    # 1 = ATTACK
    #
    # Actual benign traffic incorrectly predicted
    # as any attack class = false positive.

    actual_benign = (y_test == 0)

    predicted_attack = (y_pred != 0)

    false_positives = (
        actual_benign & predicted_attack
    ).sum()

    true_negatives = (
        actual_benign & ~predicted_attack
    ).sum()

    binary_fpr = false_positives / (
        false_positives + true_negatives
    )

    print(f"False positives: {false_positives:,}")
    print(f"True negatives:  {true_negatives:,}")
    print(f"Binary Attack FPR: {binary_fpr:.6f}")

    # --------------------------------------------------------
    # STEP 13 — SAVE PER-CLASS METRICS
    # --------------------------------------------------------

    print("\n[STEP 13] Saving per-class metrics")
    print("-" * 70)

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    rows = []

    for class_id, class_name in enumerate(CLASS_NAMES):

        row = report_dict[class_name].copy()

        row["label_id"] = class_id
        row["class"] = class_name
        row["FPR"] = per_class_fpr.loc[
            per_class_fpr["label_id"] == class_id,
            "FPR"
        ].iloc[0]

        rows.append(row)

    per_class_metrics = pd.DataFrame(rows)

    per_class_metrics = per_class_metrics[
        [
            "label_id",
            "class",
            "precision",
            "recall",
            "f1-score",
            "support",
            "FPR"
        ]
    ]

    per_class_metrics.to_csv(
        PER_CLASS_FILE,
        index=False
    )

    print(f"Saved:")
    print(PER_CLASS_FILE)

    # --------------------------------------------------------
    # STEP 14 — SAVE BASELINE RESULTS
    # --------------------------------------------------------

    print("\n[STEP 14] Saving Day 21 baseline results")
    print("-" * 70)

    results = pd.DataFrame({
        "metric": [
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
            "macro_fpr",
            "binary_attack_fpr"
        ],
        "value": [
            accuracy,
            macro_precision,
            macro_recall,
            macro_f1,
            weighted_precision,
            weighted_recall,
            weighted_f1,
            macro_fpr,
            binary_fpr
        ]
    })

    results.to_csv(
        RESULT_FILE,
        index=False
    )

    print(f"Saved:")
    print(RESULT_FILE)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DAY 21 XGBOOST EVALUATION COMPLETED")
    print("=" * 70)

    print(f"\nAccuracy:           {accuracy:.4f}")
    print(f"Macro Precision:    {macro_precision:.4f}")
    print(f"Macro Recall:       {macro_recall:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")
    print(f"Weighted F1:        {weighted_f1:.4f}")
    print(f"Macro FPR:          {macro_fpr:.4f}")
    print(f"Binary Attack FPR:  {binary_fpr:.4f}")

    print("\nGenerated files:")
    print(f"1. {RESULT_FILE}")
    print(f"2. {PER_CLASS_FILE}")


if __name__ == "__main__":
    main()