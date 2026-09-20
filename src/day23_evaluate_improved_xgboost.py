"""
Day 23 — Detailed Evaluation of Improved XGBoost

Purpose:
Evaluate the saved Config_B XGBoost model from Day 23.

Important:
- The model is NOT retrained.
- The same preprocessing pipeline is used.
- The same 20 selected features are used.
- The same train/validation/test split is recreated.
- Evaluation focuses on class-level performance and IDS-specific metrics.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
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

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FEATURE_FILE = os.path.join(
    PROJECT_ROOT,
    "docs",
    "day16_features_20.csv"
)

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "models",
    "xgboost_improved_baseline.json"
)

DOCS_DIR = os.path.join(
    PROJECT_ROOT,
    "docs"
)

FIGURE_DIR = os.path.join(
    DOCS_DIR,
    "figures"
)

CLASSIFICATION_FILE = os.path.join(
    DOCS_DIR,
    "day23_improved_classification_report.csv"
)

PER_CLASS_FILE = os.path.join(
    DOCS_DIR,
    "day23_improved_per_class_metrics.csv"
)

CONFUSION_FILE = os.path.join(
    DOCS_DIR,
    "day23_improved_confusion_matrix.csv"
)

FIGURE_FILE = os.path.join(
    FIGURE_DIR,
    "day23_improved_confusion_matrix.png"
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
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("DAY 23 — DETAILED EVALUATION OF IMPROVED XGBOOST")
    print("=" * 75)

    # --------------------------------------------------------
    # STEP 1 — CHECK MODEL
    # --------------------------------------------------------

    print("\n[STEP 1] Checking saved improved model")
    print("-" * 75)

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Improved model not found:\n{MODEL_FILE}"
        )

    print(f"Model found:")
    print(MODEL_FILE)

    # --------------------------------------------------------
    # STEP 2 — LOAD DATA
    # --------------------------------------------------------

    print("\n[STEP 2] Loading CICIDS2017 dataset")
    print("-" * 75)

    df = load_cicids2017()

    print(f"Raw dataset shape: {df.shape}")

    # --------------------------------------------------------
    # STEP 3 — CLEAN DATA
    # --------------------------------------------------------

    print("\n[STEP 3] Cleaning data")
    print("-" * 75)

    df = clean_data(df)

    print(f"Cleaned dataset shape: {df.shape}")

    # --------------------------------------------------------
    # STEP 4 — PROCESS LABELS
    # --------------------------------------------------------

    print("\n[STEP 4] Processing labels")
    print("-" * 75)

    df = process_labels(df)

    print(f"Dataset after label processing: {df.shape}")

    # --------------------------------------------------------
    # STEP 5 — LOAD FEATURES
    # --------------------------------------------------------

    print("\n[STEP 5] Loading selected 20 features")
    print("-" * 75)

    feature_df = pd.read_csv(FEATURE_FILE)

    selected_features = feature_df["feature"].tolist()

    print(f"Selected feature count: {len(selected_features)}")

    # --------------------------------------------------------
    # STEP 6 — CREATE X AND y
    # --------------------------------------------------------

    print("\n[STEP 6] Creating X and y")
    print("-" * 75)

    X = df[selected_features].copy()
    y = df["label_id"].copy()

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # --------------------------------------------------------
    # STEP 7 — RECREATE SAME SPLIT
    # --------------------------------------------------------

    print("\n[STEP 7] Recreating same stratified split")
    print("-" * 75)

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
    # STEP 8 — LOAD SAVED MODEL
    # --------------------------------------------------------

    print("\n[STEP 8] Loading saved improved XGBoost model")
    print("-" * 75)

    model = XGBClassifier()

    model.load_model(MODEL_FILE)

    print("Model loaded successfully.")
    print("No retraining performed.")

    # --------------------------------------------------------
    # STEP 9 — PREDICTION
    # --------------------------------------------------------

    print("\n[STEP 9] Generating test predictions")
    print("-" * 75)

    y_pred = model.predict(X_test)

    print("Prediction completed.")

    # --------------------------------------------------------
    # STEP 10 — OVERALL METRICS
    # --------------------------------------------------------

    print("\n[STEP 10] Overall evaluation metrics")
    print("-" * 75)

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
    # STEP 11 — CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\n[STEP 11] Classification report")
    print("-" * 75)

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_text = classification_report(
        y_test,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print(report_text)

    report_df = pd.DataFrame(report_dict).transpose()

    report_df.to_csv(
        CLASSIFICATION_FILE
    )

    # --------------------------------------------------------
    # STEP 12 — CONFUSION MATRIX
    # --------------------------------------------------------

    print("\n[STEP 12] Creating confusion matrix")
    print("-" * 75)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    cm_df.to_csv(
        CONFUSION_FILE
    )

    print("Confusion matrix:")
    print(cm_df)

    # --------------------------------------------------------
    # STEP 13 — TP / FP / TN / FN
    # --------------------------------------------------------

    print("\n[STEP 13] Calculating per-class TP/FP/TN/FN")
    print("-" * 75)

    total = cm.sum()

    rows = []

    for class_id, class_name in enumerate(CLASS_NAMES):

        tp = cm[class_id, class_id]

        fn = cm[class_id, :].sum() - tp

        fp = cm[:, class_id].sum() - tp

        tn = total - tp - fn - fp

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        tpr = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        f1 = (
            2 * precision * tpr / (precision + tpr)
            if (precision + tpr) > 0
            else 0.0
        )

        rows.append({
            "class": class_name,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "FPR": fpr,
            "Recall": tpr,
            "Precision": precision,
            "F1": f1
        })

    per_class_df = pd.DataFrame(rows)

    per_class_df.to_csv(
        PER_CLASS_FILE,
        index=False
    )

    print(
        per_class_df.to_string(index=False)
    )

    # --------------------------------------------------------
    # STEP 14 — MACRO FPR
    # --------------------------------------------------------

    macro_fpr = per_class_df["FPR"].mean()

    print("\nMacro FPR:")
    print(f"{macro_fpr:.6f}")

    # --------------------------------------------------------
    # STEP 15 — BINARY ATTACK VS BENIGN
    # --------------------------------------------------------

    print("\n[STEP 15] Binary Attack-vs-BENIGN evaluation")
    print("-" * 75)

    # BENIGN = 0
    # ATTACK = all non-zero classes

    y_test_binary = (
        y_test.to_numpy() != 0
    ).astype(int)

    y_pred_binary = (
        np.asarray(y_pred) != 0
    ).astype(int)

    binary_cm = confusion_matrix(
        y_test_binary,
        y_pred_binary,
        labels=[0, 1]
    )

    binary_tn = binary_cm[0, 0]
    binary_fp = binary_cm[0, 1]
    binary_fn = binary_cm[1, 0]
    binary_tp = binary_cm[1, 1]

    binary_fpr = (
        binary_fp / (binary_fp + binary_tn)
        if (binary_fp + binary_tn) > 0
        else 0.0
    )

    binary_recall = (
        binary_tp / (binary_tp + binary_fn)
        if (binary_tp + binary_fn) > 0
        else 0.0
    )

    print(f"Binary TP:     {binary_tp}")
    print(f"Binary FP:     {binary_fp}")
    print(f"Binary TN:     {binary_tn}")
    print(f"Binary FN:     {binary_fn}")

    print(f"Binary Attack Recall: {binary_recall:.6f}")
    print(f"Binary Attack FPR:    {binary_fpr:.6f}")
    print(f"Binary Attack FPR %:  {binary_fpr * 100:.4f}%")

    # --------------------------------------------------------
    # STEP 16 — SAVE VISUALIZATION
    # --------------------------------------------------------

    print("\n[STEP 16] Creating confusion matrix visualization")
    print("-" * 75)

    os.makedirs(
        FIGURE_DIR,
        exist_ok=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    display.plot(
        ax=ax,
        xticks_rotation=45
    )

    plt.title(
        "Day 23 — Improved XGBoost Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_FILE,
        dpi=300
    )

    plt.close()

    print(f"Figure saved to:")
    print(FIGURE_FILE)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("DAY 23 IMPROVED MODEL EVALUATION COMPLETED")
    print("=" * 75)

    print(f"Accuracy:           {accuracy:.6f}")
    print(f"Macro Precision:    {macro_precision:.6f}")
    print(f"Macro Recall:       {macro_recall:.6f}")
    print(f"Macro F1:           {macro_f1:.6f}")
    print(f"Weighted F1:        {weighted_f1:.6f}")
    print(f"Macro FPR:          {macro_fpr:.6f}")
    print(f"Binary Attack FPR:  {binary_fpr:.6f}")

    print("\nGenerated files:")
    print(f"1. {CLASSIFICATION_FILE}")
    print(f"2. {PER_CLASS_FILE}")
    print(f"3. {CONFUSION_FILE}")
    print(f"4. {FIGURE_FILE}")


if __name__ == "__main__":
    main()