"""
Day 20 — XGBoost Centralized Baseline

Purpose:
Train a traditional centralized XGBoost classifier on the
20 selected CICIDS2017 features.

This is the first ML baseline for comparison with the
future Federated Deep Reinforcement Learning IDS.
"""

import os
import sys

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

TRAIN_SIZE = 0.80
VALIDATION_SIZE = 0.10
TEST_SIZE = 0.10

NUM_CLASSES = 8

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_FILE = os.path.join(
    PROJECT_ROOT,
    "docs",
    "day16_features_20.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

DOCS_DIR = os.path.join(
    PROJECT_ROOT,
    "docs"
)

FIGURE_DIR = os.path.join(
    DOCS_DIR,
    "figures"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "xgboost_baseline.json"
)

RESULT_FILE = os.path.join(
    DOCS_DIR,
    "day20_baseline_results.csv"
)

CONFUSION_FILE = os.path.join(
    FIGURE_DIR,
    "day20_confusion_matrix.png"
)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from data_loader import load_cicids2017
from preprocessing import clean_data
from label_processing import process_labels


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DAY 20 — XGBOOST CENTRALIZED BASELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1 — LOAD DATA
    # --------------------------------------------------------

    print("\n[STEP 1] Loading CICIDS2017 dataset")
    print("-" * 70)

    df = load_cicids2017()

    print(f"Raw dataset shape: {df.shape}")

    # --------------------------------------------------------
    # STEP 2 — CLEAN DATA
    # --------------------------------------------------------

    print("\n[STEP 2] Cleaning data")
    print("-" * 70)

    df = clean_data(df)

    print(f"Cleaned dataset shape: {df.shape}")

    # --------------------------------------------------------
    # STEP 3 — PROCESS LABELS
    # --------------------------------------------------------

    print("\n[STEP 3] Processing attack labels")
    print("-" * 70)

    df = process_labels(df)

    print(f"Dataset after label processing: {df.shape}")

    # --------------------------------------------------------
    # STEP 4 — LOAD SELECTED FEATURES
    # --------------------------------------------------------

    print("\n[STEP 4] Loading selected 20 features")
    print("-" * 70)

    feature_df = pd.read_csv(FEATURE_FILE)

    selected_features = feature_df["feature"].tolist()

    print(f"Selected feature count: {len(selected_features)}")

    for i, feature in enumerate(selected_features, start=1):
        print(f"{i:2d}. {feature}")

    # --------------------------------------------------------
    # STEP 5 — CREATE X AND y
    # --------------------------------------------------------

    print("\n[STEP 5] Creating feature matrix and labels")
    print("-" * 70)

    X = df[selected_features].copy()
    y = df["label_id"].copy()

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Safety check
    if X.isnull().sum().sum() > 0:
        raise ValueError("Missing values found in X.")

    if not X.map(lambda x: pd.api.types.is_number(x)).all().all():
        raise ValueError("Non-numeric values found in X.")

    # --------------------------------------------------------
    # STEP 6 — TRAIN / VALIDATION / TEST SPLIT
    # --------------------------------------------------------

    print("\n[STEP 6] Creating stratified train/validation/test split")
    print("-" * 70)

    # First split:
    # 80% train
    # 20% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(VALIDATION_SIZE + TEST_SIZE),
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Second split:
    # temporary 20% -> 10% validation + 10% test
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
    # STEP 7 — CREATE XGBOOST MODEL
    # --------------------------------------------------------

    print("\n[STEP 7] Creating XGBoost model")
    print("-" * 70)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        eval_metric="mlogloss",

        max_depth=3,
        learning_rate=0.1,
        n_estimators=20,

        subsample=0.8,
        colsample_bytree=0.8,

        random_state=RANDOM_STATE,

        tree_method="hist",
        n_jobs=4
    )

    print("XGBoost configuration:")
    print("  objective       = multi:softprob")
    print("  num_class       = 8")
    print("  max_depth       = 3")
    print("  learning_rate   = 0.1")
    print("  n_estimators    = 20")
    print("  subsample       = 0.8")
    print("  colsample       = 0.8")
    print("  random_state    = 42")

    # --------------------------------------------------------
    # STEP 8 — TRAIN
    # --------------------------------------------------------

    print("\n[STEP 8] Training XGBoost")
    print("-" * 70)
    print("Training started...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    print("\nTraining completed.")

    # --------------------------------------------------------
    # STEP 9 — TEST PREDICTION
    # --------------------------------------------------------

    print("\n[STEP 9] Making predictions on test data")
    print("-" * 70)

    y_pred = model.predict(X_test)

    print("Prediction completed.")

    # --------------------------------------------------------
    # STEP 10 — CALCULATE METRICS
    # --------------------------------------------------------

    print("\n[STEP 10] Calculating evaluation metrics")
    print("-" * 70)

    accuracy = accuracy_score(y_test, y_pred)

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

    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Macro Precision:    {macro_precision:.4f}")
    print(f"Macro Recall:       {macro_recall:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall:    {weighted_recall:.4f}")
    print(f"Weighted F1:        {weighted_f1:.4f}")

    # --------------------------------------------------------
    # STEP 11 — CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\n[STEP 11] Classification report")
    print("-" * 70)

    class_names = [
        "BENIGN",
        "DoS",
        "DDoS",
        "Brute Force",
        "Web Attack",
        "Infiltration",
        "Botnet",
        "Port Scan"
    ]

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        zero_division=0
    )

    print(report)

    # --------------------------------------------------------
    # STEP 12 — SAVE MODEL
    # --------------------------------------------------------

    print("\n[STEP 12] Saving trained model")
    print("-" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)

    model.save_model(MODEL_FILE)

    print(f"Model saved to:")
    print(MODEL_FILE)

    # --------------------------------------------------------
    # STEP 13 — SAVE METRICS
    # --------------------------------------------------------

    print("\n[STEP 13] Saving baseline results")
    print("-" * 70)

    results = pd.DataFrame({
        "metric": [
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1"
        ],
        "value": [
            accuracy,
            macro_precision,
            macro_recall,
            macro_f1,
            weighted_precision,
            weighted_recall,
            weighted_f1
        ]
    })

    results.to_csv(
        RESULT_FILE,
        index=False
    )

    print(f"Results saved to:")
    print(RESULT_FILE)

    # --------------------------------------------------------
    # STEP 14 — CONFUSION MATRIX
    # --------------------------------------------------------

    print("\n[STEP 14] Creating confusion matrix")
    print("-" * 70)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    os.makedirs(FIGURE_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        xticks_rotation=45
    )

    plt.title("Day 20 — XGBoost Baseline Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        CONFUSION_FILE,
        dpi=300
    )

    plt.close()

    print(f"Confusion matrix saved to:")
    print(CONFUSION_FILE)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DAY 20 XGBOOST BASELINE COMPLETED")
    print("=" * 70)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nGenerated files:")
    print(f"1. {MODEL_FILE}")
    print(f"2. {RESULT_FILE}")
    print(f"3. {CONFUSION_FILE}")


if __name__ == "__main__":
    main()