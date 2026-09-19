"""
Day 22 — Confusion Matrix

Purpose:
Generate a confusion matrix for the saved Day 20 XGBoost model.

Important:
- The model is NOT retrained.
- The same preprocessing pipeline from Day 21 is used.
- The same test split is recreated.
- The confusion matrix is saved as CSV and PNG.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

FIGURES_DIR = os.path.join(
    DOCS_DIR,
    "figures"
)

MATRIX_CSV = os.path.join(
    DOCS_DIR,
    "day22_confusion_matrix.csv"
)

MATRIX_IMAGE = os.path.join(
    FIGURES_DIR,
    "day22_confusion_matrix.png"
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

    print("=" * 70)
    print("DAY 22 — CONFUSION MATRIX")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1 — CHECK MODEL
    # --------------------------------------------------------

    print("\n[STEP 1] Checking saved XGBoost model")
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
    # STEP 7 — RECREATE DAY 20 TEST SPLIT
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
    # STEP 8 — LOAD SAVED MODEL
    # --------------------------------------------------------

    print("\n[STEP 8] Loading saved Day 20 XGBoost model")
    print("-" * 70)

    model = XGBClassifier()

    model.load_model(MODEL_FILE)

    print("Saved model loaded successfully.")
    print("No retraining performed.")

    # --------------------------------------------------------
    # STEP 9 — GENERATE PREDICTIONS
    # --------------------------------------------------------

    print("\n[STEP 9] Generating predictions")
    print("-" * 70)

    y_pred = model.predict(X_test)

    y_pred = np.asarray(y_pred).astype(int)

    print(f"Predictions generated: {len(y_pred):,}")

    # --------------------------------------------------------
    # STEP 10 — CREATE CONFUSION MATRIX
    # --------------------------------------------------------

    print("\n[STEP 10] Creating confusion matrix")
    print("-" * 70)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # STEP 11 — SAVE MATRIX AS CSV
    # --------------------------------------------------------

    print("\n[STEP 11] Saving confusion matrix CSV")
    print("-" * 70)

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    cm_df.to_csv(
        MATRIX_CSV
    )

    print(f"Saved: {MATRIX_CSV}")

    # --------------------------------------------------------
    # STEP 12 — CALCULATE TP / FP / FN / TN
    # --------------------------------------------------------

    print("\n[STEP 12] Calculating TP, FP, FN and TN")
    print("-" * 70)

    total = cm.sum()

    metrics = []

    for class_id, class_name in enumerate(CLASS_NAMES):

        tp = cm[class_id, class_id]

        fp = cm[:, class_id].sum() - tp

        fn = cm[class_id, :].sum() - tp

        tn = total - tp - fp - fn

        metrics.append({
            "label_id": class_id,
            "class": class_name,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn
        })

        print(
            f"{class_name:15s} "
            f"TP={tp:8,d} "
            f"FP={fp:8,d} "
            f"TN={tn:8,d} "
            f"FN={fn:8,d}"
        )

    metrics_df = pd.DataFrame(metrics)

    metrics_file = os.path.join(
        DOCS_DIR,
        "day22_confusion_matrix_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_file,
        index=False
    )

    print(f"\nSaved: {metrics_file}")

    # --------------------------------------------------------
    # STEP 13 — GENERATE GRAPH
    # --------------------------------------------------------

    print("\n[STEP 13] Generating confusion matrix graph")
    print("-" * 70)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 10))

    # Use a clear blue heatmap
    image = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues"
    )

    ax.set_title(
        "Day 22 — Centralized XGBoost Confusion Matrix",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel(
        "Predicted Label",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel(
        "True Label",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))

    ax.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right",
        fontsize=10
    )

    ax.set_yticklabels(
        CLASS_NAMES,
        fontsize=10
    )

    # Add values inside every cell
    threshold = cm.max() / 2

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):

            value = cm[i, j]

            if value > threshold:
                text_color = "white"
            else:
                text_color = "black"

            ax.text(
                j,
                i,
                f"{value:,}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold"
            )

    # Add color scale
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(
        "Number of Samples",
        rotation=270,
        labelpad=20,
        fontsize=11
    )

    plt.tight_layout()
    plt.savefig(
        MATRIX_IMAGE,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Saved: {MATRIX_IMAGE}")

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DAY 22 CONFUSION MATRIX COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")

    print(f"1. {MATRIX_CSV}")
    print(f"2. {metrics_file}")
    print(f"3. {MATRIX_IMAGE}")

    print("\nImportant:")
    print("Diagonal cells = correct predictions.")
    print("Off-diagonal cells = classification mistakes.")


if __name__ == "__main__":
    main()