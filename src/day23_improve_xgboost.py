"""
Day 23 — Improve XGBoost Baseline

Purpose:
Test a small number of reasonable XGBoost parameter configurations
to determine whether the centralized baseline can be improved.

Important:
- Same CICIDS2017 dataset
- Same cleaning pipeline
- Same label processing
- Same 20 selected features
- Same train/validation/test split
- Same random_state
- No SMOTE
- No class weighting
- Day 20 baseline is preserved as the reference
"""

import os
import sys

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

VALIDATION_SIZE = 0.10
TEST_SIZE = 0.10

NUM_CLASSES = 8

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

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

RESULT_FILE = os.path.join(
    DOCS_DIR,
    "day23_xgboost_comparison.csv"
)

BEST_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "xgboost_improved_baseline.json"
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
# MODEL CONFIGURATIONS
# ============================================================

CONFIGURATIONS = {

    "Day20_Original": {
        "max_depth": 3,
        "learning_rate": 0.10,
        "n_estimators": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    "Config_A": {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    "Config_B": {
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    "Config_C": {
        "max_depth": 6,
        "learning_rate": 0.03,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("DAY 23 — IMPROVE XGBOOST BASELINE")
    print("=" * 75)

    # --------------------------------------------------------
    # STEP 1 — LOAD DATA
    # --------------------------------------------------------

    print("\n[STEP 1] Loading CICIDS2017 dataset")
    print("-" * 75)

    df = load_cicids2017()

    print(f"Raw dataset shape: {df.shape}")

    # --------------------------------------------------------
    # STEP 2 — CLEAN DATA
    # --------------------------------------------------------

    print("\n[STEP 2] Cleaning data")
    print("-" * 75)

    df = clean_data(df)

    print(f"Cleaned dataset shape: {df.shape}")

    # --------------------------------------------------------
    # STEP 3 — PROCESS LABELS
    # --------------------------------------------------------

    print("\n[STEP 3] Processing labels")
    print("-" * 75)

    df = process_labels(df)

    print(f"Dataset after label processing: {df.shape}")

    # --------------------------------------------------------
    # STEP 4 — LOAD SELECTED FEATURES
    # --------------------------------------------------------

    print("\n[STEP 4] Loading selected 20 features")
    print("-" * 75)

    feature_df = pd.read_csv(FEATURE_FILE)

    selected_features = feature_df["feature"].tolist()

    print(f"Selected feature count: {len(selected_features)}")

    # --------------------------------------------------------
    # STEP 5 — CREATE X AND y
    # --------------------------------------------------------

    print("\n[STEP 5] Creating X and y")
    print("-" * 75)

    X = df[selected_features].copy()
    y = df["label_id"].copy()

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    if X.isnull().sum().sum() > 0:
        raise ValueError("Missing values found in X.")

    if not X.map(
        lambda x: pd.api.types.is_number(x)
    ).all().all():
        raise ValueError("Non-numeric values found in X.")

    # --------------------------------------------------------
    # STEP 6 — RECREATE SAME SPLIT
    # --------------------------------------------------------

    print("\n[STEP 6] Creating identical stratified split")
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
    # STEP 7 — TRAIN CONFIGURATIONS
    # --------------------------------------------------------

    results = []

    best_model = None
    best_config_name = None
    best_macro_f1 = -1

    for config_name, params in CONFIGURATIONS.items():

        print("\n" + "=" * 75)
        print(f"TRAINING: {config_name}")
        print("=" * 75)

        print("\nParameters:")

        for key, value in params.items():
            print(f"  {key:20s}= {value}")

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=NUM_CLASSES,
            eval_metric="mlogloss",

            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],

            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],

            random_state=RANDOM_STATE,

            tree_method="hist",
            n_jobs=4
        )

        print("\nTraining started...")

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        print("Training completed.")

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        y_pred = model.predict(X_test)

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

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

        print("\nResults:")

        print(f"  Accuracy:           {accuracy:.6f}")
        print(f"  Macro Precision:    {macro_precision:.6f}")
        print(f"  Macro Recall:       {macro_recall:.6f}")
        print(f"  Macro F1:           {macro_f1:.6f}")
        print(f"  Weighted Precision: {weighted_precision:.6f}")
        print(f"  Weighted Recall:    {weighted_recall:.6f}")
        print(f"  Weighted F1:        {weighted_f1:.6f}")

        results.append({
            "configuration": config_name,
            "max_depth": params["max_depth"],
            "learning_rate": params["learning_rate"],
            "n_estimators": params["n_estimators"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
        })

        # ----------------------------------------------------
        # TRACK BEST MODEL
        # ----------------------------------------------------

        if macro_f1 > best_macro_f1:

            best_macro_f1 = macro_f1
            best_model = model
            best_config_name = config_name

    # --------------------------------------------------------
    # STEP 8 — SAVE COMPARISON RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("SAVING COMPARISON RESULTS")
    print("=" * 75)

    os.makedirs(DOCS_DIR, exist_ok=True)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULT_FILE,
        index=False
    )

    print(f"Comparison saved to:")
    print(RESULT_FILE)

    # --------------------------------------------------------
    # STEP 9 — SAVE BEST MODEL
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("BEST CONFIGURATION")
    print("=" * 75)

    print(f"Best configuration: {best_config_name}")
    print(f"Best Macro F1:      {best_macro_f1:.6f}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    best_model.save_model(
        BEST_MODEL_FILE
    )

    print("\nBest model saved to:")
    print(BEST_MODEL_FILE)

    # --------------------------------------------------------
    # STEP 10 — DISPLAY FINAL TABLE
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)

    print(
        results_df[
            [
                "configuration",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("DAY 23 EXPERIMENT COMPLETED")
    print("=" * 75)

    print(f"Configurations tested: {len(CONFIGURATIONS)}")
    print(f"Best configuration:     {best_config_name}")
    print(f"Best Macro F1:          {best_macro_f1:.6f}")

    print("\nGenerated files:")
    print(f"1. {RESULT_FILE}")
    print(f"2. {BEST_MODEL_FILE}")


if __name__ == "__main__":
    main()