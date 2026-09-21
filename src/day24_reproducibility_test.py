"""
Day 24 — XGBoost Reproducibility Test

Purpose:
Retrain the Day 23 Config B XGBoost model from scratch and
verify whether the same experiment produces the same results.

Important:
- Same CICIDS2017 dataset
- Same cleaning pipeline
- Same label processing
- Same 20 selected features
- Same train/validation/test split
- Same random_state
- Same XGBoost parameters
- No SMOTE
- No class weighting
- Day 23 model is NOT overwritten
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
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

VALIDATION_SIZE = 0.10
TEST_SIZE = 0.10

NUM_CLASSES = 8


# Day 23 Config B
CONFIG_B = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


FEATURE_FILE = os.path.join(
    PROJECT_ROOT,
    "docs",
    "day16_features_20.csv"
)


DAY23_RESULT_FILE = os.path.join(
    PROJECT_ROOT,
    "docs",
    "day23_xgboost_comparison.csv"
)


REPRO_RESULT_FILE = os.path.join(
    PROJECT_ROOT,
    "docs",
    "day24_reproducibility_results.csv"
)


REPRO_MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "models",
    "xgboost_day24_reproducibility.json"
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
    print("DAY 24 — XGBOOST REPRODUCIBILITY TEST")
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
    # STEP 4 — LOAD SAME 20 FEATURES
    # --------------------------------------------------------

    print("\n[STEP 4] Loading same 20 selected features")
    print("-" * 75)

    feature_df = pd.read_csv(
        FEATURE_FILE
    )

    selected_features = feature_df["feature"].tolist()

    print(f"Selected feature count: {len(selected_features)}")

    if len(selected_features) != 20:
        raise ValueError(
            f"Expected 20 features, found {len(selected_features)}"
        )

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
        raise ValueError(
            "Missing values found in X."
        )

    # --------------------------------------------------------
    # STEP 6 — RECREATE IDENTICAL SPLIT
    # --------------------------------------------------------

    print("\n[STEP 6] Recreating identical stratified split")
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
    # STEP 7 — TRAIN CONFIG B FROM SCRATCH
    # --------------------------------------------------------

    print("\n[STEP 7] Training Day 23 Config B from scratch")
    print("-" * 75)

    print("\nConfig B parameters:")

    for key, value in CONFIG_B.items():
        print(f"  {key:20s}= {value}")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        eval_metric="mlogloss",

        max_depth=CONFIG_B["max_depth"],
        learning_rate=CONFIG_B["learning_rate"],
        n_estimators=CONFIG_B["n_estimators"],

        subsample=CONFIG_B["subsample"],
        colsample_bytree=CONFIG_B["colsample_bytree"],

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

    # --------------------------------------------------------
    # STEP 8 — SAVE SEPARATE DAY 24 MODEL
    # --------------------------------------------------------

    print("\n[STEP 8] Saving Day 24 reproducibility model")
    print("-" * 75)

    os.makedirs(
        os.path.dirname(REPRO_MODEL_FILE),
        exist_ok=True
    )

    model.save_model(
        REPRO_MODEL_FILE
    )

    print(f"Model saved to:")
    print(REPRO_MODEL_FILE)

    # --------------------------------------------------------
    # STEP 9 — GENERATE TEST PREDICTIONS
    # --------------------------------------------------------

    print("\n[STEP 9] Generating test predictions")
    print("-" * 75)

    y_pred = model.predict(X_test)

    print("Prediction completed.")

    # --------------------------------------------------------
    # STEP 10 — CALCULATE METRICS
    # --------------------------------------------------------

    print("\n[STEP 10] Calculating reproducibility metrics")
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

    print(f"\nAccuracy:           {accuracy:.10f}")
    print(f"Macro Precision:    {macro_precision:.10f}")
    print(f"Macro Recall:       {macro_recall:.10f}")
    print(f"Macro F1:           {macro_f1:.10f}")
    print(f"Weighted Precision: {weighted_precision:.10f}")
    print(f"Weighted Recall:    {weighted_recall:.10f}")
    print(f"Weighted F1:        {weighted_f1:.10f}")

    # --------------------------------------------------------
    # STEP 11 — LOAD DAY 23 REFERENCE RESULT
    # --------------------------------------------------------

    print("\n[STEP 11] Loading Day 23 reference result")
    print("-" * 75)

    if not os.path.exists(DAY23_RESULT_FILE):
        raise FileNotFoundError(
            f"Day 23 result file not found:\n{DAY23_RESULT_FILE}"
        )

    day23_df = pd.read_csv(
        DAY23_RESULT_FILE
    )

    config_b_row = day23_df[
        day23_df["configuration"] == "Config_B"
    ]

    if config_b_row.empty:
        raise ValueError(
            "Config_B was not found in Day 23 comparison results."
        )

    day23 = config_b_row.iloc[0]

    # --------------------------------------------------------
    # STEP 12 — COMPARE DAY 23 AND DAY 24
    # --------------------------------------------------------

    print("\n[STEP 12] Comparing Day 23 and Day 24")
    print("-" * 75)

    metrics = [
        ("accuracy", accuracy),
        ("macro_precision", macro_precision),
        ("macro_recall", macro_recall),
        ("macro_f1", macro_f1),
        ("weighted_precision", weighted_precision),
        ("weighted_recall", weighted_recall),
        ("weighted_f1", weighted_f1),
    ]

    comparison_rows = []

    exact_match = True

    tolerance = 1e-12

    for metric_name, day24_value in metrics:

        day23_value = float(
            day23[metric_name]
        )

        difference = day24_value - day23_value

        same = np.isclose(
            day23_value,
            day24_value,
            rtol=0.0,
            atol=tolerance
        )

        if not same:
            exact_match = False

        comparison_rows.append({
            "metric": metric_name,
            "day23_value": day23_value,
            "day24_value": day24_value,
            "difference": difference,
            "reproduced": same
        })

        print(
            f"{metric_name:20s} "
            f"Day23={day23_value:.10f} "
            f"Day24={day24_value:.10f} "
            f"Difference={difference:.12f} "
            f"Match={same}"
        )

    # --------------------------------------------------------
    # STEP 13 — COMPARE PREDICTIONS WITH DAY 23 MODEL
    # --------------------------------------------------------

    print("\n[STEP 13] Comparing predictions with Day 23 saved model")
    print("-" * 75)

    day23_model_file = os.path.join(
        PROJECT_ROOT,
        "models",
        "xgboost_improved_baseline.json"
    )

    if not os.path.exists(day23_model_file):
        raise FileNotFoundError(
            f"Day 23 model not found:\n{day23_model_file}"
        )

    day23_model = XGBClassifier()

    day23_model.load_model(
        day23_model_file
    )

    y_pred_day23 = day23_model.predict(
        X_test
    )

    prediction_match = np.array_equal(
        y_pred_day23,
        y_pred
    )

    different_predictions = np.sum(
        y_pred_day23 != y_pred
    )

    print(
        f"Prediction arrays identical: {prediction_match}"
    )

    print(
        f"Different predictions:        {different_predictions:,}"
    )

    # --------------------------------------------------------
    # STEP 14 — SAVE RESULTS
    # --------------------------------------------------------

    print("\n[STEP 14] Saving Day 24 reproducibility results")
    print("-" * 75)

    results_df = pd.DataFrame(
        comparison_rows
    )

    results_df.to_csv(
        REPRO_RESULT_FILE,
        index=False
    )

    print(
        f"Results saved to:\n{REPRO_RESULT_FILE}"
    )

    # --------------------------------------------------------
    # STEP 15 — FINAL DECISION
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("DAY 24 REPRODUCIBILITY RESULT")
    print("=" * 75)

    print(
        f"Metric reproducibility:     {exact_match}"
    )

    print(
        f"Prediction reproducibility: {prediction_match}"
    )

    if exact_match and prediction_match:

        print(
            "\nRESULT: REPRODUCIBLE"
        )

        print(
            "The Day 23 Config B experiment produced "
            "the same evaluation metrics and predictions "
            "when retrained using the same dataset, "
            "preprocessing, split, seed, and parameters."
        )

    else:

        print(
            "\nRESULT: DIFFERENCE DETECTED"
        )

        print(
            "The Day 24 experiment produced differences "
            "from the Day 23 reference result."
        )

        print(
            "Further investigation is required."
        )

    print("\nGenerated files:")
    print(f"1. {REPRO_RESULT_FILE}")
    print(f"2. {REPRO_MODEL_FILE}")


if __name__ == "__main__":
    main()