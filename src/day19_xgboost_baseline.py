"""
Day 19 — XGBoost Minimal Classification Experiment

Purpose:
    Learn and experimentally evaluate XGBoost using a small dataset
    before applying machine learning to the CICIDS2017 IDS dataset.
"""

import os
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


RANDOM_STATE = 42


def main():

    print("=" * 70)
    print("DAY 19 — XGBOOST MINIMAL CLASSIFICATION EXPERIMENT")
    print("=" * 70)

    # ==============================================================
    # STEP 1 — LOAD DATA
    # ==============================================================

    print("\n[STEP 1] Loading Iris dataset")

    iris = load_iris()

    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    class_names = iris.target_names

    print(f"Samples:  {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes:  {len(np.unique(y))}")

    # ==============================================================
    # STEP 2 — TRAIN / VALIDATION / TEST SPLIT
    # ==============================================================

    print("\n[STEP 2] Creating train/validation/test split")

    # First: 80% temporary data, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Second: half of temporary data becomes validation
    # Final ratio = 60% train / 20% validation / 20% test
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")

    # ==============================================================
    # STEP 3 — XGBOOST EXPERIMENTS
    # ==============================================================

    print("\n[STEP 3] Running XGBoost experiments")

    experiments = [
        {
            "name": "A",
            "max_depth": 2,
            "learning_rate": 0.1,
            "n_estimators": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "name": "B",
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "name": "C",
            "max_depth": 4,
            "learning_rate": 0.1,
            "n_estimators": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "name": "D",
            "max_depth": 3,
            "learning_rate": 0.3,
            "n_estimators": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "name": "E",
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 40,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    ]

    results = []

    for config in experiments:

        print(f"\nExperiment {config['name']}")

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            n_estimators=config["n_estimators"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
        )

        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)

        val_accuracy = accuracy_score(y_val, y_val_pred)

        val_macro_f1 = f1_score(
            y_val,
            y_val_pred,
            average="macro",
        )

        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Macro F1: {val_macro_f1:.4f}")

        results.append(
            {
                **config,
                "validation_accuracy": val_accuracy,
                "validation_macro_f1": val_macro_f1,
            }
        )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)

    print(results_df.to_string(index=False))

    # Save experiment results
    os.makedirs("docs", exist_ok=True)

    results_path = "docs/day19_xgboost_experiments.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\nExperiment results saved to: {results_path}")

    # ==============================================================
    # STEP 4 — SELECT CONFIGURATION
    # ==============================================================

    best_index = results_df["validation_macro_f1"].idxmax()

    best_row = results_df.loc[best_index]

    print("\nSelected configuration based on validation Macro F1:")

    print(best_row.to_dict())

    # ==============================================================
    # STEP 5 — TRAIN FINAL MODEL
    # ==============================================================

    final_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        max_depth=int(best_row["max_depth"]),
        learning_rate=float(best_row["learning_rate"]),
        n_estimators=int(best_row["n_estimators"]),
        subsample=float(best_row["subsample"]),
        colsample_bytree=float(best_row["colsample_bytree"]),
    )

    final_model.fit(X_train, y_train)

    y_test_pred = final_model.predict(X_test)

    # ==============================================================
    # STEP 6 — FINAL TEST RESULTS
    # ==============================================================

    accuracy = accuracy_score(y_test, y_test_pred)

    macro_precision = precision_score(
        y_test,
        y_test_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        y_test_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        y_test_pred,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_test_pred,
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")

    cm = confusion_matrix(y_test, y_test_pred)

    print(cm)

    # ==============================================================
    # STEP 7 — FEATURE IMPORTANCE
    # ==============================================================

    booster = final_model.get_booster()

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE — GAIN")
    print("=" * 70)

    gain = booster.get_score(importance_type="gain")

    gain_named = {}

    for feature, value in gain.items():

        index = int(feature.replace("f", ""))

        gain_named[feature_names[index]] = value

    for feature, value in sorted(
        gain_named.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"{feature:30s} {value:.6f}")

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE — WEIGHT")
    print("=" * 70)

    weight = booster.get_score(importance_type="weight")

    weight_named = {}

    for feature, value in weight.items():

        index = int(feature.replace("f", ""))

        weight_named[feature_names[index]] = value

    print(weight_named)

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE — COVER")
    print("=" * 70)

    cover = booster.get_score(importance_type="cover")

    cover_named = {}

    for feature, value in cover.items():

        index = int(feature.replace("f", ""))

        cover_named[feature_names[index]] = value

    print(cover_named)

    # ==============================================================
    # STEP 8 — FINAL SUMMARY
    # ==============================================================

    print("\n" + "=" * 70)
    print("DAY 19 COMPLETED")
    print("=" * 70)

    print("XGBoost training:       SUCCESS")
    print("Validation experiments: SUCCESS")
    print("Test evaluation:        SUCCESS")
    print("Feature importance:     SUCCESS")
    print(f"Saved results:          {results_path}")


if __name__ == "__main__":
    main()