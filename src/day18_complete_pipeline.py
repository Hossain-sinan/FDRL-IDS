from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from data_loader import load_cicids2017
from preprocessing import clean_data
from label_processing import process_labels


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TRAIN_SIZE = 0.80
VALIDATION_SIZE = 0.10
TEST_SIZE = 0.10

# Small tolerance for floating-point precision
SCALING_TOLERANCE = 1e-10


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = PROJECT_ROOT / "docs" / "day16_features_20.csv"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("DAY 18 — COMPLETE FDRL-IDS DATA PIPELINE")
print("=" * 70)


# ============================================================
# STEP 1 — LOAD RAW DATA
# ============================================================

print("\n[STEP 1] LOADING RAW CICIDS2017 DATA")
print("-" * 70)

df = load_cicids2017()

print(f"Raw dataset shape: {df.shape}")


# ============================================================
# STEP 2 — CLEAN DATA
# ============================================================

print("\n[STEP 2] CLEANING DATA")
print("-" * 70)

df = clean_data(df)

print(f"Cleaned dataset shape: {df.shape}")


# ============================================================
# STEP 3 — PROCESS LABELS
# ============================================================

print("\n[STEP 3] PROCESSING LABELS")
print("-" * 70)

df = process_labels(df)

print(f"Dataset shape after label processing: {df.shape}")

print("\nClass distribution:")
print(
    df["label_id"]
    .value_counts()
    .sort_index()
)


# ============================================================
# STEP 4 — LOAD SELECTED FEATURES
# ============================================================

print("\n[STEP 4] LOADING SELECTED FEATURES")
print("-" * 70)

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"Feature selection file not found:\n{FEATURE_FILE}"
    )

features_df = pd.read_csv(FEATURE_FILE)

if "feature" not in features_df.columns:
    raise ValueError(
        "The feature selection file must contain a 'feature' column."
    )

selected_features = (
    features_df["feature"]
    .dropna()
    .astype(str)
    .tolist()
)

print(f"Feature file: {FEATURE_FILE}")
print(f"Number of selected features: {len(selected_features)}")

print("\nSelected features:")

for index, feature in enumerate(selected_features, start=1):
    print(f"{index:2}. {feature}")


# ============================================================
# STEP 5 — CHECK SELECTED FEATURES
# ============================================================

print("\n[STEP 5] CHECKING SELECTED FEATURES")
print("-" * 70)

missing_features = [
    feature
    for feature in selected_features
    if feature not in df.columns
]

if missing_features:
    print("Missing features:")

    for feature in missing_features:
        print(f"  - {feature}")

    raise ValueError(
        "Some selected features are missing from the dataset."
    )

print("Feature availability check: PASSED")


# ============================================================
# STEP 6 — CREATE X AND y
# ============================================================

print("\n[STEP 6] CREATING FEATURES AND LABELS")
print("-" * 70)

X = df[selected_features].copy()
y = df["label_id"].copy()

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ============================================================
# STEP 7 — DATA QUALITY CHECK
# ============================================================

print("\n[STEP 7] CHECKING DATA QUALITY")
print("-" * 70)

nan_count = X.isna().sum().sum()

numeric_X = X.select_dtypes(include=np.number)

infinity_count = (
    np.isinf(numeric_X)
    .sum()
    .sum()
)

print(f"NaN values: {nan_count}")
print(f"Infinity values: {infinity_count}")

if nan_count > 0:
    raise ValueError(
        "NaN values found in selected features."
    )

if infinity_count > 0:
    raise ValueError(
        "Infinity values found in selected features."
    )

print("Data quality check: PASSED")


# ============================================================
# STEP 8 — TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n[STEP 8] TRAIN / VALIDATION / TEST SPLIT")
print("-" * 70)

# First split:
# 80% training
# 20% temporary data

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


# Second split:
# 10% validation
# 10% test

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print(f"Training:   {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test:       {X_test.shape}")


# ============================================================
# STEP 9 — VERIFY SPLIT
# ============================================================

print("\n[STEP 9] VERIFYING SPLIT")
print("-" * 70)

original_records = len(X)

records_after_split = (
    len(X_train)
    + len(X_val)
    + len(X_test)
)

print(f"Original modeling records: {original_records}")
print(f"Records after split:       {records_after_split}")

if original_records != records_after_split:
    raise ValueError(
        "Record count mismatch after train/validation/test split."
    )

print("Split record count check: PASSED")


# ------------------------------------------------------------
# Training distribution
# ------------------------------------------------------------

print("\nTraining class distribution:")
print(
    y_train
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# Validation distribution
# ------------------------------------------------------------

print("\nValidation class distribution:")
print(
    y_val
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# Test distribution
# ------------------------------------------------------------

print("\nTest class distribution:")
print(
    y_test
    .value_counts()
    .sort_index()
)


# ============================================================
# STEP 10 — MIN-MAX SCALING
# ============================================================

print("\n[STEP 10] MIN-MAX SCALING")
print("-" * 70)

print("Scaler fitting dataset: TRAINING ONLY")

scaler = MinMaxScaler()

# IMPORTANT:
# Fit scaler ONLY on training data.
scaler.fit(X_train)

# Transform all datasets using the training-fitted scaler.
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Scaled training shape:   {X_train_scaled.shape}")
print(f"Scaled validation shape: {X_val_scaled.shape}")
print(f"Scaled test shape:       {X_test_scaled.shape}")


# ============================================================
# STEP 11 — VALIDATE SCALING
# ============================================================

print("\n[STEP 11] VALIDATING SCALING")
print("-" * 70)

train_min = X_train_scaled.min()
train_max = X_train_scaled.max()

print(f"Training scaled minimum: {train_min}")
print(f"Training scaled maximum: {train_max}")

# ------------------------------------------------------------
# Floating-point precision handling
# ------------------------------------------------------------
#
# Sometimes MinMaxScaler can produce values such as:
#
# 1.0000000000000002
#
# This is practically 1.0 and happens because of floating-point
# numerical precision.
#
# Therefore, we use a very small tolerance instead of requiring
# an exact mathematical comparison.
# ------------------------------------------------------------

if (
    train_min < -SCALING_TOLERANCE
    or train_max > 1 + SCALING_TOLERANCE
):
    raise ValueError(
        "Training scaled values are outside the expected [0, 1] range."
    )

print("Scaling range check: PASSED")


# ============================================================
# STEP 12 — FINAL NUMERICAL VALIDATION
# ============================================================

print("\n[STEP 12] FINAL NUMERICAL VALIDATION")
print("-" * 70)

train_finite = np.isfinite(X_train_scaled).all()
val_finite = np.isfinite(X_val_scaled).all()
test_finite = np.isfinite(X_test_scaled).all()

print(f"Training values finite:   {train_finite}")
print(f"Validation values finite: {val_finite}")
print(f"Test values finite:       {test_finite}")

if not train_finite:
    raise ValueError(
        "Training data contains NaN or Infinity after scaling."
    )

if not val_finite:
    raise ValueError(
        "Validation data contains NaN or Infinity after scaling."
    )

if not test_finite:
    raise ValueError(
        "Test data contains NaN or Infinity after scaling."
    )

print("Final numerical validation: PASSED")


# ============================================================
# STEP 13 — FINAL PIPELINE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DAY 18 CORE DATA PIPELINE SUMMARY")
print("=" * 70)

print(f"Raw records:              {2830743:,}")
print(f"Cleaned records:          {len(df):,}")
print(f"Selected features:        {len(selected_features)}")

print(f"Training records:         {len(X_train):,}")
print(f"Validation records:       {len(X_val):,}")
print(f"Test records:             {len(X_test):,}")

print(f"Training features:        {X_train_scaled.shape[1]}")
print(f"Validation features:      {X_val_scaled.shape[1]}")
print(f"Test features:            {X_test_scaled.shape[1]}")

print("\nPipeline status:")
print("  Raw CSV loading          : PASSED")
print("  Data cleaning            : PASSED")
print("  Label processing         : PASSED")
print("  Feature selection        : PASSED")
print("  Data quality check       : PASSED")
print("  Train/Val/Test split     : PASSED")
print("  Min-Max scaling          : PASSED")
print("  Numerical validation     : PASSED")

print("\n" + "=" * 70)
print("DAY 18 CORE PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nNOTE:")
print("Non-IID federated client partitioning will be implemented")
print("separately using the original CICIDS2017 source/day information.")
print("=" * 70)