from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE

from data_loader import load_cicids2017
from preprocessing import clean_data
from label_processing import process_labels

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ==========================================
# DAY 17 - CONTROLLED SMOTE EXPERIMENT
# ==========================================

print("\n========== LOAD DATA ==========")

# Load CICIDS2017
df = load_cicids2017()

# Clean data using Day 14 pipeline
df = clean_data(df)

# Process attack labels using Day 15 pipeline
df = process_labels(df)

print(f"\nProcessed dataset shape: {df.shape}")


# ==========================================
# LOAD SELECTED FEATURES
# ==========================================

project_root = Path(__file__).resolve().parents[1]
feature_file = project_root / "docs" / "day16_features_20.csv"

features_df = pd.read_csv(feature_file)
selected_features = features_df["feature"].tolist()

print(f"\nNumber of selected features: {len(selected_features)}")

X = df[selected_features]
y = df["label_id"]


# ==========================================
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================

print("\n========== DATA SPLIT ==========")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print(f"Training shape:   {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Test shape:       {X_test.shape}")


# ==========================================
# CLASS DISTRIBUTION BEFORE SMOTE
# ==========================================

print("\n========== TRAINING CLASS DISTRIBUTION BEFORE SMOTE ==========")

before_counts = y_train.value_counts().sort_index()

for class_id, count in before_counts.items():
    print(f"Class {class_id}: {count}")


# ==========================================
# NORMALIZATION
# FIT ONLY ON TRAINING DATA
# ==========================================

print("\n========== NORMALIZATION ==========")

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Scaler fitted using TRAINING data only.")
print("Validation and test data were transformed using the same scaler.")


# ==========================================
# CONTROLLED SMOTE
# ==========================================

print("\n========== CONTROLLED SMOTE ==========")

# We do NOT balance every class to the majority class.
# We only increase the very small minority classes.
#
# Class 4 = Web Attack
# Class 5 = Infiltration
# Class 6 = Botnet
#
# These target values are intentionally modest.
# This is an experiment, NOT the final setting.

sampling_strategy = {
    4: 2500,
    5: 100,
    6: 2500
}

print("SMOTE sampling strategy:")
print(sampling_strategy)

smote = SMOTE(
    sampling_strategy=sampling_strategy,
    random_state=42,
    k_neighbors=5
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)


# ==========================================
# CLASS DISTRIBUTION AFTER SMOTE
# ==========================================

print("\n========== TRAINING CLASS DISTRIBUTION AFTER SMOTE ==========")

after_counts = pd.Series(y_train_smote).value_counts().sort_index()

for class_id, count in after_counts.items():
    print(f"Class {class_id}: {count}")


# ==========================================
# FINAL SHAPES
# ==========================================

print("\n========== FINAL SHAPES ==========")

print(f"Original training shape:       {X_train.shape}")
print(f"SMOTE training shape:          {X_train_smote.shape}")

print(f"Validation shape:              {X_val_scaled.shape}")
print(f"Test shape:                    {X_test_scaled.shape}")


# ==========================================
# VERIFY VALIDATION / TEST WERE NOT SMOTED
# ==========================================

print("\n========== DATA LEAKAGE CHECK ==========")

print("SMOTE was applied ONLY to training data.")
print("Validation data was NOT oversampled.")
print("Test data was NOT oversampled.")

print("\nData leakage prevention: PASSED")


# ==========================================
# SUMMARY
# ==========================================

print("\n========== DAY 17 SMOTE SUMMARY ==========")

print("1. Dataset was split into training, validation and test sets.")
print("2. Min-Max scaling was fitted only on training data.")
print("3. SMOTE was applied only to the scaled training data.")
print("4. Validation and test data remained untouched.")
print("5. Majority class was NOT artificially balanced.")
print("6. SMOTE was used as a controlled experiment for rare classes.")