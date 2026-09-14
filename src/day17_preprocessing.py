from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from data_loader import load_cicids2017
from preprocessing import clean_data
from label_processing import process_labels


print("========== DAY 17 PREPROCESSING PIPELINE ==========")


# --------------------------------------------------
# Step 1: Load dataset
# --------------------------------------------------

print("\nLoading CICIDS2017 dataset...")

df = load_cicids2017()

print("Original shape:", df.shape)


# --------------------------------------------------
# Step 2: Clean dataset
# --------------------------------------------------

print("\nCleaning dataset...")

df = clean_data(df)

print("Cleaned shape:", df.shape)


# --------------------------------------------------
# Step 3: Process labels
# --------------------------------------------------

print("\nProcessing labels...")

df = process_labels(df)

print("Processed shape:", df.shape)


# --------------------------------------------------
# Step 4: Load selected features
# --------------------------------------------------

project_root = Path(__file__).resolve().parents[1]
feature_file = project_root / "docs" / "day16_features_20.csv"

features_df = pd.read_csv(feature_file)

selected_features = features_df["feature"].tolist()

print("\n========== SELECTED FEATURES ==========")

print("Feature set:", feature_file)
print("Number of features:", len(selected_features))

for number, feature in enumerate(selected_features, start=1):
    print(f"{number}. {feature}")


# --------------------------------------------------
# Step 5: Create X and y
# --------------------------------------------------

X = df[selected_features]
y = df["label_id"]

print("\n========== DATA FOR MODELING ==========")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nClass distribution:")

print(
    y.value_counts()
    .sort_index()
)


# --------------------------------------------------
# Step 6: Train / Validation / Test split
# --------------------------------------------------

print("\n========== DATA SPLITTING ==========")

# First split:
# 80% training
# 20% temporary data
#
# The temporary data will later be divided equally
# into validation and test data.

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Split the temporary 20% into:
# 10% validation
# 10% test

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset split:")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# --------------------------------------------------
# Step 7: Check class distribution
# --------------------------------------------------

print("\n========== CLASS DISTRIBUTION ==========")

print("\nTraining:")
print(y_train.value_counts().sort_index())

print("\nValidation:")
print(y_val.value_counts().sort_index())

print("\nTest:")
print(y_test.value_counts().sort_index())


# --------------------------------------------------
# Step 8: Create Min-Max scaler
# --------------------------------------------------

print("\n========== MIN-MAX SCALING ==========")

scaler = MinMaxScaler()


# IMPORTANT:
# Fit ONLY on training data.

scaler.fit(X_train)

print("Scaler fitted using training data only.")


# --------------------------------------------------
# Step 9: Transform datasets
# --------------------------------------------------

X_train_scaled = scaler.transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)


print("\nScaling completed.")

print("Scaled training shape:", X_train_scaled.shape)
print("Scaled validation shape:", X_val_scaled.shape)
print("Scaled test shape:", X_test_scaled.shape)


# --------------------------------------------------
# Step 10: Verify scaling
# --------------------------------------------------

print("\n========== SCALING CHECK ==========")

print(
    "Training minimum:",
    X_train_scaled.min()
)

print(
    "Training maximum:",
    X_train_scaled.max()
)

print(
    "\nFirst training sample after scaling:"
)

print(X_train_scaled[0])


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n========== DAY 17 STEP 1 COMPLETE ==========")

print("Feature count:", len(selected_features))

print("Training records:", len(X_train))

print("Validation records:", len(X_val))

print("Test records:", len(X_test))

print("\nMin-Max scaler was fitted only on training data.")

print("\nData leakage prevention: PASSED")