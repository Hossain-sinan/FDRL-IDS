import numpy as np
import pandas as pd

from data_loader import load_cicids2017
from preprocessing import clean_data
from label_processing import process_labels


print("========== DAY 16 FEATURE SELECTION ==========")

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
# Step 4: Identify target/helper columns
# --------------------------------------------------

target_columns = [
    "Label",
    "attack_category",
    "label_id",
    "binary_label"
]

feature_columns = [
    column for column in df.columns
    if column not in target_columns
]


# --------------------------------------------------
# Step 5: Feature inventory
# --------------------------------------------------

print("\n========== FEATURE INVENTORY ==========")

print("Total columns:", len(df.columns))
print("Feature columns:", len(feature_columns))

print("\nTarget/helper columns:")
for column in target_columns:
    if column in df.columns:
        print("-", column)

print("\nFeature columns:")
for number, column in enumerate(feature_columns, start=1):
    print(f"{number}. {column}")


# --------------------------------------------------
# Step 6: Feature data types
# --------------------------------------------------

print("\n========== FEATURE DATA TYPES ==========")

print(df[feature_columns].dtypes.value_counts())


# --------------------------------------------------
# Step 7: Check numeric features
# --------------------------------------------------

numeric_features = df[feature_columns].select_dtypes(
    include=np.number
).columns.tolist()

non_numeric_features = [
    column for column in feature_columns
    if column not in numeric_features
]

print("\nNumeric features:", len(numeric_features))
print("Non-numeric features:", len(non_numeric_features))

if non_numeric_features:
    print("\nNon-numeric features:")
    for column in non_numeric_features:
        print("-", column)


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n========== DAY 16 STEP 1 COMPLETE ==========")
print(f"Total dataset columns: {len(df.columns)}")
print(f"Feature columns: {len(feature_columns)}")
print(f"Numeric features: {len(numeric_features)}")
print(f"Non-numeric features: {len(non_numeric_features)}")

# --------------------------------------------------
# Step 2: Feature quality analysis
# --------------------------------------------------

print("\n========== FEATURE QUALITY ANALYSIS ==========")

# Check missing values
missing_values = df[feature_columns].isna().sum()

print("\nTotal missing values across features:")
print(missing_values.sum())

# Check infinite values
infinite_values = np.isinf(
    df[feature_columns]
).sum()

print("\nTotal infinite values across features:")
print(infinite_values.sum())

# Check constant features
unique_counts = df[feature_columns].nunique()

constant_features = unique_counts[
    unique_counts <= 1
]

print("\nConstant features:")
print(constant_features)

print(
    "\nNumber of constant features:",
    len(constant_features)
)

# Check low-variation features
print("\nFeatures with 2 or fewer unique values:")
print(unique_counts[unique_counts <= 2])

# Feature summary table
feature_quality = pd.DataFrame({
    "data_type": df[feature_columns].dtypes.astype(str),
    "unique_values": unique_counts,
    "missing_values": missing_values,
    "infinite_values": infinite_values
})

print("\n========== FEATURE QUALITY SUMMARY ==========")
print(feature_quality)
# --------------------------------------------------
# Step 3: Create candidate feature set
# --------------------------------------------------

print("\n========== CANDIDATE FEATURE SET ==========")

# Features with only one unique value cannot provide
# useful variation for machine learning.
constant_feature_names = constant_features.index.tolist()

candidate_features = [
    column
    for column in feature_columns
    if column not in constant_feature_names
]

print("Original feature count:", len(feature_columns))
print("Constant feature count:", len(constant_feature_names))
print("Candidate feature count:", len(candidate_features))

print("\nConstant features removed from candidate set:")

for column in constant_feature_names:
    print("-", column)

print("\nRemaining candidate features:")
for number, column in enumerate(candidate_features, start=1):
    print(f"{number}. {column}")


 # --------------------------------------------------
# Step 4: Correlation analysis
# --------------------------------------------------

print("\n========== CORRELATION ANALYSIS ==========")

# Calculate correlation matrix for the 70 candidate features.
correlation_matrix = df[candidate_features].corr()

print("\nCorrelation matrix calculated.")
print("Matrix shape:", correlation_matrix.shape)


# --------------------------------------------------
# Find highly correlated feature pairs
# --------------------------------------------------

correlation_threshold = 0.90

high_correlation_pairs = []

for i in range(len(candidate_features)):
    for j in range(i + 1, len(candidate_features)):

        feature_a = candidate_features[i]
        feature_b = candidate_features[j]

        correlation_value = correlation_matrix.loc[
            feature_a,
            feature_b
        ]

        if abs(correlation_value) >= correlation_threshold:
            high_correlation_pairs.append(
                (
                    feature_a,
                    feature_b,
                    correlation_value
                )
            )


print(
    "\nHighly correlated feature pairs "
    f"(absolute correlation >= {correlation_threshold}):"
)

if high_correlation_pairs:
    for feature_a, feature_b, correlation_value in sorted(
        high_correlation_pairs,
        key=lambda x: abs(x[2]),
        reverse=True
    ):
        print(
            f"{feature_a} <-> {feature_b} "
            f"| correlation = {correlation_value:.4f}"
        )
else:
    print("No highly correlated feature pairs found.")


print(
    "\nNumber of highly correlated pairs:",
    len(high_correlation_pairs)
)   


# --------------------------------------------------
# Step 5: Save correlation analysis results
# --------------------------------------------------

print("\n========== SAVE CORRELATION RESULTS ==========")

import os

os.makedirs("docs/figures", exist_ok=True)

# Save correlation matrix
correlation_matrix.to_csv(
    "docs/day16_correlation_matrix.csv"
)

# Create DataFrame for highly correlated pairs
high_correlation_df = pd.DataFrame(
    high_correlation_pairs,
    columns=[
        "feature_a",
        "feature_b",
        "correlation"
    ]
)

# Save highly correlated pairs
high_correlation_df = high_correlation_df.sort_values(
    by="correlation",
    key=lambda x: x.abs(),
    ascending=False
)

high_correlation_df.to_csv(
    "docs/day16_high_correlation_pairs.csv",
    index=False
)

print(
    "Correlation matrix saved to:",
    "docs/day16_correlation_matrix.csv"
)

print(
    "High-correlation pairs saved to:",
    "docs/day16_high_correlation_pairs.csv"
)

print(
    "Number of high-correlation pairs:",
    len(high_correlation_df)
)

# --------------------------------------------------
# Step 6: Redundancy analysis
# --------------------------------------------------

print("\n========== REDUNDANCY ANALYSIS ==========")

redundant_pairs = [
    ("Total Fwd Packets", "Subflow Fwd Packets"),
    ("Total Backward Packets", "Subflow Bwd Packets"),
    ("Total Length of Fwd Packets", "Subflow Fwd Bytes"),
    ("Total Length of Bwd Packets", "Subflow Bwd Bytes"),
    ("Fwd Packet Length Mean", "Avg Fwd Segment Size"),
    ("Bwd Packet Length Mean", "Avg Bwd Segment Size"),
    ("Fwd Header Length", "Fwd Header Length.1"),
]

print("\nPotentially redundant feature pairs:")

for feature_a, feature_b in redundant_pairs:

    correlation_value = correlation_matrix.loc[
        feature_a,
        feature_b
    ]

    print(
        f"{feature_a} <-> {feature_b} "
        f"| correlation = {correlation_value:.4f}"
    )

print("\nRedundancy analysis completed.")


# --------------------------------------------------
# Step 7: Feature importance analysis
# --------------------------------------------------

print("\n========== FEATURE IMPORTANCE ANALYSIS ==========")

from sklearn.ensemble import ExtraTreesClassifier

# Number of samples used for feature importance
sample_size = 100000
random_state = 42

# --------------------------------------------------
# Class-aware sampling
# --------------------------------------------------
# The dataset is highly imbalanced.
# Some attack classes have very few records.
# We therefore make sure that rare classes are represented.

print("\nCreating class-aware sample...")

class_counts = df["label_id"].value_counts().sort_index()

print("\nAvailable records per class:")
print(class_counts)

# Take up to 5,000 samples from each class first.
# If a class has fewer than 5,000 records,
# use all available records from that class.
samples_per_class = 5000

class_samples = []

for class_id, count in class_counts.items():

    n_samples = min(samples_per_class, count)

    class_data = df[df["label_id"] == class_id].sample(
        n=n_samples,
        random_state=random_state
    )

    class_samples.append(class_data)

# Combine the initial class-aware samples
sample_df = pd.concat(class_samples)

# If we still need more samples to reach 100,000,
# randomly sample additional records from the remaining data.
remaining_needed = sample_size - len(sample_df)

if remaining_needed > 0:

    remaining_df = df.drop(sample_df.index)

    additional_sample = remaining_df.sample(
        n=min(remaining_needed, len(remaining_df)),
        random_state=random_state
    )

    sample_df = pd.concat(
        [sample_df, additional_sample]
    )

# Shuffle the final sample
sample_df = sample_df.sample(
    frac=1,
    random_state=random_state
)

print(f"\nFinal sample size: {len(sample_df)}")
print(f"Random state: {random_state}")

print("\nClass distribution in feature-importance sample:")
print(
    sample_df["label_id"]
    .value_counts()
    .sort_index()
)

print(f"\nSample size: {len(sample_df)}")
print(f"Random state: {random_state}")

# Features
X_sample = sample_df[candidate_features]

# Target
y_sample = sample_df["label_id"]

print("\nTraining ExtraTreesClassifier...")

model = ExtraTreesClassifier(
    n_estimators=100,
    random_state=random_state,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_sample, y_sample)

# Feature importance
importance_df = pd.DataFrame({
    "feature": candidate_features,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
).reset_index(drop=True)

print("\n========== TOP 20 FEATURE IMPORTANCE ==========")

print(
    importance_df.head(20).to_string(index=False)
)

# Save complete importance results
importance_df.to_csv(
    "docs/day16_feature_importance.csv",
    index=False
)

print(
    "\nFeature importance saved to:",
    "docs/day16_feature_importance.csv"
)

print(
    "Total features ranked:",
    len(importance_df)
)

# --------------------------------------------------
# Step 8: Resolve obvious redundant feature pairs
# --------------------------------------------------

print("\n========== REDUNDANCY + IMPORTANCE ANALYSIS ==========")

# Obvious redundant feature pairs identified from correlation analysis
redundant_pairs = [
    ("Total Fwd Packets", "Subflow Fwd Packets"),
    ("Total Backward Packets", "Subflow Bwd Packets"),
    ("Total Length of Fwd Packets", "Subflow Fwd Bytes"),
    ("Total Length of Bwd Packets", "Subflow Bwd Bytes"),
    ("Fwd Packet Length Mean", "Avg Fwd Segment Size"),
    ("Bwd Packet Length Mean", "Avg Bwd Segment Size"),
    ("Fwd Header Length", "Fwd Header Length.1")
]

# Start with all 70 candidate features
selected_after_redundancy = candidate_features.copy()

removed_redundant_features = []

for feature_a, feature_b in redundant_pairs:

    importance_a = importance_df.loc[
        importance_df["feature"] == feature_a,
        "importance"
    ].iloc[0]

    importance_b = importance_df.loc[
        importance_df["feature"] == feature_b,
        "importance"
    ].iloc[0]

    # Keep the feature with higher importance
    if importance_a >= importance_b:
        keep = feature_a
        remove = feature_b
    else:
        keep = feature_b
        remove = feature_a

    if remove in selected_after_redundancy:
        selected_after_redundancy.remove(remove)
        removed_redundant_features.append(remove)

    print(
        f"\nPair: {feature_a} <-> {feature_b}"
    )
    print(
        f"  {feature_a}: {importance_a:.6f}"
    )
    print(
        f"  {feature_b}: {importance_b:.6f}"
    )
    print(
        f"  Keep: {keep}"
    )
    print(
        f"  Remove: {remove}"
    )

print("\n========== REDUNDANCY SUMMARY ==========")

print(
    f"Features before redundancy filtering: "
    f"{len(candidate_features)}"
)

print(
    f"Redundant features removed: "
    f"{len(removed_redundant_features)}"
)

print(
    f"Features after redundancy filtering: "
    f"{len(selected_after_redundancy)}"
)

print("\nRemoved redundant features:")

for feature in removed_redundant_features:
    print(f"- {feature}")

# Save the feature list
redundancy_df = pd.DataFrame({
    "removed_feature": removed_redundant_features
})

redundancy_df.to_csv(
    "docs/day16_removed_redundant_features.csv",
    index=False
)

print(
    "\nRedundancy results saved to:",
    "docs/day16_removed_redundant_features.csv"
)