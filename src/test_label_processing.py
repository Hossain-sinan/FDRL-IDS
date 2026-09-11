from data_loader import load_cicids2017
from preprocessing import clean_data
from label_processing import process_labels


print("========== DAY 15 LABEL PROCESSING TEST ==========")

# Load raw dataset
print("\nLoading dataset...")
df = load_cicids2017()

print("Original shape:", df.shape)

# Clean dataset
print("\nCleaning dataset...")
df = clean_data(df)

print("Cleaned shape:", df.shape)

# Process labels
print("\nProcessing labels...")
df = process_labels(df)

print("Processed shape:", df.shape)


# --------------------------------------------------
# Verification 1: Check the number of classes
# --------------------------------------------------

expected_classes = {
    "BENIGN",
    "DoS",
    "DDoS",
    "Brute Force",
    "Web Attack",
    "Infiltration",
    "Botnet",
    "Port Scan"
}

actual_classes = set(df["attack_category"].unique())

print("\n========== VERIFICATION ==========")

print("\nExpected classes:")
print(expected_classes)

print("\nActual classes:")
print(actual_classes)

assert actual_classes == expected_classes
print("PASS: All 8 expected classes are present.")


# --------------------------------------------------
# Verification 2: Check numeric label IDs
# --------------------------------------------------

expected_ids = set(range(8))
actual_ids = set(df["label_id"].unique())

print("\nExpected label IDs:")
print(expected_ids)

print("\nActual label IDs:")
print(actual_ids)

assert actual_ids == expected_ids
print("PASS: Label IDs are exactly 0-7.")


# --------------------------------------------------
# Verification 3: Check binary labels
# --------------------------------------------------

binary_values = set(df["binary_label"].unique())

print("\nBinary label values:")
print(binary_values)

assert binary_values == {0, 1}
print("PASS: Binary labels contain only 0 and 1.")


# --------------------------------------------------
# Verification 4: Check missing labels
# --------------------------------------------------

missing_categories = df["attack_category"].isna().sum()
missing_ids = df["label_id"].isna().sum()
missing_binary = df["binary_label"].isna().sum()

print("\nMissing attack categories:", missing_categories)
print("Missing label IDs:", missing_ids)
print("Missing binary labels:", missing_binary)

assert missing_categories == 0
assert missing_ids == 0
assert missing_binary == 0

print("PASS: No missing processed labels.")


# --------------------------------------------------
# Verification 5: Check final row count
# --------------------------------------------------

expected_rows = 2520787
actual_rows = len(df)

print("\nExpected processed rows:", expected_rows)
print("Actual processed rows:", actual_rows)

assert actual_rows == expected_rows
print("PASS: Final processed row count is correct.")


# --------------------------------------------------
# Final class distribution
# --------------------------------------------------

print("\n========== FINAL MULTI-CLASS DISTRIBUTION ==========")
print(df["attack_category"].value_counts())

print("\n========== FINAL NUMERIC DISTRIBUTION ==========")
print(df["label_id"].value_counts().sort_index())

print("\n========== FINAL BINARY DISTRIBUTION ==========")
print(df["binary_label"].value_counts())

print("\n==============================================")
print("DAY 15 LABEL PROCESSING TEST PASSED")
print("==============================================")