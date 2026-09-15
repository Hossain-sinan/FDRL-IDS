from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from preprocessing import clean_data
from label_processing import process_labels


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
SCALING_TOLERANCE = 1e-10

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

FEATURE_FILE = (
    PROJECT_ROOT
    / "docs"
    / "day16_features_20.csv"
)


# ============================================================
# FEDERATED CLIENT DEFINITION
# ============================================================

CLIENT_FILES = {

    "Hospital": [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
    ],

    "Bank": [
        "Wednesday-workingHours.pcap_ISCX.csv",
    ],

    "University": [
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    ],

    "ISP": [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    ],
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("DAY 18 — NON-IID FEDERATED CLIENT PARTITION")
print("=" * 70)


# ============================================================
# STEP 1 — LOAD SELECTED FEATURES
# ============================================================

print("\n[STEP 1] LOADING SELECTED FEATURES")
print("-" * 70)

if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"Feature file not found:\n{FEATURE_FILE}"
    )


features_df = pd.read_csv(FEATURE_FILE)


if "feature" not in features_df.columns:

    raise ValueError(
        "Feature file must contain a 'feature' column."
    )


selected_features = (
    features_df["feature"]
    .dropna()
    .astype(str)
    .tolist()
)


print(
    f"Selected feature count: "
    f"{len(selected_features)}"
)


# ============================================================
# STEP 2 — CHECK SOURCE FILES
# ============================================================

print("\n[STEP 2] CHECKING CLIENT SOURCE FILES")
print("-" * 70)

total_files = 0

for client, filenames in CLIENT_FILES.items():

    print(f"\n{client}:")

    for filename in filenames:

        file_path = RAW_DIR / filename

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required source file not found:\n"
                f"{file_path}"
            )

        print(f"  ✓ {filename}")

        total_files += 1


print(
    f"\nTotal source files: "
    f"{total_files}"
)


if total_files != 8:

    raise ValueError(
        "Expected exactly 8 CICIDS2017 source files."
    )


print("Source file check: PASSED")


# ============================================================
# STEP 3 — LOAD, CLEAN AND LABEL EACH SOURCE FILE
# ============================================================

print("\n[STEP 3] PROCESSING ORIGINAL SOURCE FILES")
print("-" * 70)

client_data = {}


for client, filenames in CLIENT_FILES.items():

    print("\n" + "=" * 70)
    print(f"CLIENT: {client}")
    print("=" * 70)

    processed_files = []

    for filename in filenames:

        print("\n" + "-" * 60)
        print(f"SOURCE FILE: {filename}")
        print("-" * 60)

        file_path = RAW_DIR / filename


        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        df = pd.read_csv(file_path)

        print(
            f"Raw shape: {df.shape}"
        )


        # ----------------------------------------------------
        # STRIP COLUMN NAMES
        # ----------------------------------------------------

        df.columns = df.columns.str.strip()


        # Verify label column

        if "Label" not in df.columns:

            raise ValueError(
                f"'Label' column not found in {filename}."
            )


        # ----------------------------------------------------
        # CLEAN
        # ----------------------------------------------------

        print("\nCleaning source file...")

        df = clean_data(df)

        print(
            f"Cleaned shape: {df.shape}"
        )


        # ----------------------------------------------------
        # LABEL PROCESSING
        # ----------------------------------------------------

        print("\nProcessing labels...")

        df = process_labels(df)

        print(
            f"After label processing: "
            f"{df.shape}"
        )


        # ----------------------------------------------------
        # SOURCE IDENTITY
        # ----------------------------------------------------

        df["_source_file"] = filename

        df["_client"] = client


        # ----------------------------------------------------
        # CHECK FEATURES
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in selected_features
            if feature not in df.columns
        ]

        if missing_features:

            raise ValueError(
                f"Missing selected features in "
                f"{filename}:\n"
                f"{missing_features}"
            )


        # ----------------------------------------------------
        # SELECT FEATURES + LABEL INFORMATION
        # ----------------------------------------------------

        keep_columns = (
            selected_features
            + [
                "label_id",
                "binary_label",
                "attack_category",
                "_source_file",
                "_client",
            ]
        )


        df = df[keep_columns].copy()


        # ----------------------------------------------------
        # DATA QUALITY
        # ----------------------------------------------------

        X_check = df[selected_features]

        nan_count = (
            X_check
            .isna()
            .sum()
            .sum()
        )

        infinity_count = (
            np.isinf(
                X_check
                .select_dtypes(include=np.number)
            )
            .sum()
            .sum()
        )


        if nan_count > 0:

            raise ValueError(
                f"{filename}: "
                f"{nan_count} NaN values found."
            )


        if infinity_count > 0:

            raise ValueError(
                f"{filename}: "
                f"{infinity_count} infinity values found."
            )


        print(
            "Data quality: PASSED"
        )


        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        processed_files.append(df)


    # ========================================================
    # COMBINE FILES BELONGING TO THIS CLIENT
    # ========================================================

    client_df = pd.concat(
        processed_files,
        ignore_index=True
    )


    client_data[client] = client_df


    print("\n" + "-" * 60)

    print(
        f"{client} final records: "
        f"{len(client_df):,}"
    )

    print(
        f"{client} features: "
        f"{len(selected_features)}"
    )


# ============================================================
# STEP 4 — CLIENT SUMMARY BEFORE SPLIT
# ============================================================

print("\n" + "=" * 70)
print("[STEP 4] CLIENT DATA SUMMARY")
print("=" * 70)


for client, df in client_data.items():

    print("\n" + "-" * 60)

    print(
        f"{client}"
    )

    print("-" * 60)

    print(
        f"Total records: "
        f"{len(df):,}"
    )

    print("\nClass distribution:")

    distribution = (
        df["label_id"]
        .value_counts()
        .sort_index()
    )

    for label_id, count in distribution.items():

        percentage = (
            count
            / len(df)
            * 100
        )

        print(
            f"  Class {label_id}: "
            f"{count:,} "
            f"({percentage:.4f}%)"
        )


# ============================================================
# STEP 5 — CLIENT-SPECIFIC TRAIN / VAL / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("[STEP 5] CLIENT-SPECIFIC TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)


client_splits = {}


for client, df in client_data.items():

    print("\n" + "-" * 60)
    print(f"CLIENT: {client}")
    print("-" * 60)


    X = df[selected_features].copy()

    y = df["label_id"].copy()


    print(
        f"Total records: "
        f"{len(X):,}"
    )


    # --------------------------------------------------------
    # Check class counts before stratification
    # --------------------------------------------------------

    class_counts = (
        y.value_counts()
        .sort_index()
    )


    print("\nClass counts:")

    print(class_counts)


    # --------------------------------------------------------
    # Stratified 80/20 split
    # --------------------------------------------------------

    try:

        X_train, X_temp, y_train, y_temp = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=RANDOM_STATE,
                stratify=y,
            )
        )

    except ValueError as error:

        print(
            f"\nWARNING: Stratified split failed "
            f"for {client}."
        )

        print(
            f"Reason: {error}"
        )

        raise


    # --------------------------------------------------------
    # Stratified 10/10 validation/test split
    # --------------------------------------------------------

    try:

        X_val, X_test, y_val, y_test = (
            train_test_split(
                X_temp,
                y_temp,
                test_size=0.50,
                random_state=RANDOM_STATE,
                stratify=y_temp,
            )
        )

    except ValueError as error:

        print(
            f"\nWARNING: Validation/test stratified split "
            f"failed for {client}."
        )

        print(
            f"Reason: {error}"
        )

        raise


    # --------------------------------------------------------
    # Verify counts
    # --------------------------------------------------------

    total_after_split = (
        len(X_train)
        + len(X_val)
        + len(X_test)
    )


    if total_after_split != len(X):

        raise ValueError(
            f"{client}: "
            f"record count mismatch."
        )


    client_splits[client] = {

        "X_train": X_train,
        "y_train": y_train,

        "X_val": X_val,
        "y_val": y_val,

        "X_test": X_test,
        "y_test": y_test,
    }


    print(
        f"Training:   {X_train.shape}"
    )

    print(
        f"Validation: {X_val.shape}"
    )

    print(
        f"Test:       {X_test.shape}"
    )

    print(
        "Split verification: PASSED"
    )


# ============================================================
# STEP 6 — CLIENT-SPECIFIC MIN-MAX SCALING
# ============================================================

print("\n" + "=" * 70)
print("[STEP 6] CLIENT-SPECIFIC MIN-MAX SCALING")
print("=" * 70)


scaled_clients = {}


for client, data in client_splits.items():

    print("\n" + "-" * 60)

    print(
        f"SCALING CLIENT: {client}"
    )

    print("-" * 60)


    scaler = MinMaxScaler()


    X_train = data["X_train"]

    X_val = data["X_val"]

    X_test = data["X_test"]


    # --------------------------------------------------------
    # IMPORTANT:
    # Fit ONLY on this client's training data.
    # --------------------------------------------------------

    scaler.fit(X_train)


    X_train_scaled = scaler.transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # --------------------------------------------------------
    # Numerical validation
    # --------------------------------------------------------

    if not np.isfinite(
        X_train_scaled
    ).all():

        raise ValueError(
            f"{client}: "
            f"training contains NaN/Infinity."
        )


    if not np.isfinite(
        X_val_scaled
    ).all():

        raise ValueError(
            f"{client}: "
            f"validation contains NaN/Infinity."
        )


    if not np.isfinite(
        X_test_scaled
    ).all():

        raise ValueError(
            f"{client}: "
            f"test contains NaN/Infinity."
        )


    train_min = X_train_scaled.min()

    train_max = X_train_scaled.max()


    print(
        f"Training scaled minimum: "
        f"{train_min}"
    )

    print(
        f"Training scaled maximum: "
        f"{train_max}"
    )


    if (
        train_min < -SCALING_TOLERANCE
        or train_max > 1 + SCALING_TOLERANCE
    ):

        raise ValueError(
            f"{client}: "
            f"scaled values outside expected range."
        )


    print(
        "Scaling validation: PASSED"
    )


    scaled_clients[client] = {

        "X_train": X_train_scaled,
        "y_train": data["y_train"].to_numpy(),

        "X_val": X_val_scaled,
        "y_val": data["y_val"].to_numpy(),

        "X_test": X_test_scaled,
        "y_test": data["y_test"].to_numpy(),

        "scaler": scaler,
    }


# ============================================================
# STEP 7 — FINAL NON-IID DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("[STEP 7] FINAL NON-IID CLIENT DISTRIBUTIONS")
print("=" * 70)


for client, data in scaled_clients.items():

    print("\n" + "-" * 60)

    print(
        f"{client.upper()} TRAINING DISTRIBUTION"
    )

    print("-" * 60)


    y_train = pd.Series(
        data["y_train"]
    )


    distribution = (
        y_train
        .value_counts()
        .sort_index()
    )


    for label_id, count in distribution.items():

        percentage = (
            count
            / len(y_train)
            * 100
        )


        print(
            f"Class {label_id}: "
            f"{count:,} "
            f"({percentage:.4f}%)"
        )


# ============================================================
# STEP 8 — FINAL CLIENT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("[STEP 8] FINAL CLIENT SUMMARY")
print("=" * 70)


total_train = 0
total_val = 0
total_test = 0


for client, data in scaled_clients.items():

    train_count = len(
        data["y_train"]
    )

    val_count = len(
        data["y_val"]
    )

    test_count = len(
        data["y_test"]
    )


    total_train += train_count

    total_val += val_count

    total_test += test_count


    print(
        f"{client:12} | "
        f"Train: {train_count:>9,} | "
        f"Val: {val_count:>8,} | "
        f"Test: {test_count:>8,}"
    )


print("-" * 70)


print(
    f"{'TOTAL':12} | "
    f"Train: {total_train:>9,} | "
    f"Val: {total_val:>8,} | "
    f"Test: {total_test:>8,}"
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("DAY 18 NON-IID CLIENT PARTITION COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    """
Client mapping:

Hospital   = Monday + Tuesday
Bank       = Wednesday
University = Thursday
ISP        = Friday

Research properties:

✓ Original CICIDS2017 source/day identity preserved
✓ No random four-way client partition
✓ Non-IID client distributions preserved
✓ 20 selected features used
✓ Client-specific 80/10/10 split
✓ Stratification used
✓ Scaler fitted only on each client's training data
✓ No SMOTE in final client pipeline
✓ NaN/Infinity checks completed
✓ Floating-point scaling tolerance applied
"""
)

print("=" * 70)
print("DAY 18 DATA PIPELINE + CLIENT PARTITION: COMPLETE")
print("=" * 70)