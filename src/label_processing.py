import pandas as pd


# Map original CICIDS2017 labels
# into the project's broader attack categories.
LABEL_MAPPING = {
    "BENIGN": "BENIGN",

    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",

    "DDoS": "DDoS",

    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",

    "Web Attack � Brute Force": "Web Attack",
    "Web Attack � XSS": "Web Attack",
    "Web Attack � Sql Injection": "Web Attack",

    "Infiltration": "Infiltration",

    "Bot": "Botnet",

    "PortScan": "Port Scan"
}


# Numeric IDs for the 8 project classes.
CLASS_TO_ID = {
    "BENIGN": 0,
    "DoS": 1,
    "DDoS": 2,
    "Brute Force": 3,
    "Web Attack": 4,
    "Infiltration": 5,
    "Botnet": 6,
    "Port Scan": 7
}


def process_labels(df, label_column="Label"):
    """
    Process CICIDS2017 labels into the project's
    8-class attack representation.
    """

    df = df.copy()

    # Convert original labels into project categories
    df["attack_category"] = df[label_column].map(LABEL_MAPPING)

    # Check labels that are not included in the mapping
    unknown_labels = df.loc[
        df["attack_category"].isna(),
        label_column
    ].value_counts()

    if not unknown_labels.empty:
        print("\n========== UNMAPPED LABELS ==========")
        print(unknown_labels)

    # Remove labels that are not included
    # in the project's 8-class strategy.
    df = df.dropna(subset=["attack_category"])

    # Convert categories into numeric IDs
    df["label_id"] = df["attack_category"].map(CLASS_TO_ID)

    # Binary label:
    # 0 = BENIGN
    # 1 = ATTACK
    df["binary_label"] = (
        df["attack_category"] != "BENIGN"
    ).astype(int)

    return df