
"""
Day 35 - DQN Validation
Step 2: Evaluate the model on separate traffic samples.

This is an initial holdout test using the same CICIDS2017 CSV file.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:                   
    
    sys.path.insert(0, str(PROJECT_ROOT))

from src.day32_dqn_network import DQNNetwork


# ---------------------------------------------------------
# 1. Paths and settings
# ---------------------------------------------------------

MODEL_DIR = PROJECT_ROOT / "models"

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

CHECKPOINT_PATH = MODEL_DIR / "day34_improved_dqn_checkpoint.pt"
SCALER_PATH = MODEL_DIR / "day34_improved_scaler.joblib"
METADATA_PATH = MODEL_DIR / "day34_improved_metadata.json"

SAMPLES_PER_CLASS = 500


# ---------------------------------------------------------
# 2. Load saved model settings
# ---------------------------------------------------------

print("=" * 65)
print("DAY 35 - DQN VALIDATION")
print("=" * 65)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

selected_features = metadata["selected_features"]
state_size = metadata["feature_count"]
action_size = metadata["action_size"]

scaler = joblib.load(SCALER_PATH)

model = DQNNetwork(
    input_size=state_size,
    output_size=action_size,
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(
    checkpoint["online_network_state_dict"]
)
model.eval()

print("Saved DQN model loaded.")
print("Saved scaler loaded.")
print("Selected features:", len(selected_features))


# ---------------------------------------------------------
# 3. Load the traffic dataset
# ---------------------------------------------------------

print()
print("Loading CICIDS2017 traffic data...")

df = pd.read_csv(RAW_FILE)
df.columns = df.columns.str.strip()

required_columns = selected_features + ["Label"]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df[required_columns].copy()

print("Total rows loaded:", len(df))


# ---------------------------------------------------------
# 4. Recreate the Day 34 training selection
# ---------------------------------------------------------

# Day 34 selected these rows using random_state=42.
# Exclude those rows from the validation candidates.

benign_train = df[df["Label"] == "BENIGN"].sample(
    n=SAMPLES_PER_CLASS,
    random_state=42,
)

attack_train = df[df["Label"] == "DDoS"].sample(
    n=SAMPLES_PER_CLASS,
    random_state=42,
)

training_indices = set(benign_train.index) | set(attack_train.index)

remaining_df = df.drop(index=list(training_indices))

print("Day 34 training rows excluded:", len(training_indices))


# ---------------------------------------------------------
# 5. Select separate validation samples
# ---------------------------------------------------------

benign_remaining = remaining_df[
    remaining_df["Label"] == "BENIGN"
]

attack_remaining = remaining_df[
    remaining_df["Label"] == "DDoS"
]

if (
    len(benign_remaining) < SAMPLES_PER_CLASS
    or len(attack_remaining) < SAMPLES_PER_CLASS
):
    raise ValueError(
        "Not enough remaining samples for validation."
    )

benign_val = benign_remaining.sample(
    n=SAMPLES_PER_CLASS,
    random_state=123,
)

attack_val = attack_remaining.sample(
    n=SAMPLES_PER_CLASS,
    random_state=123,
)

validation_df = pd.concat(
    [benign_val, attack_val],
).sample(
    frac=1,
    random_state=123,
).reset_index(drop=True)

print("Validation samples:", len(validation_df))
print("BENIGN samples:", (validation_df["Label"] == "BENIGN").sum())
print("DDoS samples:", (validation_df["Label"] == "DDoS").sum())


# ---------------------------------------------------------
# 6. Prepare validation features
# ---------------------------------------------------------

X_val = validation_df[selected_features].copy()

X_val = X_val.apply(
    pd.to_numeric,
    errors="coerce",
)

X_val = X_val.replace(
    [np.inf, -np.inf],
    np.nan,
)

X_val = X_val.fillna(0)

y_true = (
    validation_df["Label"] != "BENIGN"
).astype(int).to_numpy()

# Important: use the saved Day 34 scaler.
# Do not fit a new scaler on validation data.
X_val_scaled = scaler.transform(X_val)

X_val_tensor = torch.tensor(
    X_val_scaled,
    dtype=torch.float32,
)

print()
print("Validation feature shape:", X_val_scaled.shape)
print("Validation labels shape:", y_true.shape)


# ---------------------------------------------------------
# 7. Run the DQN model
# ---------------------------------------------------------

print()
print("Running DQN predictions...")

with torch.no_grad():
    q_values = model(X_val_tensor)
    selected_actions = torch.argmax(
        q_values,
        dim=1,
    ).numpy()

# Binary detection rule matching the current environment:
# action 0 = BENIGN
# actions 1, 2, 3 = ATTACK detected
y_pred = (selected_actions != 0).astype(int)


# ---------------------------------------------------------
# 8. Calculate evaluation metrics
# ---------------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0,
)

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
)


# ---------------------------------------------------------
# 9. Display results
# ---------------------------------------------------------

print()
print("=" * 65)
print("VALIDATION RESULTS")
print("=" * 65)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print()
print("Confusion matrix:")
print("Rows = actual labels; columns = predicted labels")
print("Label order: 0 = BENIGN, 1 = ATTACK")
print(matrix)

print()
print("Classification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0,
    )
)

print()
print("Selected DQN action counts:")
action_names = metadata.get("actions", {
    "0": "BENIGN",
    "1": "ATTACK",
    "2": "BLOCK",
    "3": "MONITOR",
})
for action_id, action_name in action_names.items():
    count = int((selected_actions == int(action_id)).sum())
    print(f"{action_id} ({action_name}): {count}")

print()
print("STEP 2 COMPLETE")
print("Initial holdout validation finished.")
# ---------------------------------------------------------
# 10. Save validation results
# ---------------------------------------------------------

RESULTS_FILE = PROJECT_ROOT / "docs" / "day35_improved_validation_results.json"

results = {
    "model": "Improved DQN",
    "validation_samples": int(len(y_true)),
    "benign_samples": int((y_true == 0).sum()),
    "attack_samples": int((y_true == 1).sum()),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": matrix.tolist(),
   "action_counts": {
    action_names[action_id]: int(
        (selected_actions == int(action_id)).sum()
    )
    for action_id in action_names
},
}

with open(RESULTS_FILE, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print()
print("Validation results saved:", RESULTS_FILE)