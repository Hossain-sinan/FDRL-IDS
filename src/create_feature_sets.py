from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Resolve project paths reliably
# --------------------------------------------------

project_root = Path(__file__).resolve().parents[1]
docs_dir = project_root / "docs"
legacy_docs_dir = project_root.parent / "docs"

docs_dir.mkdir(exist_ok=True)


# --------------------------------------------------
# Load the 63-feature pool
# --------------------------------------------------

candidate_file = docs_dir / "day16_candidate_features_63.csv"
legacy_candidate_file = legacy_docs_dir / "day16_candidate_features_63.csv"
importance_file = docs_dir / "day16_feature_importance.csv"
legacy_importance_file = legacy_docs_dir / "day16_feature_importance.csv"

redundant_features = {
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Bytes",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
}

if candidate_file.exists():
    source_path = candidate_file
elif legacy_candidate_file.exists():
    source_path = legacy_candidate_file
elif importance_file.exists():
    source_path = importance_file
elif legacy_importance_file.exists():
    source_path = legacy_importance_file
else:
    raise FileNotFoundError(
        "No feature source file was found. Run the feature selection step first, "
        "or make sure one of these files exists: "
        f"{candidate_file}, {legacy_candidate_file}, {importance_file}, {legacy_importance_file}"
    )

features_df = pd.read_csv(source_path)

if "feature" in features_df.columns:
    features = features_df["feature"].tolist()
else:
    features = features_df.iloc[:, 0].tolist()

# If a feature-importance file is used, remove the obvious redundant features
# to recover the same 63-feature pool used in the analysis.
if source_path.name in {"day16_feature_importance.csv"}:
    features = [feature for feature in features if feature not in redundant_features]

    candidate_file.write_text("")
    pd.DataFrame({"feature": features}).to_csv(
        candidate_file,
        index=False
    )
    print(f"Built 63-feature pool at: {candidate_file}")

print(f"Total available features: {len(features)}")


# --------------------------------------------------
# Create candidate feature sets
# --------------------------------------------------

candidate_sizes = [63, 50, 40, 30, 20]

for size in candidate_sizes:
    selected_features = features[:size]
    output_file = docs_dir / f"day16_features_{size}.csv"

    output_df = pd.DataFrame({
        "feature": selected_features
    })

    output_df.to_csv(output_file, index=False)
    print(f"Saved {size}-feature set: {output_file}")


print("\nFeature-set creation completed.")