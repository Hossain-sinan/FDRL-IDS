
from pathlib import Path
import pandas as pd


def load_cicids2017():
    """
    Find and load all CICIDS2017 CSV files from data/raw
    and combine them into one DataFrame.
    """

    # Find project root directory
    project_root = Path(__file__).resolve().parent.parent

    # Dataset directory
    data_dir = project_root / "data" / "raw"

    # Find all CSV files
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {data_dir}"
        )

    print(f"Found {len(csv_files)} CSV file(s).")

    dataframes = []

    # Load each CSV file
    for csv_file in csv_files:
        print(f"Loading: {csv_file.name}")

        df = pd.read_csv(csv_file)

        print(f"  Shape: {df.shape}")

        dataframes.append(df)

    # Combine all CSV files
    combined_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    # Remove unnecessary spaces from column names
    combined_df.columns = combined_df.columns.str.strip()

    print(f"\nCombined dataset shape: {combined_df.shape}")

    return combined_df

