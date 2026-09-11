import numpy as np
import pandas as pd


def clean_data(df):
	"""
	Clean the CICIDS2017 network traffic dataset.

	Handles:
	- Infinity values
	- NaN values
	- Duplicate rows
	- Invalid numeric values
	"""
	# Make a copy so the original DataFrame is not changed
	df = df.copy()

	print("========== DATA CLEANING ==========")

	# Before cleaning
	print("\nBefore cleaning")
	print(f"Rows: {len(df)}")
	print(f"Columns: {len(df.columns)}")
	original_rows = len(df)

	# Handle infinity values
	numeric_columns = df.select_dtypes(include=np.number).columns
	infinity_count = np.isinf(df[numeric_columns]).sum().sum()
	print(f"\nInfinity values found: {infinity_count}")

	# Convert infinity values to NaN
	df.replace([np.inf, -np.inf], np.nan, inplace=True)

	# Handle NaN values
	rows_before_nan = len(df)
	df.dropna(inplace=True)
	rows_removed_nan = rows_before_nan - len(df)
	print(f"Rows removed because of NaN/Infinity: {rows_removed_nan}")

	# Remove duplicate rows
	rows_before_duplicates = len(df)
	duplicate_count = df.duplicated().sum()
	df.drop_duplicates(inplace=True)
	rows_removed_duplicates = rows_before_duplicates - len(df)
	print(f"Duplicate rows found: {duplicate_count}")
	print(f"Duplicate rows removed: {rows_removed_duplicates}")

	# Check for invalid numeric values
	invalid_count = 0
	numeric_columns = df.select_dtypes(include=np.number).columns
	for column in numeric_columns:
		invalid_values = (~np.isfinite(df[column])).sum()
		invalid_count += invalid_values
	print(f"Invalid numeric values remaining: {invalid_count}")

	# After cleaning
	final_rows = len(df)
	print("\nAfter cleaning")
	print(f"Rows: {final_rows}")
	print(f"Columns: {len(df.columns)}")

	print("\n========== CLEANING SUMMARY ==========")
	print(f"Original rows: {original_rows}")
	print(f"Final rows: {final_rows}")
	print(f"Total rows removed: {original_rows - final_rows}")

	return df