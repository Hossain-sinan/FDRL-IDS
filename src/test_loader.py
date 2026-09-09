
from data_loader import load_cicids2017


df = load_cicids2017()

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn count:")
print(len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nLabel counts:")
print(df["Label"].value_counts())

