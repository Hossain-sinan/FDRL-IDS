# Day 13 — Build Data Loader

## Goal

The goal of Day 13 was to create a reusable Python data loader that can automatically find the CICIDS2017 CSV files, load them, combine them into one DataFrame, and return the combined dataset.

## Work Done

I created:

```text
src/data_loader.py
```

The data loader performs these steps:

```text
Find dataset
      ↓
Find CSV files
      ↓
Load each CSV
      ↓
Combine all CSV files
      ↓
Clean column-name spaces
      ↓
Return DataFrame
```

I also created:

```text
src/test_loader.py
```

to test whether the data loader works correctly.

## Dataset Loading Result

The loader successfully found and loaded 8 CICIDS2017 CSV files.

The files loaded were:

1. Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
2. Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
3. Friday-WorkingHours-Morning.pcap_ISCX.csv
4. Monday-WorkingHours.pcap_ISCX.csv
5. Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
6. Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
7. Tuesday-WorkingHours.pcap_ISCX.csv
8. Wednesday-workingHours.pcap_ISCX.csv

## Final Dataset

After combining all 8 CSV files:

```text
Rows: 2,830,743
Columns: 79
```

The dataset contains:

```text
78 network traffic features
+
1 Label column
=
79 columns
```

## Label Verification

The loader successfully accessed the `Label` column and displayed the class distribution.

Main labels include:

* BENIGN
* DoS Hulk
* PortScan
* DDoS
* DoS GoldenEye
* FTP-Patator
* SSH-Patator
* DoS slowloris
* DoS Slowhttptest
* Bot
* Web Attack
* Infiltration
* Heartbleed

## Problem Found and Fixed

Initially, the CSV column names contained unnecessary spaces.

For example:

```text
' Label'
```

instead of:

```text
'Label'
```

Because of this, trying to access:

```python
df["Label"]
```

caused:

```text
KeyError: 'Label'
```

I fixed this by removing leading and trailing spaces from all column names:

```python
combined_df.columns = combined_df.columns.str.strip()
```

After the fix, the `Label` column could be accessed successfully.

## Final Verification

The final test successfully showed:

```text
Found 8 CSV file(s).

Combined dataset shape:
(2830743, 79)

Column count:
79

Label counts:
Successfully displayed
```

Therefore, the data loader is working correctly.

## What I Learned

From this task, I learned:

1. How to automatically find CSV files using `Path`.
2. How to load CSV files using Pandas.
3. How to combine multiple DataFrames using `pd.concat()`.
4. How to remove unnecessary spaces from column names.
5. How to create a reusable Python function.
6. How to test a Python data-loading module.

## Day 13 Conclusion

Day 13 is successfully completed.

The FDRL-IDS project now has a reusable data loader that can automatically load and combine the available CICIDS2017 CSV files into a single Pandas DataFrame.

No data preprocessing, feature scaling, label encoding, or model training was performed on Day 13