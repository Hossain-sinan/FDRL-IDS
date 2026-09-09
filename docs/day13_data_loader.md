# Day 13 — Build Data Loader

## Goal

The main goal of Day 13 was to create a Python data loader for the CICIDS2017 dataset.

The loader should automatically find the CSV files, load them, combine them into one DataFrame, and return the final dataset.

## Work Done

I created the following file:

```text
src/data_loader.py
```

The data loader works in these steps:

```text
Find dataset
      ↓
Find CSV files
      ↓
Load CSV files
      ↓
Combine the files
      ↓
Clean column names
      ↓
Return DataFrame
```

I also created:

```text
src/test_loader.py
```

This file was used to test whether the data loader was working correctly.

## Dataset Loading

The loader successfully found and loaded **8 CICIDS2017 CSV files**.

The files were:

1. Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
2. Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
3. Friday-WorkingHours-Morning.pcap_ISCX.csv
4. Monday-WorkingHours.pcap_ISCX.csv
5. Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
6. Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
7. Tuesday-WorkingHours.pcap_ISCX.csv
8. Wednesday-workingHours.pcap_ISCX.csv

Each file was loaded using Pandas.

## Final Dataset

After loading and combining all 8 files, the final dataset size was:

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

This matches the CICIDS2017 structure that I checked during the previous dataset study.

## Label Verification

After combining the files, I checked the `Label` column to make sure the attack labels were available.

The dataset contains labels such as:

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

The label counts were also successfully displayed using Pandas.

## Problem Found and Fixed

During the first test, I got an error:

```text
KeyError: 'Label'
```

The reason was that the original CSV files had extra spaces in some column names.

For example:

```text
' Label'
```

was used instead of:

```text
'Label'
```

Because of this, the following code did not work:

```python
df["Label"]
```

I fixed the problem by removing the extra spaces from all column names:

```python
combined_df.columns = combined_df.columns.str.strip()
```

After this change, I was able to access the `Label` column correctly.

## Final Testing

After fixing the column-name problem, I ran the test file again.

The final result was:

```text
Found 8 CSV file(s).

Combined dataset shape:
(2830743, 79)

Column count:
79

Label counts:
Successfully displayed
```

So, the data loader is working correctly.

## What I Learned

From Day 13, I learned:

1. How to find CSV files automatically using Python `Path`.
2. How to load CSV files using Pandas.
3. How to combine multiple DataFrames using `pd.concat()`.
4. How to remove extra spaces from column names.
5. How to create a reusable Python function.
6. How to create a separate test file for checking a Python module.
7. How to check the shape and labels of a dataset after loading.

## Day 13 Conclusion

Day 13 was successfully completed.

I created a reusable data loader for the FDRL-IDS project. It can automatically find the available CICIDS2017 CSV files, load them, combine them into one DataFrame, and return the dataset.

The final dataset loaded successfully with **2,830,743 rows and 79 columns**.

No data preprocessing, feature scaling, label encoding, or model training was done on Day 13. These tasks will be handled in the later stages of the project.
