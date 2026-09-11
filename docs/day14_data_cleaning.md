# Day 14 — Data Cleaning

## Goal

The goal of Day 14 was to clean the CICIDS2017 dataset before using it for machine learning and the FDRL-IDS project.

The main problems checked were:

* NaN values
* Infinity values
* Duplicate rows
* Invalid numeric values

---

## Dataset Before Cleaning

The Day 13 data loader found 8 CICIDS2017 CSV files and combined them into one dataset.

The combined dataset contained:

* **Rows:** 2,830,743
* **Columns:** 79

So the original dataset size was:

```text
2,830,743 rows × 79 columns
```

---

## Data Quality Check

Before cleaning, I checked the dataset for common data-quality problems.

The results were:

| Problem               |    Result |
| --------------------- | --------: |
| Total NaN values      |     1,358 |
| Total Infinity values |     4,376 |
| Duplicate rows        |   308,381 |
| Total rows            | 2,830,743 |
| Total columns         |        79 |

The duplicate rows were the biggest data-quality issue found in the dataset.

---

## Cleaning Process

### 1. Handle Infinity Values

Positive and negative infinity values were converted into NaN values.

```python
df.replace([np.inf, -np.inf], np.nan, inplace=True)
```

This was done because infinity is not a suitable numeric value for the machine-learning pipeline.

---

### 2. Remove Rows Containing NaN or Infinity

After converting infinity values to NaN, rows containing NaN values were removed.

The number of rows removed at this stage was:

```text
2,867 rows
```

---

### 3. Remove Duplicate Rows

Duplicate rows were checked and removed.

After the NaN/Infinity cleaning stage, the dataset contained:

```text
307,078 duplicate rows
```

These duplicate rows were removed.

---

### 4. Check Invalid Numeric Values

After cleaning, all numeric columns were checked again for invalid non-finite values.

Result:

```text
Invalid numeric values remaining: 0
```

This confirms that no invalid numeric values remained after preprocessing.

---

## Final Result

The cleaning process changed the dataset as follows:

```text
Original dataset
2,830,743 rows
        ↓
Remove NaN/Infinity rows
2,867 rows removed
        ↓
Remove duplicate rows
307,078 rows removed
        ↓
Clean dataset
2,520,798 rows
```

Final dataset:

```text
2,520,798 rows × 79 columns
```

Total rows removed:

```text
309,945 rows
```

Approximately **10.95%** of the original rows were removed during cleaning.

---

## Cleaning Summary

| Stage                     |          Rows |
| ------------------------- | ------------: |
| Original dataset          |     2,830,743 |
| Removed NaN/Infinity      |         2,867 |
| Removed duplicates        |       307,078 |
| **Final cleaned dataset** | **2,520,798** |

---

## Files Created

The following files were created for Day 14:

```text
src/preprocessing.py
src/test_preprocessing.py
docs/day14_data_cleaning.md
```

---

## Conclusion

Day 14 successfully completed the basic data cleaning stage of the FDRL-IDS project.

The CICIDS2017 dataset was checked for missing values, infinity values, duplicate rows, and invalid numeric values.

After cleaning, the final dataset contains **2,520,798 rows and 79 columns**, with **0 invalid numeric values remaining**.

This cleaned dataset can now be used for the next preprocessing stages of the FDRL-IDS project.
