# Day 9 — CICIDS2017 Dataset Exploration Report

## 1. Objective

The objective of Day 9 was to inspect all CICIDS2017 CSV files before starting data preprocessing and machine learning.

The following were checked:

* Number of rows and columns
* Data types
* Labels and attack categories
* Duplicate columns
* Constant columns
* Missing values
* Infinite values
* Suspicious values
* Class distribution
* General dataset structure

---

## 2. Dataset Files

A total of **8 CSV files** were inspected.

| File                            |    Rows | Columns |
| ------------------------------- | ------: | ------: |
| Monday-WorkingHours             | 529,918 |      79 |
| Tuesday-WorkingHours            | 445,909 |      79 |
| Wednesday-workingHours          | 692,703 |      79 |
| Thursday-Morning-WebAttacks     | 170,366 |      79 |
| Thursday-Afternoon-Infiltration | 288,602 |      79 |
| Friday-Morning                  | 191,033 |      79 |
| Friday-Afternoon-PortScan       | 286,467 |      79 |
| Friday-Afternoon-DDoS           | 225,745 |      79 |

Every file contains **79 columns**.

The structure is:

* **78 network-flow features**
* **1 Label column**

This matches the expected CICIDS2017 structure used in this project.

---

# 3. Data Types

The files consistently contain:

* **54 integer columns**
* **24 floating-point columns**
* **1 object column**

The object column is the `Label` column.

Therefore, the dataset is mostly numerical and is suitable for machine-learning processing after preprocessing.

---

# 4. Labels and Attack Categories

Different files contain different types of network traffic.

### Monday

| Label  | Records |
| ------ | ------: |
| BENIGN | 529,918 |

Monday contains only benign traffic.

---

### Tuesday

| Label       | Records |
| ----------- | ------: |
| BENIGN      | 432,074 |
| FTP-Patator |   7,938 |
| SSH-Patator |   5,897 |

Tuesday contains benign traffic and brute-force attacks.

---

### Wednesday

| Label            | Records |
| ---------------- | ------: |
| BENIGN           | 440,031 |
| DoS Hulk         | 231,073 |
| DoS GoldenEye    |  10,293 |
| DoS slowloris    |   5,796 |
| DoS Slowhttptest |   5,499 |
| Heartbleed       |      11 |

Wednesday contains several DoS attack categories and Heartbleed.

---

### Thursday Morning — Web Attacks

| Label                      | Records |
| -------------------------- | ------: |
| BENIGN                     | 168,186 |
| Web Attack – Brute Force   |   1,507 |
| Web Attack – XSS           |     652 |
| Web Attack – SQL Injection |      21 |

The Web Attack labels appeared with encoding/display characters (`�`) during inspection. This should be handled carefully during later preprocessing.

---

### Thursday Afternoon — Infiltration

| Label        | Records |
| ------------ | ------: |
| BENIGN       | 288,566 |
| Infiltration |      36 |

This file shows an extremely imbalanced distribution because only 36 records are labeled as Infiltration.

---

### Friday Morning

| Label  | Records |
| ------ | ------: |
| BENIGN | 189,067 |
| Bot    |   1,966 |

Friday morning contains benign traffic and Bot attacks.

---

### Friday Afternoon — PortScan

| Label    | Records |
| -------- | ------: |
| PortScan | 158,930 |
| BENIGN   | 127,537 |

PortScan represents a large portion of this file.

---

### Friday Afternoon — DDoS

| Label  | Records |
| ------ | ------: |
| DDoS   | 128,027 |
| BENIGN |  97,718 |

DDoS also represents a large portion of this file.

---

# 5. Duplicate Columns

Duplicate-column inspection was performed on all 8 files.

### Result

**No duplicate columns were found.**

Therefore, there is no immediate need to remove columns because of duplicated column names.

---

# 6. Constant Columns

Several features were found to have only one unique value within individual files.

The commonly repeated constant columns include:

```text
Bwd PSH Flags
Fwd URG Flags
Bwd URG Flags
CWE Flag Count
Fwd Avg Bytes/Bulk
Fwd Avg Packets/Bulk
Fwd Avg Bulk Rate
Bwd Avg Bytes/Bulk
Bwd Avg Packets/Bulk
Bwd Avg Bulk Rate
```

These columns contain little or no useful variation in the inspected files.

They should be investigated further during preprocessing and may be removed if they remain constant in the final combined dataset.

The `Label` column was constant in the Monday file because Monday contains only BENIGN traffic. The label column must **not** be removed.

---

# 7. Missing Values

Missing values were found in several files.

Examples:

| File                              | Total Missing Values |
| --------------------------------- | -------------------: |
| Monday                            |                   64 |
| Thursday Afternoon — Infiltration |                   18 |
| Friday Morning                    |                   28 |
| Friday PortScan                   |                   15 |
| Friday DDoS                       |                    4 |

Other inspected files also require proper missing-value checking during preprocessing.

Missing values must be handled before model training.

---

# 8. Infinite Values

Infinite values were found during the inspection of numerical features.

In the Monday file, for example:

* `Flow Bytes/s` contained infinite values.
* `Flow Packets/s` contained infinite values.

Infinite values can cause problems during normalization and machine-learning model training.

Therefore, the preprocessing pipeline must explicitly detect and handle:

```text
NaN
+∞
-∞
```

---

# 9. Suspicious Values

Some numerical features contained negative values that require investigation.

Examples observed during inspection include:

```text
Flow Duration = -1
Flow IAT Mean = -4
Flow IAT Max = -4
Init_Win_bytes_forward = -1
Init_Win_bytes_backward = -1
```

Some features also had minimum values of `0`, such as:

```text
Fwd Header Length.1 = 0
Avg Bwd Segment Size = 0
ECE Flag Count = 0
```

These values should **not be automatically deleted** during Day 9.

They need to be investigated during preprocessing because some may represent special/default values in the original CICIDS2017 flow-generation process.

---

# 10. Class Imbalance

A major observation from the dataset inspection is **class imbalance**.

Some attack categories contain many records, while others contain very few.

The most obvious example is:

```text
Infiltration = 36 records
BENIGN = 288,566 records
```

Heartbleed is also extremely rare:

```text
Heartbleed = 11 records
```

Some other attack categories are much larger, such as:

```text
DoS Hulk = 231,073
PortScan = 158,930
DDoS = 128,027
```

This imbalance is important because a machine-learning model could become biased toward the majority class.

Later preprocessing and training must therefore consider appropriate techniques for class imbalance.

---

# 11. Important Dataset Observation — Non-IID Distribution

The attack categories are not evenly distributed across the eight files.

For example:

* Monday → only BENIGN
* Tuesday → FTP-Patator and SSH-Patator
* Wednesday → mainly DoS attacks
* Thursday Morning → Web Attacks
* Thursday Afternoon → Infiltration
* Friday Morning → Bot
* Friday Afternoon → PortScan
* Friday Afternoon → DDoS

This is very important for the proposed **Federated Learning** experiment.

Different clients can receive different portions of the dataset, creating a **non-IID data distribution**.

This will allow the project to investigate how the FDRL-IDS system performs when federated clients have different types of network traffic.

---

# 12. Important Findings

The Day 9 inspection produced the following findings:

1. All 8 CSV files contain **79 columns**.
2. Each file contains **78 numerical features and 1 label column**.
3. No duplicate columns were found.
4. Several features are constant within files.
5. Missing values exist.
6. Infinite values exist.
7. Some numerical features contain suspicious negative values such as `-1`.
8. The dataset contains significant class imbalance.
9. Some attack classes are extremely rare.
10. Different files contain different attack categories.
11. The label column must be preserved.
12. The dataset structure supports the planned federated/non-IID experiment.

---

# 13. Preprocessing Problems Identified

Based on this exploration, the future preprocessing pipeline must address:

```text
1. Missing values
2. Infinite values
3. Suspicious numerical values
4. Constant features
5. Label encoding
6. Label-name consistency
7. Class imbalance
8. Feature scaling
9. Possible feature selection/reduction
10. Non-IID client partitioning
```

These issues will be handled in the upcoming preprocessing stages rather than during this initial exploration.

---

# 14. Conclusion

The CICIDS2017 dataset has been successfully inspected across all **8 CSV files**.

The dataset contains a large number of network-flow records with **78 numerical features and one label**. The inspection confirmed that the data contains benign traffic as well as multiple attack categories, including brute force, DoS, Web Attacks, Infiltration, Bot, PortScan, and DDoS.

Several data-quality issues were identified, including missing values, infinite values, constant features, suspicious numerical values, and severe class imbalance.

The most important observation for the FDRL-IDS project is that attack types are distributed differently across the dataset files. This provides a useful foundation for creating **non-IID federated clients** in the later Federated Learning experiment.

Therefore, the dataset is now sufficiently understood to move to the next stage: **data preprocessing and preparation**.

## Day 9 Status

**✅ COMPLETE — All 8 CICIDS2017 CSV files inspected.**
