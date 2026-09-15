# Day 18 — Complete Data Pipeline and Non-IID Client Partition

## 1. Introduction

Today I completed the main data preparation pipeline for my FDRL-IDS project.

In the previous days, I already completed:

* CICIDS2017 dataset loading
* Data cleaning
* Attack label processing
* Feature selection
* Train/validation/test splitting
* Feature scaling
* SMOTE experiment

The main work of Day 18 was to prepare the dataset for the federated learning part of the project.

For this, I divided the CICIDS2017 dataset into four simulated organizations:

* Hospital
* Bank
* University
* ISP

I did not randomly divide the complete dataset into four parts. I used the original CICIDS2017 working-day/source structure so that each client has a different data distribution. This is important because my project needs a **non-IID federated learning environment**.

---

# 2. Objectives of Day 18

The main objectives were:

1. Complete the overall data pipeline.
2. Use the selected 20 features.
3. Process the original CICIDS2017 CSV files.
4. Create four simulated federated clients.
5. Preserve the original source/day information.
6. Keep the client data non-IID.
7. Split each client into training, validation, and test sets.
8. Apply scaling without data leakage.
9. Check the final data quality.
10. Prepare the data for future federated learning experiments.

---

# 3. Dataset

The project uses the CICIDS2017 dataset.

There are 8 CSV files in my project:

```text
Monday-WorkingHours.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
```

The original dataset contains 78 traffic features and one label column.

After loading the data, I removed extra spaces from the column names so that the label column could be used as:

```text
Label
```

---

# 4. Why I Selected 20 Features

The original CICIDS2017 dataset contains 78 features.

I decided not to directly use all 78 features in the initial FDRL-IDS model.

The reason is that using all features can increase the input dimension and may also include unnecessary or highly redundant information.

During Day 16, I first checked the features and removed 8 constant features. These features did not provide useful variation.

After removing them, I had 70 candidate features.

Then I performed correlation analysis using:

```text
Absolute correlation >= 0.90
```

This helped me identify highly correlated features.

For example:

```text
Bwd Packet Length Mean
        ↕
Avg Bwd Segment Size
```

and:

```text
Total Fwd Packets
        ↕
Subflow Fwd Packets
```

These features can contain very similar information.

So, keeping every highly correlated feature may increase the input size without giving the same amount of new information.

After this analysis, I selected 20 features as a practical feature subset for the initial FDRL-IDS experiments.

The selected features are:

```text
1.  min_seg_size_forward
2.  PSH Flag Count
3.  Init_Win_bytes_forward
4.  Init_Win_bytes_backward
5.  Flow Duration
6.  ACK Flag Count
7.  Destination Port
8.  Bwd Packet Length Min
9.  Bwd Packet Length Mean
10. Fwd IAT Total
11. Fwd IAT Std
12. Bwd IAT Total
13. Flow IAT Max
14. Average Packet Size
15. Packet Length Mean
16. Fwd IAT Max
17. Bwd Header Length
18. Packet Length Std
19. Min Packet Length
20. URG Flag Count
```

These features cover different types of network traffic information, such as:

* Flow information
* Packet length
* Forward and backward traffic
* Inter-arrival time
* TCP flags
* Window size
* Header information
* Destination port

### Important limitation

I am not claiming that these 20 features are the globally best 20 features for CICIDS2017.

The number 20 is a practical experimental choice to reduce dimensionality and redundancy while keeping the model input manageable.

The real effectiveness of these features will be checked later through model performance and comparison experiments.

The selected feature list is stored in:

```text
docs/day16_features_20.csv
```

---

# 5. Client Design

The main part of Day 18 was creating the four federated clients.

I used the following mapping:

| Client     | Source Data      |
| ---------- | ---------------- |
| Hospital   | Monday + Tuesday |
| Bank       | Wednesday        |
| University | Thursday         |
| ISP        | Friday           |

More specifically:

```text
Hospital
    = Monday + Tuesday

Bank
    = Wednesday

University
    = Thursday Infiltration + Thursday WebAttacks

ISP
    = Friday DDoS + Friday PortScan + Friday Morning
```

---

# 6. Why I Did Not Randomly Divide the Dataset

I did not use random partitioning for the four clients.

If I randomly divided the complete dataset into four parts, each client would probably contain a similar mixture of attack classes.

That would make the clients more IID-like.

But in a real federated learning system, different organizations can have very different types of network traffic.

For example:

* A hospital may have mostly normal traffic and some brute-force traffic.
* A bank may have more DoS-related traffic.
* A university may have web attacks and rare infiltration traffic.
* An ISP may have DDoS and port scanning traffic.

Therefore, I kept the original CICIDS2017 day/source structure to create different client distributions.

This makes the federated learning environment more suitable for testing non-IID data.

---

# 7. Processing Method

I processed each original CSV file separately.

The process was:

```text
Original CSV
     ↓
Load CSV
     ↓
Remove spaces from column names
     ↓
Check Label column
     ↓
Clean data
     ↓
Process attack labels
     ↓
Select 20 features
     ↓
Add source file information
     ↓
Add client information
     ↓
Combine files belonging to same client
     ↓
Train / Validation / Test split
     ↓
Client-specific scaling
```

I chose this method because it keeps the original source information clear.

It also makes it easier to find which source file has a particular problem.

---

# 8. Data Cleaning Results

The source files were cleaned individually.

## Monday

```text
Original rows:             529,918
Infinity values:               810
Rows removed NaN/Infinity:     437
Duplicate rows:             26,831
Final rows:                502,650
```

## Tuesday

```text
Original rows:             445,909
Infinity values:               327
Rows removed NaN/Infinity:     264
Duplicate rows:             24,019
Final rows:                421,626
```

## Wednesday

```text
Original rows:             692,703
Infinity values:             1,586
Rows removed NaN/Infinity:   1,297
Duplicate rows:             80,914
Final rows:                610,492
```

After label processing, 11 Heartbleed records were removed because Heartbleed is not included in my current class mapping.

Final Wednesday records:

```text
610,481
```

## Thursday — Infiltration

```text
Original rows:             288,602
Infinity values:               396
Rows removed NaN/Infinity:     207
Duplicate rows:             35,605
Final rows:                252,790
```

## Thursday — Web Attacks

```text
Original rows:             170,366
Infinity values:               250
Rows removed NaN/Infinity:     135
Duplicate rows:              6,052
Final rows:                164,179
```

## Friday — DDoS

```text
Original rows:             225,745
Infinity values:                64
Rows removed NaN/Infinity:      34
Duplicate rows:              2,629
Final rows:                223,082
```

## Friday — PortScan

```text
Original rows:             286,467
Infinity values:               727
Rows removed NaN/Infinity:     371
Duplicate rows:             72,319
Final rows:                213,777
```

## Friday — Morning

```text
Original rows:             191,033
Infinity values:               216
Rows removed NaN/Infinity:     122
Duplicate rows:              6,867
Final rows:                184,044
```

---

# 9. Final Client Sizes

After cleaning and label processing, the final client sizes are:

| Client     |       Records |
| ---------- | ------------: |
| Hospital   |       924,276 |
| Bank       |       610,481 |
| University |       416,969 |
| ISP        |       620,903 |
| **Total**  | **2,572,629** |

The four clients are clearly different in size.

This is acceptable because real organizations will not necessarily have the same amount of network traffic.

---

# 10. Label Classes

I am using the following 8 classes:

| ID | Class        |
| -: | ------------ |
|  0 | BENIGN       |
|  1 | DoS          |
|  2 | DDoS         |
|  3 | Brute Force  |
|  4 | Web Attack   |
|  5 | Infiltration |
|  6 | Botnet       |
|  7 | Port Scan    |

Heartbleed was not included in the current mapping.

There were only 11 Heartbleed records, and they were removed during label processing.

---

# 11. Client Class Distribution

## Hospital

Total records:

```text
924,276
```

| Class       | Records | Percentage |
| ----------- | ------: | ---------: |
| BENIGN      | 915,126 |   99.0100% |
| Brute Force |   9,150 |    0.9900% |

The Hospital client is highly dominated by BENIGN traffic.

---

## Bank

Total records:

```text
610,481
```

| Class  | Records | Percentage |
| ------ | ------: | ---------: |
| BENIGN | 416,736 |   68.2635% |
| DoS    | 193,745 |   31.7365% |

The Bank client has a much higher percentage of DoS traffic than the Hospital client.

---

## University

Total records:

```text
416,969
```

| Class        | Records | Percentage |
| ------------ | ------: | ---------: |
| BENIGN       | 414,790 |   99.4774% |
| Web Attack   |   2,143 |    0.5139% |
| Infiltration |      36 |    0.0086% |

The University client has an extremely small number of Infiltration samples.

---

## ISP

Total records:

```text
620,903
```

| Class     | Records | Percentage |
| --------- | ------: | ---------: |
| BENIGN    | 400,247 |   64.4621% |
| DDoS      | 128,014 |   20.6174% |
| Botnet    |   1,948 |    0.3137% |
| Port Scan |  90,694 |   14.6068% |

The ISP client contains several different attack categories.

---

# 12. Why the Client Data Is Non-IID

The distributions are clearly different between clients.

For example:

```text
Hospital:
BENIGN + Brute Force

Bank:
BENIGN + DoS

University:
BENIGN + Web Attack + Infiltration

ISP:
BENIGN + DDoS + Botnet + Port Scan
```

So the four clients do not have the same data distribution.

This creates the non-IID condition required for the federated learning experiments.

This is important because later I need to study how federated learning behaves when each client has different local data.

---

# 13. Train, Validation and Test Split

Each client was divided separately using:

```text
80% Training
10% Validation
10% Test
```

I used:

```text
random_state = 42
```

and stratified splitting.

The purpose of stratification is to keep the class distribution as balanced as possible between the training, validation, and test sets.

---

# 14. Final Client Split

| Client     |      Training |  Validation |        Test |
| ---------- | ------------: | ----------: | ----------: |
| Hospital   |       739,420 |      92,428 |      92,428 |
| Bank       |       488,384 |      61,048 |      61,049 |
| University |       333,575 |      41,697 |      41,697 |
| ISP        |       496,722 |      62,090 |      62,091 |
| **Total**  | **2,058,101** | **257,263** | **257,265** |

All client split counts were verified successfully.

---

# 15. Rare Infiltration Class Problem

One important problem was the very small number of Infiltration samples in the University client.

There are only:

```text
36 Infiltration records
```

After the stratified split, the training set contains:

```text
29 Infiltration records
```

This is a difficult situation for machine learning.

However, I did not artificially change the dataset just to make the numbers look balanced.

I kept the original distribution because this is also an important part of the research problem.

Later, I need to carefully check:

* Precision
* Recall
* F1-score
* Macro F1
* Confusion matrix
* Per-class performance

Accuracy alone will not be enough for this project.

---

# 16. Feature Scaling

I used Min-Max Scaling for each client.

The scaler was fitted only on the training data.

The process was:

```text
Client Training Data
        ↓
Fit MinMaxScaler
        ↓
Transform Training Data
        ↓
Transform Validation Data
        ↓
Transform Test Data
```

I did not use validation or test data to fit the scaler.

This is important because using test data during preprocessing can cause data leakage.

---

# 17. Client-Specific Scaling

Each client has its own scaler:

```text
Hospital    → Hospital scaler
Bank        → Bank scaler
University  → University scaler
ISP         → ISP scaler
```

The scaler is fitted using only that client's training data.

This keeps the preprocessing local to each client.

---

# 18. Scaling Results

The expected Min-Max range is approximately:

```text
0 to 1
```

The results were:

| Client     | Minimum |            Maximum | Result |
| ---------- | ------: | -----------------: | ------ |
| Hospital   |     0.0 | 1.0000000000000002 | PASS   |
| Bank       |     0.0 |                1.0 | PASS   |
| University |     0.0 |                1.0 | PASS   |
| ISP        |     0.0 |                1.0 | PASS   |

The Hospital value:

```text
1.0000000000000002
```

is only a small floating-point precision difference from 1.0.

Therefore, I used:

```python
SCALING_TOLERANCE = 1e-10
```

instead of checking the value with a strict mathematical comparison.

---

# 19. SMOTE Decision

I performed a SMOTE experiment on Day 17.

However, I did not include SMOTE in the final Day 18 client pipeline.

The reason is that Day 18 is mainly focused on creating the natural non-IID client environment.

Applying SMOTE directly to all clients could change the original class distributions.

Therefore:

```text
Day 17:
SMOTE = Experimental

Day 18:
SMOTE = Not used in final client pipeline
```

If necessary, SMOTE can be studied later as a separate experiment.

---

# 20. Problems Faced and Solutions

## Problem 1 — Floating-Point Scaling Error

In the first complete pipeline, scaling produced:

```text
1.0000000000000002
```

The first validation considered anything greater than 1 as an error.

### Solution

I added a small tolerance:

```python
SCALING_TOLERANCE = 1e-10
```

Now very small floating-point differences are accepted.

---

## Problem 2 — `KeyError: 'Label'`

In the first client partition script, I got:

```text
KeyError: 'Label'
```

The reason was that the original CSV contained:

```text
' Label'
```

with an extra space.

The reusable data loader already removed this space, but the client script was reading the CSV directly.

### Solution

I added:

```python
df.columns = df.columns.str.strip()
```

After that, the column became:

```text
Label
```

and the label processing worked correctly.

---

## Problem 3 — Client Processing Method

Initially, I considered combining the files for each client first and then processing them.

After checking the research design, I changed the approach.

Now each original CSV is processed separately first.

This is better because:

* Source identity is preserved.
* Cleaning can be checked file by file.
* Problems can be identified more easily.
* The original CICIDS2017 structure is easier to explain.
* Client creation remains transparent.

---

## Problem 4 — Very Rare Infiltration

The University client has only 36 Infiltration samples.

This creates a serious class imbalance problem.

The stratified split still completed successfully, so I did not change the data artificially.

This issue will be considered during later model evaluation.

---

# 21. Final Training Distribution

After splitting, the training distributions were:

### Hospital

```text
Class 0: 732,100 (99.0100%)
Class 3:   7,320 (0.9900%)
```

### Bank

```text
Class 0: 333,388 (68.2635%)
Class 1: 154,996 (31.7365%)
```

### University

```text
Class 0: 331,832 (99.4775%)
Class 4:   1,714 (0.5138%)
Class 5:      29 (0.0087%)
```

### ISP

```text
Class 0: 320,197 (64.4620%)
Class 2: 102,411 (20.6174%)
Class 6:   1,559 (0.3139%)
Class 7:  72,555 (14.6068%)
```

These results confirm that the non-IID property is still present after splitting.

---

# 22. Final Data Pipeline

The completed Day 18 pipeline can be summarized as:

```text
                CICIDS2017
                     |
                8 CSV Files
                     |
          Source-wise Processing
                     |
          ---------------------
          |         |         |
       Cleaning   Labels   Features
          |         |         |
          --------20 Features-
                     |
              Client Mapping
                     |
       -----------------------------
       |          |         |       |
    Hospital     Bank   University  ISP
       |          |         |       |
       -----------------------------
                     |
        Client-specific 80/10/10 Split
                     |
           Client-specific Scaling
                     |
          Train / Validation / Test
                     |
             Federated Learning
```

---

# 23. Files Used in Day 18

Main pipeline file:

```text
src/day18_complete_pipeline.py
```

Client partition file:

```text
src/day18_client_partition.py
```

Selected feature file:

```text
docs/day16_features_20.csv
```

Current report:

```text
docs/day18_complete_data_pipeline.md
```

---

# 24. Final Verification

The following checks were completed successfully:

```text
✓ All 8 source files found
✓ Source file mapping verified
✓ 20 selected features loaded
✓ Source-wise cleaning completed
✓ Labels processed
✓ Heartbleed handled consistently
✓ Four clients created
✓ Original source/day identity preserved
✓ Non-IID distributions preserved
✓ Client-specific train/validation/test split completed
✓ Stratified splitting completed
✓ Training-only scaling completed
✓ NaN checks passed
✓ Infinity checks passed
✓ Scaling checks passed
✓ Final client distributions verified
✓ SMOTE not used in final client pipeline
```

---

# 25. Final Result

The final Day 18 client dataset contains:

```text
Total records: 2,572,629

Training:      2,058,101
Validation:      257,263
Test:            257,265
```

The four clients are:

```text
Hospital
Bank
University
ISP
```

with different data distributions.

The final client mapping is:

```text
Hospital   = Monday + Tuesday
Bank       = Wednesday
University = Thursday
ISP        = Friday
```

---

# 26. Research Importance

Day 18 is an important step in my FDRL-IDS project because the project is now ready to move from general data preprocessing toward federated learning.

Before Day 18, I mainly worked with the complete dataset.

After Day 18, the data is organized into four different local organizations.

This allows me to study:

* Federated learning with non-IID data
* Client-level model training
* Global model aggregation
* Convergence behavior
* Effect of different client distributions
* Rare attack detection
* FedProx and other approaches for non-IID data
* Comparison between federated and centralized learning

So, Day 18 creates the data foundation for the main federated learning experiments.

---

# 27. Conclusion

Today I completed the complete data preparation and non-IID client partitioning stage.

I processed the eight CICIDS2017 source files, cleaned the data, processed the labels, selected the 20 features, and created four simulated organizations.

I used the original CICIDS2017 day/source structure instead of randomly dividing the dataset. Because of this, the four clients have clearly different attack distributions.

I also created separate training, validation, and test sets for every client and applied Min-Max scaling using only the training data.

During this work, I faced several problems, including a floating-point scaling issue and a `Label` column whitespace problem. I fixed both issues and improved the processing methodology.

The final pipeline passed all data quality, splitting, and scaling checks.

The dataset is now ready for the next stage of the project, where the actual federated learning and Deep Q-Network components will be developed.

**Day 18 Status: COMPLETED SUCCESSFULLY.**
