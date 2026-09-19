# Day 22 — Confusion Matrix

## 1. Objective

The main objective of Day 22 was to generate and understand the **confusion matrix** for the saved centralized XGBoost baseline model.

The confusion matrix was used to:

* See correct and incorrect predictions for each attack class.
* Understand True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN).
* Visualize which classes are being confused with other classes.
* Generate a clear confusion matrix graph for later project analysis.

The XGBoost model was **not retrained** on Day 22.

---

## 2. Work Completed

The following tasks were completed:

1. Loaded the saved Day 20 XGBoost model.
2. Loaded and cleaned the CICIDS2017 dataset using the existing project pipeline.
3. Applied the existing label-processing method.
4. Loaded the 20 selected features from Day 16.
5. Recreated the same stratified test split used for the XGBoost baseline.
6. Generated predictions using the saved model.
7. Generated an 8-class confusion matrix.
8. Calculated TP, FP, TN, and FN for every class.
9. Saved the confusion matrix as a CSV file.
10. Saved TP/FP/FN/TN values as a CSV file.
11. Generated a readable heatmap visualization.

---

## 3. Dataset and Test Set

The same CICIDS2017 processing pipeline was reused to maintain consistency with the previous evaluation.

After cleaning and label processing:

* Cleaned dataset: **2,520,798 rows**
* Dataset after removing the 11 unmapped Heartbleed records: **2,520,787 rows**
* Selected features: **20**
* Test samples: **252,079**

The same `random_state=42` and stratified split were used.

Therefore, the confusion matrix is based on the **same test set used for the previous XGBoost evaluation**.

---

## 4. Confusion Matrix

The generated confusion matrix was:

```text
[[209352      8      5     26      4      0     13     98]
 [  2786  16582      0      0      6      0      0      0]
 [   359     23  12419      0      0      0      0      0]
 [    34      0      0    881      0      0      0      0]
 [   215      0      0      0      0      0      0      0]
 [     4      0      0      0      0      0      0      0]
 [   133      0      0      0      0      0      62      0]
 [   155     14      0      0      0      0      0   8900]]
```

The row represents the **true class**, while the column represents the **predicted class**.

The diagonal values represent correct predictions.

---

## 5. Class-Wise Interpretation

### BENIGN

Correctly predicted:

**209,352**

Incorrectly predicted as:

* DoS: 8
* DDoS: 5
* Brute Force: 26
* Web Attack: 4
* Botnet: 13
* Port Scan: 98

This means most BENIGN traffic was correctly identified.

---

### DoS

Correctly predicted:

**16,582**

Incorrectly predicted as:

**2,786 → BENIGN**

A smaller number was predicted as Web Attack:

**6**

The major error for DoS is therefore confusing some DoS traffic with BENIGN traffic.

---

### DDoS

Correctly predicted:

**12,419**

Incorrectly predicted:

* BENIGN: 359
* DoS: 23

The model detected most DDoS samples correctly, but some DDoS traffic was classified as BENIGN.

---

### Brute Force

Correctly predicted:

**881**

Incorrectly predicted:

**34 → BENIGN**

This indicates that the model detected most Brute Force samples in the test set.

---

### Web Attack

Correctly predicted:

**0**

All **215 Web Attack samples** were predicted as BENIGN.

This is an important weakness observed from the confusion matrix.

---

### Infiltration

Correctly predicted:

**0**

All **4 Infiltration samples** were predicted as BENIGN.

The number of test samples is extremely small, but the class was not correctly detected in this evaluation.

---

### Botnet

Correctly predicted:

**62**

Incorrectly predicted:

**133 → BENIGN**

Therefore, a significant portion of Botnet samples were classified as BENIGN.

---

### Port Scan

Correctly predicted:

**8,900**

Incorrectly predicted:

* BENIGN: 155
* DoS: 14

Most Port Scan samples were correctly identified.

---

## 6. TP, FP, TN and FN

The calculated one-vs-rest values are:

| Class        |      TP |    FP |      TN |    FN |
| ------------ | ------: | ----: | ------: | ----: |
| BENIGN       | 209,352 | 3,686 |  38,887 |   154 |
| DoS          |  16,582 |    45 | 232,660 | 2,792 |
| DDoS         |  12,419 |     5 | 239,273 |   382 |
| Brute Force  |     881 |    26 | 251,138 |    34 |
| Web Attack   |       0 |    10 | 251,854 |   215 |
| Infiltration |       0 |     0 | 252,075 |     4 |
| Botnet       |      62 |    13 | 251,871 |   133 |
| Port Scan    |   8,900 |    98 | 242,912 |   169 |

### Meaning

**TP — True Positive**

The sample belongs to a class and the model correctly predicts that class.

Example:

Actual DoS → Predicted DoS.

**TN — True Negative**

The sample does not belong to the class and the model correctly does not predict that class.

**FP — False Positive**

The sample does not belong to the class, but the model predicts that class.

**FN — False Negative**

The sample belongs to the class, but the model predicts another class.

---

## 7. Important Observation

The confusion matrix makes the class imbalance problem much easier to understand.

Large classes such as BENIGN have many samples, while some attack classes have very few samples.

The most important observations are:

* BENIGN has a very large number of correct predictions.
* DoS and DDoS are mostly detected correctly.
* Brute Force and Port Scan also have strong correct prediction counts.
* Web Attack has **0 correct predictions** in this test set.
* Infiltration has **0 correct predictions**, but only 4 test samples were available.
* Botnet has only **62 correct predictions out of 195** test samples.
* Several minority attack samples are predicted as BENIGN.

This shows why overall accuracy alone is not enough for evaluating an intrusion detection system.

---

## 8. Confusion Matrix Visualization

A heatmap was generated using a blue color scale.

The visualization makes the high-frequency and low-frequency prediction cells easier to distinguish.

The diagonal represents correct classification, while the off-diagonal cells represent classification errors.

The graph was saved at:

```text
docs/figures/day22_confusion_matrix.png
```

---

## 9. Generated Files

Day 22 generated the following files:

```text
docs/day22_confusion_matrix.csv
docs/day22_confusion_matrix_metrics.csv
docs/figures/day22_confusion_matrix.png
src/day22_confusion_matrix.py
```

### File descriptions

**`day22_confusion_matrix.csv`**

Contains the complete 8 × 8 confusion matrix.

**`day22_confusion_matrix_metrics.csv`**

Contains TP, FP, TN, and FN for every class.

**`day22_confusion_matrix.png`**

Contains the visual heatmap of the confusion matrix.

**`day22_confusion_matrix.py`**

Contains the complete Day 22 implementation.

---

## 10. Problems Observed

The main issue identified from the confusion matrix is the difficulty in detecting some minority attack classes.

In particular:

* Web Attack was completely classified as BENIGN.
* Infiltration was completely classified as BENIGN.
* Many Botnet samples were classified as BENIGN.
* Some DoS and DDoS samples were also classified as BENIGN.

This does not mean the Day 22 implementation failed. Instead, the confusion matrix provides evidence about the limitations of the current centralized XGBoost baseline.

These observations will be useful when comparing the supervised baseline with the later federated deep reinforcement learning approach.

---

## 11. Day 22 Conclusion

Day 22 successfully generated and analyzed the confusion matrix of the saved XGBoost baseline without retraining the model.

The confusion matrix provided a more detailed view of model behavior than overall accuracy alone.

The main finding is that the baseline performs well on several common classes but has difficulty identifying some minority attack classes, especially Web Attack, Infiltration, and Botnet.

This gives a clear evaluation reference for the later FDRL-IDS model.

---

## 12. Day 22 Status

**Status: COMPLETED**

The confusion matrix, TP/FP/TN/FN calculations, and visualization were successfully generated and saved in the project repository.
