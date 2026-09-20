# Day 23 — Improve XGBoost Baseline Properly

## 1. Objective

The main objective of Day 23 was to improve the existing centralized XGBoost baseline from Day 20 using controlled model parameter changes.

The purpose was not to search for a perfect score. The goal was to check whether a stronger but still reasonable XGBoost configuration could improve the detection of minority attack classes while using the same dataset, selected features, train/test split, and evaluation procedure.

---

## 2. What Was Changed

The original Day 20 XGBoost configuration was compared with three new configurations.

| Configuration  | Max Depth | Learning Rate | Estimators | Subsample | Column Sample |
| -------------- | --------: | ------------: | ---------: | --------: | ------------: |
| Day20 Original |         3 |          0.10 |         20 |      0.80 |          0.80 |
| Config A       |         4 |          0.05 |        100 |      0.80 |          0.80 |
| Config B       |         6 |          0.05 |        200 |      0.80 |          0.80 |
| Config C       |         6 |          0.03 |        300 |      0.80 |          0.80 |

No SMOTE, class weighting, or additional feature engineering was introduced in this experiment.

This was intentional so that the effect of the XGBoost parameter changes could be observed more clearly.

---

## 3. Experiment Results

| Configuration  |     Accuracy | Macro Precision | Macro Recall |     Macro F1 |  Weighted F1 |
| -------------- | -----------: | --------------: | -----------: | -----------: | -----------: |
| Day20 Original |     98.4596% |        72.0837% |     63.5934% |     66.3540% |     98.3673% |
| Config A       |     99.6592% |        98.8845% |     76.6232% |     81.2717% |     99.6209% |
| **Config B**   | **99.8453%** |    **96.4764%** | **82.4992%** | **87.4321%** | **99.8337%** |
| Config C       |     99.8370% |        96.8347% |     81.7875% |     86.9177% |     99.8232% |

Config B produced the highest Macro F1 and Macro Recall among the tested configurations.

Therefore, Config B was selected as the improved centralized XGBoost baseline.

---

## 4. Improvement Compared With Day 20

The most important improvement was not only accuracy.

| Metric          |   Day 20 | Day 23 Config B |     Improvement |
| --------------- | -------: | --------------: | --------------: |
| Accuracy        | 98.4596% |        99.8453% |      +1.3857 pp |
| Macro Precision | 72.0837% |        96.4764% |     +24.3927 pp |
| Macro Recall    | 63.5934% |        82.4992% |     +18.9058 pp |
| Macro F1        | 66.3540% |        87.4321% | **+21.0781 pp** |
| Weighted F1     | 98.3673% |        99.8337% |      +1.4664 pp |

The Macro F1 improvement is especially important because the dataset contains highly imbalanced attack classes.

The original model had high overall accuracy but much lower macro-level performance. The improved configuration substantially reduced this gap.

---

## 5. Class-Level Improvement

The saved Config B model was evaluated again without retraining.

The class-level recall results show where the improvement happened.

| Class        | Day 20 Recall | Day 23 Recall |        Change |
| ------------ | ------------: | ------------: | ------------: |
| BENIGN       |      99.9266% |      99.9165% |    -0.0101 pp |
| DoS          |        85.56% |      99.9226% | **+14.36 pp** |
| DDoS         |        97.01% |      99.9844% |  **+2.97 pp** |
| Brute Force  |        96.28% |      99.5628% |  **+3.28 pp** |
| Web Attack   |            0% |    **40.93%** | **+40.93 pp** |
| Infiltration |            0% |    **50.00%** | **+50.00 pp** |
| Botnet       |        31.80% |    **69.74%** | **+37.94 pp** |
| Port Scan    |        98.14% |  **99.9338%** |  **+1.79 pp** |

The most important changes were observed in previously weak classes:

* Web Attack improved from 0% to 40.93% recall.
* Infiltration improved from 0% to 50% recall.
* Botnet improved from 31.80% to 69.74% recall.
* DoS improved from approximately 85.56% to 99.92% recall.

This shows that the improvement in Macro F1 was not only caused by the majority BENIGN class.

---

## 6. Confusion Matrix Findings

The Day 23 confusion matrix also shows the improvement.

### Web Attack

Previously, the Day 20 model classified all 215 Web Attack test samples as BENIGN.

In Day 23:

* 88 Web Attack samples were correctly detected.
* 127 were still classified as BENIGN.

Therefore, Web Attack recall improved to:

**40.93%**

However, the class is still difficult for the model.

### Infiltration

There were only 4 Infiltration samples in the test set.

Day 23 correctly detected:

* 2 Infiltration samples
* 2 were classified as BENIGN

Therefore:

**50% recall**

Because the test support is only 4 samples, this result should not be treated as strong statistical evidence.

### Botnet

Day 23 detected:

* 136 of 195 Botnet samples correctly
* 59 were classified as BENIGN

Recall:

**69.74%**

This is a substantial improvement compared with the previous 31.80% recall.

### DoS

Day 23 correctly detected:

**19,359 of 19,374 DoS samples**

Only 15 DoS samples were missed.

This resulted in approximately:

**99.92% recall**

---

## 7. Important Observation About Config B and Config C

Config C used more trees than Config B:

* Config B: 200 trees
* Config C: 300 trees

However, Config C did not improve the results.

| Metric       |     Config B | Config C |
| ------------ | -----------: | -------: |
| Accuracy     | **99.8453%** | 99.8370% |
| Macro Recall | **82.4992%** | 81.7875% |
| Macro F1     | **87.4321%** | 86.9177% |

This shows that simply increasing the number of trees does not automatically produce better performance.

The experiment therefore provides a useful parameter-tuning result rather than simply increasing model complexity.

---

## 8. What Was Actually Improved Today?

Day 23 improved the **model configuration**, not the dataset or feature set.

The following remained unchanged:

* CICIDS2017 dataset
* 20 selected features
* 8-class classification problem
* Same train/test methodology
* Same random seed
* Same evaluation approach
* No SMOTE
* No class weighting

The main changes were:

* Increased tree depth
* Reduced learning rate
* Increased number of estimators
* Compared multiple controlled configurations

The best configuration substantially improved macro-level classification performance.

---

## 9. Remaining Limitations

The improved model is better, but it is not perfect.

### Web Attack

Recall is still only:

**40.93%**

A significant number of Web Attack samples are still classified as BENIGN.

### Infiltration

Recall is:

**50%**

But the test set contains only 4 Infiltration samples, so this result has very limited statistical reliability.

### Botnet

Recall improved to:

**69.74%**

but 59 Botnet samples are still classified as BENIGN.

Therefore, the model still has difficulty detecting some rare attack categories.

These limitations are important because they give a clear research reason for later investigation of more advanced approaches.

---

## 10. Files Generated

Day 23 produced the following important files:

```text
src/day23_improve_xgboost.py
src/day23_evaluate_improved_xgboost.py

docs/day23_xgboost_comparison.csv
docs/day23_improved_classification_report.csv
docs/day23_improved_per_class_metrics.csv
docs/day23_improved_confusion_matrix.csv

docs/figures/day23_improved_confusion_matrix.png

models/xgboost_improved_baseline.json
```

---

## 11. Final Result

The original Day 20 XGBoost model achieved:

**98.4596% Accuracy**
**66.3540% Macro F1**

After controlled parameter tuning, Config B achieved:

**99.8453% Accuracy**
**87.4321% Macro F1**

The improvement in Macro F1 was:

**+21.0781 percentage points**

The most important class-level improvements were observed in DoS, Web Attack, Infiltration, and Botnet.

Therefore, the Day 23 experiment successfully established a stronger centralized XGBoost baseline for the FDRL-IDS project.

However, the remaining weakness in rare attack classes will be kept as an important limitation rather than claiming that the problem has been completely solved.

This improved XGBoost model can now be used as the stronger supervised ML reference when the later DRL and Federated DRL approaches are evaluated.
