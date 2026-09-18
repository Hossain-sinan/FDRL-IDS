# Day 21 — XGBoost Baseline Evaluation

## Goal

The main goal of Day 21 was to **evaluate the XGBoost model created on Day 20 without retraining it**.

Day 20 focused on training the centralized XGBoost baseline.
Day 21 focused on deeper evaluation of that saved model, especially **False Positive Rate (FPR)** and class-wise detection performance.

---

## 1. What I Did Today

Today I performed the following new work:

* Loaded the saved XGBoost model from Day 20.
* Recreated the same train/validation/test split used during Day 20.
* Used the same test set for evaluation to keep the comparison fair.
* Generated predictions from the saved model.
* Calculated overall evaluation metrics again.
* Calculated **per-class True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN)**.
* Calculated **one-vs-rest FPR for every attack class**.
* Calculated **Macro FPR**.
* Calculated **binary Attack-vs-BENIGN FPR**.
* Created separate CSV files for the new evaluation results.
* Analyzed the weaknesses of the centralized baseline.

The important point is that **the model was not trained again on Day 21**.

---

## 2. Saved Model Evaluation

I loaded the already trained model:

```text
models/xgboost_baseline.json
```

The purpose was to evaluate the existing baseline rather than create another model.

This makes Day 21 an **evaluation and analysis day**, not another training day.

---

## 3. Same Test Split

For a fair evaluation, I recreated the exact test split from Day 20 using:

```text
80% Training
10% Validation
10% Testing

random_state = 42
```

The resulting test set contained:

```text
252,079 samples
```

Using the same test set is important because changing the test data could make the Day 20 and Day 21 results difficult to compare.

---

# 4. New Metric: False Positive Rate

The main new evaluation work on Day 21 was adding **False Positive Rate (FPR)**.

FPR measures how many negative samples are incorrectly predicted as positive.

The formula is:

```text
FPR = FP / (FP + TN)
```

For an IDS, FPR is important because false alarms can cause unnecessary security alerts.

---

# 5. Per-Class FPR

I calculated FPR separately for each of the 8 classes using a one-vs-rest approach.

| Class        |     TP |   FP |     TN |   FN |     FPR |
| ------------ | -----: | ---: | -----: | ---: | ------: |
| BENIGN       | 209352 | 3686 |  38887 |  154 | 8.6581% |
| DoS          |  16582 |   45 | 232660 | 2792 | 0.0193% |
| DDoS         |  12419 |    5 | 239273 |  382 | 0.0021% |
| Brute Force  |    881 |   26 | 251138 |   34 | 0.0104% |
| Web Attack   |      0 |   10 | 251854 |  215 | 0.0040% |
| Infiltration |      0 |    0 | 252075 |    4 | 0.0000% |
| Botnet       |     62 |   13 | 251871 |  133 | 0.0052% |
| Port Scan    |   8900 |   98 | 242912 |  169 | 0.0403% |

This table was newly generated on Day 21.

---

# 6. Important FPR Finding

The calculated **Macro FPR** was:

```text
0.010924
```

or approximately:

```text
1.0924%
```

However, this should not be confused with the IDS false-alarm rate.

The BENIGN one-vs-rest FPR was:

```text
8.6581%
```

This is a different measurement because it treats BENIGN as the positive class.

Therefore, I should not report:

> "The IDS has an 8.66% false alarm rate."

That would be misleading.

---

# 7. Binary Attack-vs-BENIGN FPR

For IDS analysis, I also converted the prediction into two groups:

```text
BENIGN = 0
ATTACK = all non-BENIGN classes
```

The result was:

```text
False Positives = 154
True Negatives  = 209,352
```

Therefore:

```text
Binary Attack FPR = 154 / (154 + 209352)
                  = 0.000735
                  = 0.0735%
```

This means that among the actual benign test traffic, only about **0.0735% was incorrectly classified as attack** under this binary interpretation.

This is a more directly useful false-alarm measurement for the IDS use case.

---

# 8. New Class-Level Observation

The Day 21 evaluation also made the minority-class problem clearer.

The model performed much better on classes with more training examples, while some very small classes were not detected properly.

The important observations were:

### Web Attack

```text
TP = 0
FN = 215
Recall = 0%
```

The model failed to correctly detect the Web Attack samples in the test set.

### Infiltration

```text
TP = 0
FN = 4
Recall = 0%
```

The number of Infiltration test samples was extremely small, so this result should be interpreted carefully.

### Botnet

The model detected:

```text
TP = 62
FN = 133
```

So Botnet detection was also considerably weaker than the major classes.

### DoS

The model detected most DoS samples, but some were still missed:

```text
TP = 16,582
FN = 2,792
```

This shows that overall accuracy alone does not describe the model's performance across all attack categories.

---

# 9. Why Day 21 Evaluation Was Necessary

Day 20 already showed that the model achieved high overall accuracy.

However, Day 21 showed that there is more to evaluate than accuracy.

The new analysis helped answer:

```text
How many benign samples are incorrectly treated as attacks?
How many samples of each attack class are detected?
Which classes are difficult for the model?
How does class imbalance affect the results?
```

This gives a more useful understanding of the centralized baseline before moving toward the federated deep reinforcement learning system.

---

# 10. Files Created on Day 21

I created the following new files:

```text
src/day21_evaluate_xgboost.py

docs/day21_baseline_results.csv

docs/day21_per_class_metrics.csv

docs/day21_evaluate_xgboost.md
```

The evaluation script performs the new Day 21 analysis and saves the results for later comparison.

---

# 11. Day 21 Research Observation

The main research observation from today is:

> The centralized XGBoost baseline has high overall performance, but its class-wise performance is not uniform. The new FPR analysis also shows that binary attack-vs-BENIGN false positives are much lower than the one-vs-rest BENIGN FPR, so these metrics must be clearly separated in future experiments.

This is important for the later FDRL-IDS experiments because the federated DRL system should not be evaluated only by overall accuracy.

---

# 12. What I Learned Today

Today I learned:

* How to evaluate a saved machine learning model without retraining.
* Why the same test split should be used for fair comparison.
* What False Positive Rate means.
* How to calculate FPR using TP, FP, TN and FN.
* Difference between one-vs-rest FPR and binary Attack-vs-BENIGN FPR.
* Why false alarms are important for an IDS.
* Why high accuracy does not guarantee good detection of minority attack classes.
* Why class-wise evaluation is necessary for an intrusion detection project.

---

# 13. Day 21 Completion Status

```text
[✓] Saved XGBoost model loaded
[✓] Same test split reproduced
[✓] Model evaluated without retraining
[✓] Per-class confusion values calculated
[✓] Per-class FPR calculated
[✓] Macro FPR calculated
[✓] Binary Attack-vs-BENIGN FPR calculated
[✓] Minority-class performance analyzed
[✓] Evaluation CSV files created
[✓] Day 21 documentation completed
```

## Final Summary

Day 21 was focused on **deeper evaluation of the Day 20 XGBoost baseline**.

The main new contribution was the FPR-based analysis. The binary Attack-vs-BENIGN evaluation produced an FPR of **0.0735%**, while the macro one-vs-rest FPR was **1.0924%**.

The evaluation also showed that the model's performance is not equally strong across all attack classes, especially for very small minority classes such as Web Attack, Infiltration, and Botnet.

Therefore, Day 21 provides the detailed evaluation information that will be useful when comparing the centralized baseline with the future **DRL and Federated DRL IDS models**.
