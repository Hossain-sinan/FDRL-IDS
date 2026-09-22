# DAY 25 — BASELINE IDS FINAL REPORT

## Project

**Federated Deep Reinforcement Learning Intrusion Detection System (FDRL-IDS)**

## Day 25 Objective

The main objective of Day 25 was to prepare the final report for the centralized machine learning baseline of the FDRL-IDS project.

The baseline was developed using **XGBoost** on the CICIDS2017 dataset.

At the end of Day 25, the XGBoost baseline was finalized and frozen. This baseline will later be used as a reference when evaluating the Deep Reinforcement Learning (DRL) and Federated DRL models.

---

# 1. Dataset

The project uses the **CICIDS2017** intrusion detection dataset.

The original dataset contains 8 CSV files.

### Dataset processing

| Stage                  | Number of Records |
| ---------------------- | ----------------: |
| Original dataset       |         2,830,743 |
| After data cleaning    |         2,520,798 |
| After label processing |         2,520,787 |
| Final features         |                20 |
| Number of classes      |                 8 |
| Test samples           |           252,079 |

During preprocessing, missing/invalid values and duplicate records were handled.

The final dataset contains eight attack/traffic classes:

1. BENIGN
2. DoS
3. DDoS
4. Brute Force
5. Web Attack
6. Infiltration
7. Botnet
8. Port Scan

---

# 2. Selected Features

After the feature-selection experiments, 20 features were selected for the baseline model.

The final features are:

1. `min_seg_size_forward`
2. `PSH Flag Count`
3. `Init_Win_bytes_forward`
4. `Init_Win_bytes_backward`
5. `Flow Duration`
6. `ACK Flag Count`
7. `Destination Port`
8. `Bwd Packet Length Min`
9. `Bwd Packet Length Mean`
10. `Fwd IAT Total`
11. `Fwd IAT Std`
12. `Bwd IAT Total`
13. `Flow IAT Max`
14. `Average Packet Size`
15. `Packet Length Mean`
16. `Fwd IAT Max`
17. `Bwd Header Length`
18. `Packet Length Std`
19. `Min Packet Length`
20. `URG Flag Count`

These 20 features were kept fixed for the baseline experiments so that different model configurations could be compared fairly.

---

# 3. Train, Validation and Test Split

The dataset was divided using a stratified split with:

* Random state: **42**
* Training samples: **2,016,629**
* Validation samples: **252,079**
* Test samples: **252,079**

The test set was kept separate and was used for the final evaluation.

Using the same split is important because it makes the comparison between different XGBoost configurations fair.

---

# 4. XGBoost Baseline

XGBoost was selected as the supervised machine learning baseline.

The first baseline was created on Day 20.

After that, several controlled configurations were tested on Day 23 to determine whether the baseline could be improved.

No SMOTE or class weighting was added during these experiments.

This was intentional because the purpose was to keep the baseline simple and consistent.

---

# 5. Tested XGBoost Configurations

Four configurations were compared.

| Configuration  | Max Depth | Learning Rate | Estimators |     Accuracy |     Macro F1 |
| -------------- | --------: | ------------: | ---------: | -----------: | -----------: |
| Day20 Original |         3 |          0.10 |         20 |     98.4596% |    66.35396% |
| Config A       |         4 |          0.05 |        100 |     99.6592% |     81.2717% |
| Config B       |         6 |          0.05 |        200 | **99.8453%** | **87.4321%** |
| Config C       |         6 |          0.03 |        300 |     99.8370% |     86.9177% |

Config B produced the highest Macro F1 among the tested configurations.

Therefore, **Config B was selected as the final baseline configuration**.

---

# 6. Final XGBoost Configuration

The final baseline uses the following parameters:

| Parameter            |    Value |
| -------------------- | -------: |
| Model                |  XGBoost |
| Max depth            |        6 |
| Learning rate        |     0.05 |
| Number of estimators |      200 |
| Subsample            |     0.80 |
| Column sampling      |     0.80 |
| Random state         |       42 |
| Tree method          |     hist |
| Number of classes    |        8 |
| Selected features    |       20 |
| SMOTE                | Not used |
| Class weighting      | Not used |

This configuration is now considered the **frozen centralized supervised baseline** for the project.

---

# 7. Final Baseline Results

The final Config B model was evaluated on the same test set.

### Overall results

| Metric             |       Result |
| ------------------ | -----------: |
| Accuracy           | **99.8453%** |
| Macro Precision    | **96.4764%** |
| Macro Recall       | **82.4992%** |
| Macro F1           | **87.4321%** |
| Weighted Precision | **99.8362%** |
| Weighted Recall    | **99.8453%** |
| Weighted F1        | **99.8337%** |

The overall accuracy is very high.

However, accuracy alone is not enough for an intrusion detection system because the dataset is imbalanced.

For this reason, **Macro Recall and Macro F1 are especially important**.

---

# 8. Per-Class Results

The detailed class-level results show that the model performs differently for different attack types.

| Class        | Precision |   Recall |       F1 |
| ------------ | --------: | -------: | -------: |
| BENIGN       |  99.9003% | 99.9165% | 99.9084% |
| DoS          |  99.7629% | 99.9226% | 99.8427% |
| DDoS         |  99.9453% | 99.9844% | 99.9649% |
| Brute Force  | 100.0000% | 99.5628% | 99.7809% |
| Web Attack   |  80.7339% | 40.9302% | 54.3210% |
| Infiltration | 100.0000% | 50.0000% | 66.6667% |
| Botnet       |  92.5170% | 69.7436% | 79.5322% |
| Port Scan    |  98.9519% | 99.9338% | 99.4404% |

---

# 9. What Do These Results Mean?

The model performs very well for:

* BENIGN traffic
* DoS
* DDoS
* Brute Force
* Port Scan

For example, the DDoS recall is **99.9844%**, meaning almost all DDoS samples in the test set were correctly detected.

However, some classes are more difficult.

### Web Attack

Web Attack recall is only **40.93%**.

There were 215 Web Attack samples in the test set.

Only 88 were correctly detected, while 127 were classified as BENIGN.

This means the model has difficulty distinguishing some Web Attack traffic from normal traffic.

### Infiltration

There were only **4 Infiltration samples** in the test set.

The model correctly detected 2 of them.

Therefore, the recall is 50%.

Because the number of samples is extremely small, this result should not be treated as strong evidence about the model's general Infiltration detection ability.

### Botnet

Botnet recall is **69.74%**.

The model correctly detected 136 of 195 Botnet samples.

The remaining 59 were mainly classified as BENIGN.

This shows that Botnet detection is another area where the baseline has room for improvement.

---

# 10. False Positive Rate

False Positive Rate (FPR) is important for an IDS because incorrectly identifying normal traffic as an attack can create unnecessary alerts.

The measured FPR values were:

| Class        |     FPR |
| ------------ | ------: |
| BENIGN       | 0.4909% |
| DoS          | 0.0198% |
| DDoS         | 0.0029% |
| Brute Force  | 0.0000% |
| Web Attack   | 0.0083% |
| Infiltration | 0.0000% |
| Botnet       | 0.0044% |
| Port Scan    | 0.0395% |

Most classes have very low FPR.

This indicates that the model generally produces few false alarms on the test data.

---

# 11. Confusion Matrix

The final confusion matrix is:

| Actual / Predicted | BENIGN |   DoS |  DDoS | Brute Force | Web Attack | Infiltration | Botnet | Port Scan |
| ------------------ | -----: | ----: | ----: | ----------: | ---------: | -----------: | -----: | --------: |
| BENIGN             | 209331 |    42 |     7 |           0 |         19 |            0 |     11 |        96 |
| DoS                |     15 | 19359 |     0 |           0 |          0 |            0 |      0 |         0 |
| DDoS               |      2 |     0 | 12799 |           0 |          0 |            0 |      0 |         0 |
| Brute Force        |      3 |     0 |     0 |         911 |          1 |            0 |      0 |         0 |
| Web Attack         |    127 |     0 |     0 |           0 |         88 |            0 |      0 |         0 |
| Infiltration       |      2 |     0 |     0 |           0 |          0 |            2 |      0 |         0 |
| Botnet             |     59 |     0 |     0 |           0 |          0 |            0 |    136 |         0 |
| Port Scan          |      1 |     4 |     0 |           0 |          1 |            0 |      0 |      9063 |

The confusion matrix clearly shows that the main classification problems are:

* Web Attack → BENIGN
* Botnet → BENIGN
* Infiltration → BENIGN

This is important because it explains why the overall accuracy is very high while Macro Recall is lower.

---

# 12. Improvement Over the Original Baseline

The original Day 20 model had a Macro F1 of:

**66.35396%**

After controlled parameter experiments, Config B achieved:

**87.43214%**

Therefore, Macro F1 improved by approximately:

**21.08 percentage points.**

Other improvements were also observed:

| Metric          |   Day 20 | Final Config B | Improvement |
| --------------- | -------: | -------------: | ----------: |
| Accuracy        | 98.4596% |       99.8453% |  +1.3857 pp |
| Macro Precision | 72.0837% |       96.4764% | +24.3927 pp |
| Macro Recall    | 63.5934% |       82.4992% | +18.9058 pp |
| Macro F1        | 66.3540% |       87.4321% | +21.0781 pp |
| Weighted F1     | 98.3673% |       99.8337% |  +1.4664 pp |

The biggest improvement is seen in the macro-level metrics.

This is important because macro metrics give equal importance to each class and are therefore more informative for this imbalanced intrusion detection dataset.

---

# 13. Reproducibility Test

On Day 24, the final Config B model was trained again from scratch.

The purpose was to check whether the same experiment produces the same results.

The Day 23 and Day 24 results matched.

The small differences were approximately:

**1.11 × 10⁻¹⁶**

These differences are only floating-point numerical precision and are practically zero.

The reproducibility test also showed:

* Metric reproducibility: **True**
* Prediction reproducibility: **True**
* Different predictions on test samples: **0**

Therefore, the baseline experiment was successfully reproduced under the same conditions.

---

# 14. Important Interpretation

The final XGBoost model gives very high overall performance.

However, the result should not be interpreted as meaning that every attack type is detected equally well.

The results show three important points:

### 1. Strong performance on common classes

The model detects BENIGN, DoS, DDoS, Brute Force and Port Scan traffic very effectively.

### 2. Minority attacks are more difficult

Web Attack and Botnet have lower recall.

Infiltration also has a low recall, but its test support is only 4 samples, so that result is not statistically strong.

### 3. Accuracy is not enough

The accuracy of 99.8453% looks excellent, but Macro F1 is 87.4321%.

This difference happens because the dataset contains many more samples from some classes than others.

Therefore, future FDRL-IDS experiments should report:

* Accuracy
* Precision
* Recall
* F1-score
* Macro F1
* Per-class Recall
* False Positive Rate
* Confusion Matrix

rather than using accuracy alone.

---

# 15. Why This Baseline Is Important for FDRL-IDS

This XGBoost model is a **supervised centralized baseline**.

It does not perform federated learning and it does not use reinforcement learning.

Its purpose is to provide a reference point.

Later in the project, the DRL and FDRL models can be evaluated using the same general dataset and relevant metrics.

Then we can investigate questions such as:

* Can DRL detect attacks effectively?
* Can federated learning work under non-IID client data?
* How does federated learning compare with centralized learning?
* Does federated training improve or reduce performance?
* How does the model behave when different organizations have different attack distributions?
* Can the FDRL approach maintain useful detection performance without centralizing all client data?

The XGBoost baseline therefore provides an important reference for these later experiments.

---

# 16. Baseline Freeze Decision

After completing:

* XGBoost baseline training
* Parameter improvement experiments
* Per-class evaluation
* Confusion matrix analysis
* False Positive Rate analysis
* Reproducibility testing
* Final baseline documentation

the Config B model is now **frozen as the centralized supervised baseline**.

The baseline should not be continuously changed during the later DRL/FDRL experiments.

If the baseline is changed later, it should be treated as a new experiment rather than silently replacing this baseline.

---

# 17. Files Produced

The important project files are:

```text
src/
├── day20_xgboost_baseline.py
├── day23_improve_xgboost.py
├── day23_evaluate_improved_xgboost.py
└── day24_reproducibility_test.py

models/
├── xgboost_baseline.json
├── xgboost_improved_baseline.json
└── xgboost_day24_reproducibility.json

docs/
├── day16_features_20.csv
├── day23_xgboost_comparison.csv
├── day23_improved_classification_report.csv
├── day23_improved_per_class_metrics.csv
├── day23_improved_confusion_matrix.csv
└── day24_reproducibility_results.csv
```

---

# 18. Final Conclusion

The centralized XGBoost baseline has been successfully completed.

The final model achieved:

**Accuracy:** 99.8453%

**Macro Precision:** 96.4764%

**Macro Recall:** 82.4992%

**Macro F1:** 87.4321%

**Weighted F1:** 99.8337%

The model performs extremely well on several major traffic classes, while Web Attack, Botnet, and the very small Infiltration test group remain more difficult to detect.

The reproducibility experiment also confirmed that the final baseline can be reproduced under the same experimental conditions.

Therefore, the centralized supervised IDS baseline is now considered **complete and frozen**.

The project can now move to the next major stage: **Deep Reinforcement Learning and Federated Deep Reinforcement Learning**.

## Day 25 Milestone

### 🔒 BASELINE IDS COMPLETE

**Next stage: DRL development and federated learning implementation.**
