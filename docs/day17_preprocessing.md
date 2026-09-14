# Day 17 — Normalization, Class Imbalance and SMOTE

## 1. Day 17 Goal

The main goal of Day 17 was to prepare the selected CICIDS2017 features for the next machine learning and reinforcement learning stages of the FDRL-IDS project.

The main objectives were:

* Understand feature normalization.
* Understand the class imbalance problem.
* Create a proper training, validation and testing split.
* Prevent data leakage during preprocessing.
* Investigate SMOTE for minority attack classes.
* Decide where SMOTE can safely be used.
* Build a preprocessing approach that can later support the federated learning experiments.

A major focus of this day was making sure that validation and test information does not influence the training process.

---

# 2. Starting Point

Before Day 17, the following work had already been completed:

### Day 13

A reusable CICIDS2017 data loader was created.

### Day 14

The dataset was cleaned by handling:

* Infinity values
* NaN values
* Duplicate records

The original dataset contained:

```text
Rows:       2,830,743
Columns:    79
```

After cleaning:

```text
Rows:       2,520,798
Columns:    79
```

A total of:

```text
309,945 rows
```

were removed.

### Day 15

The original attack labels were converted into the project-level attack categories.

The final classes were:

```text
0 = BENIGN
1 = DoS
2 = DDoS
3 = Brute Force
4 = Web Attack
5 = Infiltration
6 = Botnet
7 = Port Scan
```

There were 11 Heartbleed records that could not be mapped to the selected project categories, so they were removed.

Final processed dataset:

```text
2,520,787 rows
82 columns
```

### Day 16

Feature selection was investigated.

A 20-feature candidate set was created from the feature-selection experiments.

These 20 features were used for the Day 17 preprocessing experiment.

Important:

The 20 features are still a candidate feature set. They are not yet declared as the final feature count for FDRL-IDS.

---

# 3. Day 17 Step 1 — Data Loading

The CICIDS2017 loader successfully found all 8 CSV files.

The individual files were loaded and combined.

The combined dataset was:

```text
(2,830,743, 79)
```

This matched the previous dataset analysis.

Therefore, the data loader was working correctly.

---

# 4. Problem 1 — Accidentally Running the Wrong File

During the Day 17 work, the first command was:

```powershell
python -u "d:\Projects\FDRL-IDS\src\preprocessing.py"
```

This was not the correct Day 17 SMOTE experiment file.

The correct file was:

```text
src/day17_smote_experiment.py
```

The file:

```text
src/preprocessing.py
```

belongs to the Day 14 data-cleaning stage.

## Why this was a problem

The Day 14 preprocessing file and the Day 17 preprocessing experiment have different responsibilities.

Changing the Day 14 file could accidentally affect the previously completed cleaning pipeline.

## How the problem was handled

The Day 14 `src/preprocessing.py` file was kept as the original cleaning module.

The new Day 17 experiment was separated into:

```text
src/day17_preprocessing.py
src/day17_smote_experiment.py
```

This keeps the project modular and prevents old work from being accidentally overwritten.

---

# 5. Problem 2 — `imblearn` Module Error

The first execution produced:

```text
ModuleNotFoundError: No module named 'imblearn'
```

## Why this happened

The command was executed before activating the project's virtual environment.

The system was therefore using a different Python environment.

The project environment was:

```text
.venv
```

and it contains the required machine learning packages.

## How the problem was handled

The virtual environment was activated:

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& d:\Projects\.venv\Scripts\Activate.ps1)
```

The terminal then showed:

```text
(.venv)
```

This confirmed that the project virtual environment was active.

The required package was already installed in this environment:

```text
imbalanced-learn
```

Therefore, another unnecessary installation was not performed.

## Lesson

Before running project scripts, always verify that:

```text
(.venv)
```

appears in the PowerShell prompt.

---

# 6. Problem 3 — `preprocessing` Import Error

After activating the virtual environment, another error occurred:

```text
ModuleNotFoundError: No module named 'preprocessing'
```

This happened while running:

```text
src/day17_smote_experiment.py
```

## Why this happened

The script imports local project modules:

```python
from data_loader import load_cicids2017
from preprocessing import clean_data
from label_processing import process_labels
```

The script was being launched from:

```text
D:\Projects
```

rather than from the project directory.

## How the problem was handled

The project directory was identified as:

```text
D:\Projects\FDRL-IDS
```

The script was then executed from the project context.

The important project structure is:

```text
FDRL-IDS/
│
├── data/
├── docs/
└── src/
    ├── data_loader.py
    ├── preprocessing.py
    ├── label_processing.py
    └── day17_smote_experiment.py
```

After correcting the execution context, the script successfully imported the required modules.

## Lesson

For this project, project scripts should be executed from:

```text
D:\Projects\FDRL-IDS
```

using commands such as:

```powershell
python -u src\day17_smote_experiment.py
```

---

# 7. Day 17 Step 2 — Train, Validation and Test Split

The processed dataset was divided into three parts:

```text
80% Training
10% Validation
10% Test
```

Stratified splitting was used.

The final shapes were:

| Dataset    |          Shape |
| ---------- | -------------: |
| Training   | 2,016,629 × 20 |
| Validation |   252,079 × 20 |
| Test       |   252,079 × 20 |

Random state:

```text
42
```

was used to make the split reproducible.

---

# 8. Why Stratification Was Important

The dataset contains severe class imbalance.

If a normal random split were used without stratification, extremely rare classes could be distributed unevenly between training, validation and test sets.

This would be especially dangerous for:

```text
Infiltration
Web Attack
Botnet
```

Therefore, stratified splitting was used to preserve the approximate class proportions.

---

# 9. Class Imbalance Investigation

The training class distribution before SMOTE was:

| Class ID | Attack Category | Training Samples |
| -------: | --------------- | ---------------: |
|        0 | BENIGN          |        1,676,045 |
|        1 | DoS             |          154,996 |
|        2 | DDoS            |          102,411 |
|        3 | Brute Force     |            7,320 |
|        4 | Web Attack      |            1,714 |
|        5 | Infiltration    |               29 |
|        6 | Botnet          |            1,559 |
|        7 | Port Scan       |           72,555 |

This clearly shows severe class imbalance.

The BENIGN class contains the vast majority of records.

The most extreme problem is:

```text
Infiltration = only 29 training samples
```

Therefore, accuracy alone will not be a reliable metric later in the project.

Future model evaluation should consider metrics such as:

* Precision
* Recall
* F1-score
* Macro F1
* Per-class recall
* Confusion matrix
* False-positive rate

---

# 10. Day 17 Step 3 — Min-Max Normalization

Min-Max scaling was selected for the preprocessing experiment.

The purpose is to bring different numerical features into a comparable range.

The important implementation rule was:

```text
Fit scaler → Training data only
Transform → Training data
Transform → Validation data
Transform → Test data
```

The scaler was created using:

```python
MinMaxScaler()
```

and fitted using:

```python
scaler.fit(X_train)
```

The validation and test sets were not used to fit the scaler.

---

# 11. Problem 4 — Risk of Data Leakage

Data leakage was one of the most important problems considered during Day 17.

A bad approach would be:

```text
Complete dataset
       ↓
Fit scaler
       ↓
Train/Test split
```

This allows information from the future validation/test data to influence the scaling parameters.

That can make model evaluation unrealistically optimistic.

## Correct approach

The implemented approach was:

```text
Complete processed dataset
          ↓
Train / Validation / Test split
          ↓
Fit scaler ONLY on Training
          ↓
Transform Training
          ↓
Transform Validation
          ↓
Transform Test
```

The experiment confirmed:

```text
Data leakage prevention: PASSED
```

---

# 12. Important Note About Scaling Values

The training data had approximately:

```text
Minimum = 0.0
Maximum = 1.0
```

This is expected because the scaler was fitted using the training data.

Validation or test values can sometimes be slightly outside the training range because the scaler did not use those datasets to calculate its minimum and maximum.

This is acceptable and is preferable to fitting the scaler using all data.

---

# 13. Day 17 Step 4 — SMOTE Investigation

SMOTE means:

```text
Synthetic Minority Over-sampling Technique
```

It creates synthetic samples for minority classes rather than simply copying existing records.

The purpose was to investigate whether minority attack classes could receive additional training samples.

However, SMOTE was not applied to the complete dataset.

It was applied only after the train/validation/test split.

---

# 14. Problem 5 — How Much SMOTE Should Be Used?

One major issue was deciding how much to oversample.

A dangerous approach would be to increase every minority class to the size of BENIGN.

For example:

```text
Infiltration:

29 → 1,676,045
```

This would generate an enormous number of synthetic samples from only 29 original examples.

This could:

* increase computational cost
* create unrealistic synthetic patterns
* overfit rare-class characteristics
* distort the original dataset
* make the experiment less meaningful

Therefore, full majority balancing was rejected.

---

# 15. Controlled SMOTE Strategy

A small controlled experiment was used.

The following classes were selected:

```text
Class 4 — Web Attack
Class 5 — Infiltration
Class 6 — Botnet
```

The sampling targets were:

| Class        | Before | Target |
| ------------ | -----: | -----: |
| Web Attack   |  1,714 |  2,500 |
| Infiltration |     29 |    100 |
| Botnet       |  1,559 |  2,500 |

The larger classes were left unchanged.

---

# 16. SMOTE Result

After applying controlled SMOTE:

| Class ID | Class        |    Before |     After |
| -------: | ------------ | --------: | --------: |
|        0 | BENIGN       | 1,676,045 | 1,676,045 |
|        1 | DoS          |   154,996 |   154,996 |
|        2 | DDoS         |   102,411 |   102,411 |
|        3 | Brute Force  |     7,320 |     7,320 |
|        4 | Web Attack   |     1,714 |     2,500 |
|        5 | Infiltration |        29 |       100 |
|        6 | Botnet       |     1,559 |     2,500 |
|        7 | Port Scan    |    72,555 |    72,555 |

Original training size:

```text
2,016,629
```

Training size after SMOTE:

```text
2,018,427
```

Total synthetic samples generated:

```text
1,798
```

This was intentionally kept small.

---

# 17. Problem 6 — Should SMOTE Be Applied to Validation or Test Data?

No.

Applying SMOTE to validation or test data would change the real evaluation distribution.

For example, if the test set were oversampled, the final model would not be evaluated against the original data distribution.

## Correct decision

SMOTE was applied only to:

```text
Training data
```

SMOTE was NOT applied to:

```text
Validation data
Test data
```

The validation and test shapes remained:

```text
Validation = 252,079 × 20
Test       = 252,079 × 20
```

This is an important part of the leakage-safe experimental design.

---

# 18. Problem 7 — SMOTE and Federated Learning

This is especially important for FDRL-IDS.

The project will later simulate four organizations:

```text
Hospital
Bank
University
ISP
```

These organizations are intentionally designed to have different attack distributions.

This creates a non-IID federated learning environment.

If SMOTE is applied heavily before creating the clients, the natural differences between organizations may be reduced.

For example, artificially increasing rare attacks everywhere could make the clients more similar than they should naturally be.

That would weaken the non-IID research experiment.

Therefore:

> SMOTE will not automatically be applied to the final federated dataset.

Instead, SMOTE will remain an experimental option.

Its usefulness will be evaluated later based on model performance.

---

# 19. Baseline vs SMOTE Strategy

The project should eventually compare at least two approaches:

### Baseline

```text
Training data
      ↓
Normalization
      ↓
Model
```

### Controlled SMOTE

```text
Training data
      ↓
Normalization
      ↓
Controlled SMOTE
      ↓
Model
```

The validation and test data remain the same in both cases.

This allows a fair comparison.

---

# 20. Problem 8 — Should 20 Features Be the Final Feature Set?

No final decision was made on this during Day 17.

The 20 features were selected because they were the strongest candidate set from the Day 16 feature-selection experiment.

However, Day 16 also created:

```text
20-feature set
30-feature set
40-feature set
50-feature set
63-feature set
```

Therefore, the final feature count should be selected experimentally later.

The final decision should consider:

* Detection performance
* Macro F1
* Rare attack recall
* Computational cost
* DQN training stability
* Federated communication cost
* Non-IID client performance

---

# 21. Final Day 17 Pipeline

The current preprocessing design is:

```text
CICIDS2017
      ↓
Data Loader
      ↓
Data Cleaning
      ↓
Label Processing
      ↓
Candidate Feature Selection
      ↓
Train / Validation / Test Split
      ↓
Fit Min-Max Scaler on TRAIN ONLY
      ↓
Transform Train / Validation / Test
      ↓
      ┌────────────────────────────┐
      │                            │
      ↓                            ↓
Baseline Training          Controlled SMOTE
                            Training Only
      │                            │
      └──────────────┬─────────────┘
                     ↓
             Model Experiments
```

Later, this will connect to the federated client creation stage.

---

# 22. Files Created

Day 17 created:

```text
src/
├── day17_preprocessing.py
└── day17_smote_experiment.py

docs/
└── day17_preprocessing.md
```

The existing Day 14 file was kept separate:

```text
src/preprocessing.py
```

It continues to perform the original data-cleaning operation.

---

# 23. Final Problems Faced During Day 17

The main problems encountered were:

### Problem 1

Wrong script was executed.

**Solution:** Run `day17_smote_experiment.py` instead of modifying the Day 14 cleaning file.

### Problem 2

`imblearn` was not found.

**Cause:** The command was initially executed outside the project virtual environment.

**Solution:** Activate `.venv` before running the project.

### Problem 3

Local module `preprocessing` was not found.

**Cause:** The project script was launched from the wrong execution context.

**Solution:** Execute the script from the FDRL-IDS project directory.

### Problem 4

Severe class imbalance.

**Solution:** Investigate controlled SMOTE instead of blindly balancing all classes.

### Problem 5

Extremely rare Infiltration class.

**Solution:** Do not increase it to the majority-class size. Use a small controlled experiment.

### Problem 6

Risk of data leakage.

**Solution:** Split the data first and fit the scaler only on training data.

### Problem 7

SMOTE could distort the federated non-IID distribution.

**Solution:** Keep SMOTE as an experimental option instead of automatically applying it before client creation.

### Problem 8

Uncertainty about final feature count.

**Solution:** Keep the 20-feature set as a candidate and compare feature sets experimentally later.

---

# 24. Final Result

Day 17 was successfully completed.

The main results are:

```text
Processed records:       2,520,787
Candidate features:      20
Training records:        2,016,629
Validation records:        252,079
Test records:              252,079
SMOTE training records:  2,018,427
Synthetic samples:           1,798
```

The preprocessing pipeline successfully prevented validation and test information from being used to fit the scaler.

The controlled SMOTE experiment also successfully generated additional minority-class training samples without modifying the validation or test sets.

---

# 25. Research Decision After Day 17

The current decision is:

### Normalization

**Min-Max Scaling will be used.**

### Data splitting

**80% training / 10% validation / 10% test with stratification.**

### Data leakage

**Training-only fitting for preprocessing parameters.**

### SMOTE

**Controlled experimental technique, not automatically part of the final FDRL-IDS pipeline.**

### Feature count

**20 features remain a candidate, not the final decision.**

### Federated learning

The original non-IID organization distributions must be preserved when the federated clients are created.

---

# 26. Conclusion

Day 17 established the preprocessing foundation required for the next stages of FDRL-IDS.

The most important lesson from this day was that preprocessing is not only about scaling the data or fixing class imbalance. The preprocessing method must also preserve the validity of the research experiment.

In particular, the following principles will be maintained:

1. Test data will not influence training preprocessing.
2. Validation data will not be used for training preprocessing.
3. SMOTE will only be applied to training data.
4. Heavy oversampling will be avoided.
5. The natural non-IID characteristics required for federated learning will be protected.
6. The final feature count will be selected based on experimental evidence rather than an arbitrary number.

This provides a clean foundation for the next stage of the FDRL-IDS project.

