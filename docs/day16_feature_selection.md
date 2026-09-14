# Day 16 — Feature Selection

## Goal

The goal of Day 16 was to analyze the 78 original CICIDS2017 features and identify unnecessary or redundant features.

Feature selection is important for this project because using too many unnecessary features can increase computation time, make the model more complex, and may not improve the final detection performance.

For this reason, I did not directly select 40 or 50 features. Instead, I created different candidate feature sets that can be tested later.

---

## 1. Original Feature Set

After loading and preprocessing the CICIDS2017 dataset, there were:

* Total records: 2,520,787
* Original features: 78
* Data types: 54 integer features and 24 floating-point features

The label-related columns were not treated as input features.

---

## 2. Removing Constant Features

First, I checked whether any features had only one unique value throughout the dataset.

I found 8 constant features:

1. Bwd PSH Flags
2. Bwd URG Flags
3. Fwd Avg Bytes/Bulk
4. Fwd Avg Packets/Bulk
5. Fwd Avg Bulk Rate
6. Bwd Avg Bytes/Bulk
7. Bwd Avg Packets/Bulk
8. Bwd Avg Bulk Rate

These features do not provide useful variation because their values are constant.

After removing these constant features:

```text
78 features
    ↓
8 constant features removed
    ↓
70 candidate features
```

---

## 3. Correlation Analysis

Next, I performed correlation analysis on the 70 candidate features.

I used an absolute correlation threshold of 0.90 as a screening threshold.

The correlation matrix was:

```text
70 × 70
```

The analysis identified 71 feature pairs with absolute correlation greater than or equal to 0.90.

Some features were exactly or almost perfectly correlated. For example:

* Total Fwd Packets ↔ Subflow Fwd Packets = 1.0000
* Total Backward Packets ↔ Subflow Bwd Packets = 1.0000
* Total Length of Fwd Packets ↔ Subflow Fwd Bytes = 1.0000
* Total Length of Bwd Packets ↔ Subflow Bwd Bytes = 1.0000
* Fwd Packet Length Mean ↔ Avg Fwd Segment Size = 1.0000
* Bwd Packet Length Mean ↔ Avg Bwd Segment Size = 1.0000
* Fwd Header Length ↔ Fwd Header Length.1 = 1.0000

Correlation was used as a screening method. I did not automatically remove every highly correlated feature because some correlated features may still provide useful information for the model.

---

## 4. Removing Exact Redundant Features

I then selected seven exact redundant feature pairs for reduction.

For each pair, I compared their feature importance and kept the feature with the stronger importance signal.

The following features were removed:

1. Total Fwd Packets
2. Subflow Bwd Packets
3. Total Length of Fwd Packets
4. Subflow Bwd Bytes
5. Avg Fwd Segment Size
6. Avg Bwd Segment Size
7. Fwd Header Length

Therefore:

```text
70 candidate features
    ↓
7 redundant features removed
    ↓
63 features
```

The remaining 63 features were kept as the main candidate feature pool.

---

## 5. Feature Importance

After the redundancy analysis, I used ExtraTreesClassifier to estimate feature importance.

Because the dataset is highly imbalanced, I did not use only a simple random sample.

Instead, I created a class-aware sample so that rare attack classes were also represented.

The feature-importance sample contained 100,000 records.

The class-aware sample included all available records from the very small Infiltration class.

The ExtraTrees model was configured with:

* Number of trees: 100
* Random state: 42
* `class_weight="balanced"`
* `n_jobs=-1`

The top features from the feature-importance analysis included:

1. min_seg_size_forward
2. PSH Flag Count
3. Init_Win_bytes_forward
4. Init_Win_bytes_backward
5. Flow Duration
6. ACK Flag Count
7. Destination Port
8. Bwd Packet Length Min
9. Bwd Packet Length Mean
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

The complete ranking was saved for later use.

---

## 6. Candidate Feature Sets

Instead of selecting the final feature count at this stage, I created several candidate feature sets from the ranked 63-feature pool.

The candidate sets are:

| Candidate Set | Number of Features |
| ------------- | -----------------: |
| Set 1         |                 63 |
| Set 2         |                 50 |
| Set 3         |                 40 |
| Set 4         |                 30 |
| Set 5         |                 20 |

These sets will allow the project to compare different feature counts experimentally.

The 20-feature set contains the top 20 features from the feature-importance ranking.

---

## 7. Why I Did Not Select the Final Feature Count

The project proposal mentioned approximately 40–50 features, but I decided not to select the final number only from the proposal.

Instead, the final feature count should be supported by experimental results.

Later, I will compare the candidate feature sets based on factors such as:

* detection performance
* model performance
* computational cost
* training behavior
* suitability for federated learning
* suitability for the reinforcement learning model

Therefore, 20, 30, 40, 50, and 63 are currently candidate sets, not the final selected feature count.

---

## 8. Files Created

The following files were created during Day 16:

```text
docs/
├── day16_correlation_matrix.csv
├── day16_high_correlation_pairs.csv
├── day16_feature_importance.csv
├── day16_removed_redundant_features.csv
├── day16_candidate_features_63.csv
├── day16_features_20.csv
├── day16_features_30.csv
├── day16_features_40.csv
├── day16_features_50.csv
├── day16_features_63.csv
└── day16_feature_selection.md

src/
└── create_feature_sets.py
```

---

## 9. Final Status of Day 16

The feature-selection process can be summarized as:

```text
78 original features
        ↓
Remove 8 constant features
        ↓
70 candidate features
        ↓
Correlation analysis
        ↓
Remove 7 exact redundant features
        ↓
63 feature pool
        ↓
Class-aware ExtraTrees feature importance
        ↓
Create candidate sets
        ↓
63 / 50 / 40 / 30 / 20 features
```

Day 16 feature-selection work is completed.

The final feature count will be selected later based on experimental comparison rather than being fixed only from the initial proposal.
