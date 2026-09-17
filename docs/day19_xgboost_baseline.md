# Day 19 — XGBoost Baseline Learning and Experiment

## 1. Day 19 Goal

The main goal of Day 19 was to learn the basic concepts of **XGBoost (Extreme Gradient Boosting)** and perform a small classification experiment before applying machine learning to the CICIDS2017 intrusion detection dataset.

The experiment was intentionally performed on a small dataset so that I could understand the model, hyperparameters, evaluation metrics, and feature importance without using the large CICIDS2017 dataset at this stage.

---

## 2. What I Learned

### 2.1 Decision Tree

A decision tree makes predictions by dividing the data using conditions on features.

For example:

```text
             Feature?
             /       \
           Yes        No
           /           \
       Class A       Class B
```

Important decision tree concepts learned:

* Gini impurity
* Entropy
* Information gain
* Tree depth
* Overfitting
* Pruning

A deeper tree can learn more complex patterns, but if it becomes too deep, it can memorize the training data and overfit.

---

## 3. Boosting

Boosting is an ensemble machine learning technique.

Instead of creating one very strong model directly, boosting builds several weak models sequentially.

The general idea is:

```text
Weak Learner 1
      ↓
Find errors
      ↓
Weak Learner 2
      ↓
Find remaining errors
      ↓
Weak Learner 3
      ↓
Final combined model
```

I learned about:

* AdaBoost
* Gradient Boosting
* Sequential weak learners
* Bias and variance
* Overfitting in boosting

---

## 4. What is XGBoost?

XGBoost stands for **Extreme Gradient Boosting**.

It is an optimized implementation of gradient-boosted decision trees.

XGBoost builds trees sequentially and tries to improve the errors made by previous trees.

The main concepts I studied were:

### L1 and L2 Regularization

Regularization helps control model complexity and can reduce overfitting.

### Second-Order Optimization

XGBoost uses both gradient and second-order information during optimization.

### Missing Values

XGBoost can handle missing values using learned split directions.

### Row and Column Subsampling

XGBoost can use only part of the training rows and features when building trees.

In this experiment:

```text
subsample = 0.8
colsample_bytree = 0.8
```

This means the model used a fraction of the available rows and columns during tree construction.

### Parallel and Efficient Training

XGBoost is designed to train decision trees efficiently and can use optimized computational methods.

---

# 5. Classification with XGBoost

For this experiment, I used XGBoost multiclass classification.

The model was configured with:

```text
objective = multi:softprob
num_class = 3
eval_metric = mlogloss
```

`multi:softprob` produces probability values for the different classes.

`mlogloss` means multiclass logarithmic loss. It evaluates how well the predicted probabilities match the actual classes.

---

# 6. Dataset Used

For Day 19, I used the **Iris dataset** from Scikit-learn.

The dataset contains:

* 150 samples
* 4 features
* 3 classes

The features are:

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

The classes are:

* Setosa
* Versicolor
* Virginica

The dataset was selected because it is small and easy to understand.

This was important because the purpose of Day 19 was to learn XGBoost rather than immediately run a large experiment on CICIDS2017.

---

# 7. Train, Validation and Test Split

I used a fixed random seed:

```text
random_state = 42
```

The dataset was divided into:

| Dataset    | Samples | Percentage |
| ---------- | ------: | ---------: |
| Training   |      90 |        60% |
| Validation |      30 |        20% |
| Test       |      30 |        20% |
| Total      |     150 |       100% |

Stratified splitting was used so that the class distribution was maintained across the different sets.

---

# 8. Hyperparameter Experiments

I performed five controlled experiments.

| Experiment | max_depth | learning_rate | n_estimators | subsample | colsample_bytree |
| ---------- | --------: | ------------: | -----------: | --------: | ---------------: |
| A          |         2 |           0.1 |           20 |       0.8 |              0.8 |
| B          |         3 |           0.1 |           20 |       0.8 |              0.8 |
| C          |         4 |           0.1 |           20 |       0.8 |              0.8 |
| D          |         3 |           0.3 |           20 |       0.8 |              0.8 |
| E          |         3 |           0.1 |           40 |       0.8 |              0.8 |

The experiments were designed to understand the effect of:

* Tree depth
* Learning rate
* Number of trees
* Row subsampling
* Column subsampling

---

# 9. Validation Results

The validation results were:

| Experiment | Validation Accuracy | Validation Macro F1 |
| ---------- | ------------------: | ------------------: |
| A          |              0.9000 |              0.8997 |
| B          |              0.9333 |              0.9333 |
| C          |              0.9333 |              0.9333 |
| D          |              0.9333 |              0.9333 |
| E          |              0.9333 |              0.9333 |

Experiments B, C, D and E produced the same validation scores.

Therefore, there was no clear performance difference between these four configurations on this small validation set.

The program selected **Experiment B** because it was the first configuration among the tied highest Macro F1 results.

Selected configuration:

```text
max_depth = 3
learning_rate = 0.1
n_estimators = 20
subsample = 0.8
colsample_bytree = 0.8
```

I do not consider this configuration universally better than C, D or E because the validation results were tied.

---

# 10. Final Test Results

The selected XGBoost configuration was evaluated on the separate test set.

Results:

| Metric          | Result |
| --------------- | -----: |
| Accuracy        | 0.9333 |
| Macro Precision | 0.9333 |
| Macro Recall    | 0.9333 |
| Macro F1        | 0.9333 |

The model correctly classified:

```text
28 out of 30 test samples
```

---

# 11. Classification Report

```text
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.90      0.90      0.90        10
   virginica       0.90      0.90      0.90        10

    accuracy                           0.93        30
   macro avg       0.93      0.93      0.93        30
weighted avg       0.93      0.93      0.93        30
```

The Setosa class was classified correctly for all 10 test samples.

There was one error between Versicolor and Virginica in each direction.

---

# 12. Confusion Matrix

The confusion matrix was:

```text
[[10  0  0]
 [ 0  9  1]
 [ 0  1  9]]
```

It can be interpreted as:

| Actual / Predicted | Setosa | Versicolor | Virginica |
| ------------------ | -----: | ---------: | --------: |
| Setosa             |     10 |          0 |         0 |
| Versicolor         |      0 |          9 |         1 |
| Virginica          |      0 |          1 |         9 |

The model correctly classified all Setosa samples.

The two classification errors occurred between Versicolor and Virginica.

---

# 13. Feature Importance

I also checked XGBoost feature importance using three methods:

* Gain
* Weight
* Cover

## 13.1 Gain

The Gain values were:

| Feature      |     Gain |
| ------------ | -------: |
| Petal width  | 9.025739 |
| Petal length | 5.805940 |
| Sepal width  | 1.232997 |
| Sepal length | 0.509089 |

In this trained model, petal width and petal length contributed more to the tree splits than the sepal features according to Gain.

---

## 13.2 Weight

The Weight values were:

| Feature      | Weight |
| ------------ | -----: |
| Sepal length |     14 |
| Sepal width  |     10 |
| Petal length |     68 |
| Petal width  |     34 |

Weight represents how frequently a feature was used for tree splitting.

Petal length was used most frequently in the trained model.

---

## 13.3 Cover

The Cover values were:

| Feature      |   Cover |
| ------------ | ------: |
| Sepal length |  7.5040 |
| Sepal width  | 15.1211 |
| Petal length | 17.9931 |
| Petal width  | 23.6253 |

Cover gives information about the amount of training data associated with the splits involving each feature.

Petal width had the highest Cover value in this experiment.

---

# 14. Important Observation About the Results

The Iris dataset is very small.

Only 30 samples were used for validation and 30 samples were used for testing.

Because of this, small changes in predictions can produce relatively large changes in the reported metrics.

For example, changing only one or two predictions can noticeably change the accuracy.

Therefore, the 93.33% test accuracy should not be treated as a strong general conclusion about XGBoost.

The main purpose of this experiment was to understand the algorithm and establish a working baseline procedure.

---

# 15. Beginner Problems and How I Handled Them

## Problem 1 — XGBoost was not available in the system Python

Initially, the script was run without the virtual environment active.

This produced:

```text
ModuleNotFoundError: No module named 'xgboost'
```

The problem was not with the script. XGBoost was installed inside the project virtual environment.

After activating:

```text
(.venv)
```

the script ran successfully.

The installed versions were:

```text
Python:     3.13.5
XGBoost:    3.4.1
Scikit-learn: 1.9.1
Pandas:     3.0.5
NumPy:      2.5.3
```

This taught me that I must check the active Python environment before running the project.

---

## Problem 2 — DMatrix Confusion

XGBoost has a native data structure called `DMatrix`.

However, I used:

```python
XGBClassifier
```

from the Scikit-learn compatible API.

Therefore, I did not need to manually create a `DMatrix`.

The Scikit-learn API accepted the NumPy arrays directly.

---

## Problem 3 — Scaling

Unlike many algorithms such as KNN or some neural networks, tree-based XGBoost does not normally require feature scaling for this type of experiment.

Therefore, no StandardScaler or MinMaxScaler was used for the Iris XGBoost experiment.

This is different from the preprocessing pipeline used in my FDRL-IDS project, where scaling is required for the later machine learning and reinforcement learning stages.

---

## Problem 4 — Class Imbalance

The Iris dataset is relatively balanced, with 50 samples per class.

However, this is very different from CICIDS2017.

My actual IDS dataset has highly imbalanced attack classes, including very rare classes.

Therefore, accuracy alone will not be enough for the actual FDRL-IDS evaluation.

For the future IDS experiments, I need to report:

* Precision
* Recall
* F1-score
* Macro F1
* Per-class metrics
* Confusion matrix

---

# 16. Why I Did Not Use CICIDS2017 for Day 19

The CICIDS2017 dataset is much larger and more complex than Iris.

My processed pipeline contains more than 2.5 million records and 20 selected features.

Using the full dataset immediately would make it harder for me to understand:

* How XGBoost works
* How hyperparameters affect the model
* How validation should be performed
* How feature importance works
* How multiclass evaluation works

Therefore, I first used a small controlled dataset.

The actual CICIDS2017 XGBoost baseline will be performed later when the project plan reaches the appropriate stage.

---

# 17. Connection With FDRL-IDS

This Day 19 experiment is important for the FDRL-IDS project because XGBoost will provide a **traditional machine learning baseline**.

The future comparison can help answer whether the proposed federated deep reinforcement learning approach provides useful differences compared with a standard supervised machine learning model.

The comparison should be performed using the same relevant dataset partitions and appropriate evaluation metrics.

The Day 19 experiment itself is only a learning experiment and is not yet the final CICIDS2017 baseline.

---

# 18. Files Created

The following files were created:

```text
src/
└── day19_xgboost_baseline.py

docs/
└── day19_xgboost_experiments.csv
```

The CSV file contains the hyperparameter configurations and validation results for the five experiments.

---

# 19. Day 19 Final Learning Outcome

After completing Day 19, I learned the basic working concepts of XGBoost and successfully trained a multiclass XGBoost classifier.

I learned how to:

* Load a classification dataset
* Split data into train, validation and test sets
* Configure XGBoost
* Change important hyperparameters
* Train an XGBoost classifier
* Make predictions
* Calculate Accuracy
* Calculate Precision
* Calculate Recall
* Calculate Macro F1
* Generate a classification report
* Generate a confusion matrix
* Analyze Gain, Weight and Cover feature importance
* Save experiment results
* Understand why small datasets can limit the reliability of performance comparisons

The experiment successfully demonstrated the complete basic workflow needed before applying XGBoost to the actual intrusion detection dataset.

---

# 20. Conclusion

Day 19 was completed successfully.

The XGBoost model achieved **93.33% accuracy and 93.33% Macro F1 on the 30-sample Iris test set**.

However, these results should only be considered as a learning baseline because the dataset is very small.

The most important outcome of Day 19 was understanding how XGBoost works and how to perform a controlled machine learning experiment.

The next steps will move toward the machine learning baseline stage of the FDRL-IDS project and eventually apply the learned concepts to the CICIDS2017 data.

**Day 19 Status: COMPLETED**
