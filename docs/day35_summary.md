# Day 35 — Improved DQN Model Validation

## 1. Introduction

Today, I worked on testing the Improved DQN model that I trained on Day 34.

The main purpose of this work was to check how well my trained model can identify normal network traffic and attack traffic.

For this test, I used 1,000 samples from the CICIDS2017 dataset. These samples were not used during the Day 34 model training.

After running the test, I checked the model's performance and saved all the results in a JSON file.

## 2. What I Did Today

First, I loaded the trained Improved DQN model from the `models` folder. Then I loaded the saved scaler and model information.

After that, I prepared the validation data from the CICIDS2017 dataset.

I followed these steps:

1. Loaded the CICIDS2017 dataset.
2. Selected 500 BENIGN samples.
3. Selected 500 DDoS attack samples.
4. Excluded the 1,000 samples that I used for Day 34 training.
5. Combined the selected samples to make a validation dataset of 1,000 rows.
6. Selected the same 20 features that I used during training.
7. Used the saved MinMaxScaler to scale the validation data.
8. Passed the validation data to the trained DQN model.
9. Collected the model's predictions.
10. Calculated the performance results.

I used the saved scaler because the validation data must be processed in the same way as the training data.

## 3. Validation Dataset Information

I used 1,000 samples for this test.

| Traffic Type | Number of Samples |
|---|---:|
| BENIGN | 500 |
| DDoS Attack | 500 |
| **Total** | **1,000** |

The training samples were excluded from the validation data. This helped me test the model using different rows from the same dataset.

## 4. Model Performance Results

After running the validation script, I got the following results:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 98.40% |
| Precision | 96.90% |
| Recall | 100.00% |
| F1-score | 98.43% |

### What These Results Mean

- **Accuracy (98.40%):** The model correctly classified 984 out of 1,000 samples.
- **Precision (96.90%):** When the model predicted an attack, about 96.90% of those predictions were actually attacks.
- **Recall (100%):** The model detected all 500 attack samples. It did not miss any attack in this test.
- **F1-score (98.43%):** This score gives a balance between precision and recall.

These results show that the model performed well on this validation dataset.

## 5. Understanding the Confusion Matrix

The confusion matrix shows how many samples were correctly and incorrectly classified.

My result was:

```text
[[484  16]
 [  0 500]]
```

Here is the result in table format:

| Actual Traffic | Predicted BENIGN | Predicted ATTACK |
|---|---:|---:|
| BENIGN | 484 | 16 |
| ATTACK | 0 | 500 |

From this result:

- 484 normal traffic samples were correctly identified as BENIGN.
- 16 normal traffic samples were incorrectly identified as attacks.
- All 500 attack samples were correctly identified.
- No attack sample was incorrectly identified as BENIGN.

So, the model detected all attacks in this test, but it also gave 16 false alarms for normal traffic.

## 6. DQN Action Selection Results

In my project, the DQN environment has four possible actions:

| Action ID | Action Name | Meaning |
|---|---|---|
| 0 | BENIGN | Treat the traffic as normal |
| 1 | ATTACK | Identify the traffic as an attack |
| 2 | BLOCK | Block the traffic |
| 3 | MONITOR | Continue monitoring the traffic |

After checking the model's predictions, I got these results:

| Action Name | Selected Count |
|---|---:|
| BENIGN | 484 |
| ATTACK | 516 |
| BLOCK | 0 |
| MONITOR | 0 |
| **Total** | **1,000** |

One important thing I noticed is that the model selected only BENIGN and ATTACK actions. It did not select BLOCK or MONITOR.

This means that the model can perform binary attack detection in this test, but its ability to use all four actions is not working as expected yet.

I need to investigate this in the next stages of the project.

## 7. Problems I Faced and How I Fixed Them

### Problem 1: Incorrect action counts in the JSON file

After the first run, the Python terminal showed the correct action counts. However, the JSON file showed zero for all four actions.

I checked the code and found a small problem. The action IDs in my dictionary were stored as strings, but the predicted action IDs were integers.

**Solution:** I converted the dictionary keys into integers before comparing them with the predicted actions.

After fixing the code and running the script again, the JSON file showed the correct results.

### Problem 2: Understanding the four DQN actions

I noticed that the model selected BENIGN and ATTACK but did not select BLOCK or MONITOR.

This is an important issue because my project is designed with four possible actions.

I recorded this problem so that I can investigate the reward system and training process later.

## 8. Files Used and Created

| File | Purpose |
|---|---|
| `src/day35_validate_dqn.py` | Runs the validation test |
| `models/day34_improved_dqn_checkpoint.pt` | Saved trained DQN model |
| `models/day34_improved_scaler.joblib` | Saved scaler used during training |
| `models/day34_improved_metadata.json` | Stores model and training information |
| `docs/day35_improved_validation_results.json` | Stores the validation results |
| `docs/day35_summary.md` | Contains today's work summary |

## 9. Limitations of Today's Work

Although the validation results are good, there are some limitations.

1. I used different rows from the same CICIDS2017 Friday DDoS capture. I have not yet tested the model on a different capture day.
2. The model selected only two actions out of four.
3. The DQN training environment uses shuffled dataset records as the next states. This is a simplified environment, not a real network traffic sequence.
4. The training loss was not stable, so I cannot say that the model has fully converged.
5. These results are from an initial validation test. More testing is needed before making a final conclusion about the model.

## 10. Next Tasks

In the next stages, I need to:

1. Find out why the model does not select BLOCK and MONITOR.
2. Review the reward system for all four actions.
3. Improve the DQN training environment.
4. Test the model using data from a different CICIDS2017 capture.
5. Compare the Improved DQN results with the previous DQN model.

## 11. Conclusion

Today, I successfully completed the initial validation of my Improved DQN model.

The model achieved 98.40% accuracy, 100% recall, and 98.43% F1-score on 1,000 validation samples.

It detected all 500 attack samples, but incorrectly classified 16 normal samples as attacks.

I also found that the model did not select the BLOCK and MONITOR actions. This is an important issue that I need to solve in the next stages.

Finally, I saved the validation results in a JSON file for future analysis and comparison.

**Day 35 completed successfully.**
