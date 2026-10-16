## Day 34 — DQN Training Experiment

## 1. Objective

The main objective of Day 34 was to run a small training experiment using the Deep Q-Network (DQN) for the FDRL-IDS project. I tested the training data, IDS environment, DQN network, replay buffer, loss calculation, and network weight updates.

## 2. Work Completed

- Loaded CICIDS2017 data for the experiment.
- Selected 1,000 samples containing 500 BENIGN and 500 DDoS records.
- Prepared 20 input features and scaled the feature values between 0 and 1.
- Connected the training data with the IDS environment.
- Tested the DQN network and confirmed that it produced four Q-values for each state.
- Stored and sampled experiences using the replay buffer.
- Collected experiences using the epsilon-greedy policy.
- Calculated target Q-values, current Q-values, and DQN loss.
- Updated the online network weights using the optimizer.
- Performed 10 additional training updates and recorded the loss values.
- Tested the target-network update mechanism.

## 3. Results

| Item                  |   Result |
| --------------------- | -------: |
| Training samples      |    1,000 |
| Selected features     |       20 |
| BENIGN samples        |      500 |
| DDoS samples          |      500 |
| Collected experiences |       51 |
| Training updates      |       10 |
| First training loss   | 107.5330 |
| Last training loss    |  92.5920 |
| Final epsilon         |   0.7783 |

The program completed all the implemented steps without a runtime error. The loss values changed during training. The final loss was lower than the first loss, but the loss did not decrease in every step.

## 4. Problems and Solutions

### Python indentation error

The target-network update code initially had incorrect indentation. I corrected the indentation and ran the program again successfully.

### Target-network update test

The target network was tested separately at training step 100. The actual training loop performed only 10 updates, so the target-network update interval was not reached during that loop.

## 5. Time Allocation

The exact time spent on each activity was not recorded. Most of the work involved implementing the training process, fixing the code, running the experiment, and checking the output.

## 6. Pending Work

- **High priority:** Validate the DQN agent's decisions and calculate detection metrics.
- **High priority:** Check whether the four DQN actions and reward system behave as intended.
- **Medium priority:** Run further experiments after the initial validation.
- **Medium priority:** Improve the training process if the evaluation results show problems.

## 7. Key Takeaways

I learned how a DQN training loop samples experiences, calculates target and current Q-values, calculates loss, and updates network weights. I also tested the target-network update mechanism.

This experiment confirms that the basic training code runs. However, it does not establish the final detection performance of the IDS. More validation is needed before drawing conclusions about its accuracy.

## 8. Conclusion

Day 34's initial DQN training experiment was completed successfully. The next step is to validate the agent's predictions and evaluate its behavior using appropriate metrics.
