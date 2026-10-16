# DAY 31 — BUILD DQN IDS ENVIRONMENT

## Project

**Federated Deep Reinforcement Learning Intrusion Detection System (FDRL-IDS)**

---

## 1. Day 31 Objective

The main objective of Day 31 was to build the first working **IDS environment** for the DQN-based intrusion detection system.

On Day 30, the environment design was defined. On Day 31, that design was implemented as a Python class and tested using a small sample dataset.

The main interaction tested was:

```text
State
  ↓
Action
  ↓
Ground Truth
  ↓
Reward
  ↓
Next State
```

---

## 2. IDS Environment Concept

In reinforcement learning, the **environment** represents the system in which the agent operates.

For this project, the environment represents network traffic.

The DQN agent will later observe a network traffic state and select an action.

The environment then checks the actual ground-truth label and calculates the reward.

The basic process is:

```text
Network Traffic
      ↓
     State
      ↓
 DQN Agent
      ↓
    Action
      ↓
 Environment
      ↓
 Ground Truth
      ↓
    Reward
      ↓
 Next State
```

---

## 3. State

The state represents the features of one network traffic sample.

For the FDRL-IDS project, the state will eventually contain the selected features from the CICIDS2017 dataset.

The project currently uses **20 selected network features** from the previous feature-selection work.

For the Day 31 test, a small artificial dataset was used instead of the full CICIDS2017 dataset.

Example test state:

```text
[0.1, 0.2, 0.3]
```

The purpose of using a small dataset was to verify the environment logic before connecting it to the complete dataset and DQN model.

---

## 4. Actions

The environment supports the four actions defined in the project design.

| Action ID | Action  |
| --------: | ------- |
|         0 | BENIGN  |
|         1 | ATTACK  |
|         2 | BLOCK   |
|         3 | MONITOR |

The DQN agent will later select one of these four actions.

---

## 5. Ground Truth

The ground truth represents the actual class of the network traffic sample.

For the initial environment implementation:

```text
0 = BENIGN
1 = ATTACK
```

The four actions are converted into a binary security decision for the initial reward calculation.

The action:

```text
BENIGN
```

represents a benign decision.

The actions:

```text
ATTACK
BLOCK
MONITOR
```

represent attack-related responses.

---

## 6. Reward System

The reward system follows the reward values defined in the project design.

| Situation      | Meaning                                       | Reward |
| -------------- | --------------------------------------------- | -----: |
| True Negative  | BENIGN traffic correctly identified as BENIGN |     +2 |
| True Positive  | ATTACK traffic correctly handled as an attack |    +10 |
| False Positive | BENIGN traffic treated as an attack           |     -5 |
| False Negative | ATTACK traffic treated as BENIGN              |    -20 |

The reward encourages the agent to make correct security decisions.

A particularly strong penalty is given to false negatives because missing an actual attack is an important problem in intrusion detection.

---

## 7. Environment Class

A Python environment class named:

```python
IDSEnvironment
```

was created.

The class contains the main reinforcement-learning environment operations.

### Main components

```text
__init__()
reset()
step()
```

---

## 8. Initialization

The constructor receives:

```python
states
labels
```

The states contain network features, while the labels contain the ground-truth classes.

The environment also checks that:

```text
Number of states = Number of labels
```

An empty dataset is rejected.

This helps prevent invalid environment configuration.

---

## 9. Reset Function

The `reset()` function starts the environment from the first sample.

Conceptually:

```text
reset()
  ↓
Sample 0
  ↓
Initial State
```

This allows a new episode to begin from a known starting point.

---

## 10. Step Function

The `step()` function performs one interaction between the agent and the environment.

The agent provides an action:

```python
env.step(action)
```

The environment then:

1. Checks whether the action is valid.
2. Gets the ground-truth label.
3. Compares the action with the ground truth.
4. Calculates the reward.
5. Moves to the next sample.
6. Checks whether the episode has finished.
7. Returns the next state.

The function returns:

```text
next_state
reward
done
info
```

---

## 11. Test Dataset

A small three-sample dataset was created for testing.

### States

```text
[0.1, 0.2, 0.3]
[0.4, 0.5, 0.6]
[0.7, 0.8, 0.9]
```

### Labels

```text
0 = BENIGN
1 = ATTACK
0 = BENIGN
```

This small dataset was sufficient to test different reward situations.

---

## 12. Test Results

The environment test was successfully executed.

### Step 1

```text
Ground truth = BENIGN
Action = BENIGN
Result = TN
Reward = +2
```

The environment correctly identified the benign traffic.

### Step 2

```text
Ground truth = ATTACK
Action = ATTACK
Result = TP
Reward = +10
```

The environment correctly handled the attack traffic.

### Step 3

```text
Ground truth = BENIGN
Action = ATTACK
Result = FP
Reward = -5
```

The environment correctly identified this as a false positive.

After the third sample, the environment returned:

```text
Done = True
```

This confirmed that episode termination was also working correctly.

---

## 13. Actual Test Output

The test produced the following results:

```text
============================================================
DAY 31 — IDS ENVIRONMENT TEST
============================================================

Initial state:
[0.1 0.2 0.3]

Step 1
Action: 0
Ground truth: 0
Result: TN
Reward: 2
Done: False

Step 2
Action: 1
Ground truth: 1
Result: TP
Reward: 10
Done: False

Step 3
Action: 1
Ground truth: 0
Result: FP
Reward: -5
Done: True

Environment test completed.
```

The test completed successfully without errors.

---

## 14. Files Created

The following files were created for Day 31:

```text
src/
└── day31_ids_environment.py

tests/
└── test_day31_environment.py

docs/
└── day31_environment.md
```

---

## 15. Important Learning

The main concept learned on Day 31 was how an RL environment interacts with an agent.

The complete process is:

```text
             ┌──────────────┐
             │    State     │
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │    Action    │
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │ Environment  │
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │    Reward    │
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │ Next State   │
             └──────────────┘
```

This interaction is the foundation for the DQN training process.

---

## 16. Why a Small Test Was Used

The complete CICIDS2017 dataset contains millions of records.

Using the complete dataset immediately would make debugging more difficult.

Therefore, a small artificial dataset was first used to verify that:

* State handling works.
* Actions are accepted.
* Rewards are calculated correctly.
* Next states are returned.
* Episode termination works.

After the environment logic is verified, it can be connected to the actual project data.

---

## 17. Day 31 Result

The Day 31 objective was successfully completed.

The FDRL-IDS project now has a working basic IDS reinforcement-learning environment.

The environment successfully performs:

```text
State
  ↓
Action
  ↓
Ground Truth
  ↓
Reward
  ↓
Next State
```

The environment is now ready to be connected with the DQN neural network.

---

## 18. Next Step

The next task is **Day 32 — Build DQN Network**.

The proposed initial network architecture is:

```text
20 Input Features
       ↓
   Dense Layer
      256
       ↓
   Dense Layer
      128
       ↓
   Dense Layer
       64
       ↓
   Output Layer
       4
       ↓
Q-values for:
BENIGN
ATTACK
BLOCK
MONITOR
```

Day 32 will focus only on building and testing this neural network.

---

## Conclusion

Day 31 successfully implemented and tested the first working IDS environment for the FDRL-IDS project.

The environment can receive a state, process an action, calculate the corresponding reward, return the next state, and identify when the episode is complete.

This provides the required foundation for implementing the DQN model in the next stage.
