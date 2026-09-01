# Day 01 — Project Understanding

## 1. Project Name

**FDRL-IDS — Federated Deep Reinforcement Learning Intrusion Detection System**

## 2. What is an IDS?

An Intrusion Detection System (IDS) monitors network traffic and detects suspicious or malicious activity.

In this project, the IDS will analyze network traffic and make security-related decisions.

## 3. What is Machine Learning?

Machine Learning allows a computer to learn patterns from data and use those patterns to make predictions or decisions.

In this project, machine learning will be used to learn patterns from network traffic.

## 4. What is Federated Learning?

Federated Learning is a machine learning approach where multiple clients collaboratively train a model without directly sharing their raw data.

In this project, different parts of the CICIDS2017 dataset will be used to simulate different federated clients.

## 5. What is Deep Reinforcement Learning?

Deep Reinforcement Learning combines deep learning with reinforcement learning.

An agent learns by observing a state, taking an action, receiving a reward or penalty, and improving its future decisions.

## 6. What is DQN?

DQN (Deep Q-Network) is a reinforcement learning algorithm that uses a neural network to estimate the value of possible actions.

The DQN agent in this project will learn which security action is appropriate for network traffic.

## 7. What is CICIDS2017?

CICIDS2017 is a network intrusion detection dataset containing benign and malicious network traffic.

It will be used as the main dataset for this project.

## 8. Planned DRL Actions

| Action | Meaning |
| ------ | ------- |
| 0      | BENIGN  |
| 1      | ATTACK  |
| 2      | BLOCK   |
| 3      | MONITOR |

## 9. Planned Reward Function

| Situation           | Reward |
| ------------------- | -----: |
| True Positive (TP)  |    +10 |
| True Negative (TN)  |     +2 |
| False Positive (FP) |     -5 |
| False Negative (FN) |    -20 |

## 10. Overall Project Pipeline

```text
CICIDS2017
     ↓
Data Preprocessing
     ↓
Federated Client Simulation
     ↓
Local Training
     ↓
Federated Aggregation
     ↓
Global Model
     ↓
DRL / DQN Agent
     ↓
Security Decision
     ↓
Evaluation
```

## 11. My Two-Minute Explanation

FDRL-IDS is an Intrusion Detection System project that combines Federated Learning and Deep Reinforcement Learning. CICIDS2017 will provide network traffic containing benign and attack examples. The dataset will be divided to simulate multiple clients, representing different organizations or network environments. These clients will train collaboratively using Federated Learning without directly sharing their raw data. A DQN-based Deep Reinforcement Learning agent will then learn security decisions such as BENIGN, ATTACK, BLOCK, and MONITOR. The final system will be evaluated using appropriate intrusion detection metrics.

## 12. Day 1 Learning Checklist

* [ ] Understand IDS
* [ ] Understand Machine Learning
* [ ] Understand Federated Learning
* [ ] Understand Deep Reinforcement Learning
* [ ] Understand DQN
* [ ] Understand CICIDS2017
* [ ] Understand the overall FDRL-IDS pipeline
* [ ] Create project folders
* [ ] Create README.md
* [ ] Initialize Git
* [ ] Make first Git commit
* [ ] Explain the project in approximately 2 minutes
