# FDRL-IDS

## Federated Deep Reinforcement Learning Intrusion Detection System

### Project Overview

FDRL-IDS is a research project that aims to develop an Intrusion Detection System using Federated Learning and Deep Reinforcement Learning.

The project uses the CICIDS2017 dataset as network traffic data. Multiple simulated clients will represent different organizations or network environments. Federated Learning will allow the clients to collaboratively train a model without directly sharing their raw data.

A Deep Reinforcement Learning approach, particularly a Deep Q-Network (DQN), will be explored for making intelligent intrusion detection and response decisions.

### Main Technologies

* Python
* Machine Learning
* Federated Learning
* Deep Reinforcement Learning
* Deep Q-Network (DQN)
* CICIDS2017
* PyTorch
* Scikit-learn

### Planned Actions

The DRL agent will consider four possible actions:

1. BENIGN
2. ATTACK
3. BLOCK
4. MONITOR

### Project Structure

```text
FDRL-IDS/
├── data/
├── src/
├── notebooks/
├── models/
├── results/
├── experiments/
├── tests/
├── docs/
└── README.md
```

### Project Goal

The main goal is to investigate whether combining Federated Learning with Deep Reinforcement Learning can provide an effective and privacy-aware approach for network intrusion detection.
