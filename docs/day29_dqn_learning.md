# DAY 29 — DQN LEARNING AND ARCHITECTURE

## Project

**Federated Deep Reinforcement Learning Intrusion Detection System (FDRL-IDS)**

---

## 1. Day 29 Objective

The main objective of Day 29 was to understand the **Deep Q-Network (DQN)** algorithm and design the DQN architecture that will later be used in the FDRL-IDS project.

DQN combines:

* Q-learning
* Neural networks
* Experience replay
* Target networks

The purpose of using DQN instead of a traditional Q-table is to handle the large and continuous state space created by network-flow features.

---

## 2. Why DQN Is Needed

In traditional Q-learning, a Q-table stores the estimated value of every state-action combination.

For the FDRL-IDS project, the state contains multiple network-flow features. The number of possible feature combinations can become very large.

Therefore, using a Q-table directly would not be practical.

DQN solves this problem by using a neural network to approximate the Q-values.

The basic idea is:

```text
Traditional Q-learning

State
  ↓
Q-table
  ↓
Q-values
  ↓
Action
```

DQN changes this to:

```text
DQN

State
  ↓
Neural Network
  ↓
Q-values
  ↓
Action
```

---

## 3. Relationship Between Q-Learning and DQN

The Q-learning idea learned on Day 27 is still used in DQN.

The traditional Q-learning update is:

$$
Q(s,a)=r+\gamma\max_{a'}Q(s',a')
$$

In DQN, the Q-value is approximated by a neural network:

$$
Q(s,a;\theta)
$$

where:

* \(s\) = current state
* \(a\) = selected action
* \(r\) = reward
* \(s'\) = next state
* \(\gamma\) = discount factor
* \(\theta\) = neural-network parameters

Therefore, DQN can be understood as:

```text
Q-learning
     +
Neural Network
     =
Deep Q-Network
```

---

## 4. DQN in the FDRL-IDS Project

The FDRL-IDS project uses the selected network-flow features as the state input.

The current feature-selection stage produced **20 selected features**.

Therefore, the DQN receives:

```text
State
 ↓
20 network features
```

The DQN then estimates a Q-value for each of the four proposed actions.

The four actions are:

```text
0 = BENIGN
1 = ATTACK
2 = BLOCK
3 = MONITOR
```

Therefore, the output layer contains four values:

```text
Q(BENIGN)
Q(ATTACK)
Q(BLOCK)
Q(MONITOR)
```

The action with the highest Q-value can be selected during exploitation.

---

## 5. Proposed DQN Architecture

The architecture follows the current proposal design:

```text
20 → 256 → 128 → 64 → 4
```

Detailed structure:

```text
Input Layer
20 features
     ↓
Hidden Layer 1
256 neurons
ReLU
     ↓
Hidden Layer 2
128 neurons
ReLU
     ↓
Hidden Layer 3
64 neurons
ReLU
     ↓
Output Layer
4 Q-values
```

The four output values correspond to:

```text
Q(BENIGN)
Q(ATTACK)
Q(BLOCK)
Q(MONITOR)
```

This architecture is treated as the **initial proposal architecture**. It will not be assumed to be optimal without experimental evidence.

---

## 6. DQN Output Example

Suppose the network receives one network-flow state.

The DQN may produce:

```text
Q(BENIGN)  = 2.4
Q(ATTACK)  = 7.1
Q(BLOCK)   = 3.2
Q(MONITOR) = 1.8
```

The largest Q-value is:

```text
Q(ATTACK) = 7.1
```

Therefore, the selected action would be:

```text
ATTACK
```

The output values are **Q-values, not probabilities**.

---

## 7. Experience Replay

DQN uses an experience replay buffer.

Each experience contains:

```text
(state, action, reward, next_state, done)
```

For example:

```text
Current state
      ↓
Select action
      ↓
Receive reward
      ↓
Move to next state
      ↓
Store experience
```

The replay buffer allows previous experiences to be sampled randomly during training.

The proposal specifies:

```text
Replay buffer size = 10,000
```

Experience replay is intended to reduce correlation between consecutive training samples and improve learning stability.

---

## 8. Target Network

DQN uses two neural networks:

```text
Online Network
Target Network
```

The online network is updated during learning.

The target network provides a more stable Q-value target.

The proposal specifies:

```text
Target-network update interval = 100 steps
```

This component will be implemented later during the DQN implementation stage.

---

## 9. Epsilon-Greedy Exploration

During DQN training, the agent needs to balance:

```text
Exploration
```

and

```text
Exploitation
```

Exploration means trying different actions.

Exploitation means selecting the action currently estimated to have the highest Q-value.

Epsilon-greedy action selection provides this balance.

Conceptually:

```text
High epsilon
     ↓
More exploration

Lower epsilon
     ↓
More exploitation
```

The exact epsilon schedule will be implemented and tested later.

---

## 10. DQN Architecture Diagram

The planned architecture is:

```text
                 FDRL-IDS DQN

              Network Flow State
                      │
                      ▼
               20 Input Features
                      │
                      ▼
              ┌───────────────┐
              │  256 Neurons  │
              │     ReLU      │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │  128 Neurons  │
              │     ReLU      │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │   64 Neurons  │
              │     ReLU      │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │  4 Q-values   │
              └───────┬───────┘
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼              ▼
    BENIGN         ATTACK          BLOCK         MONITOR
```

---

## 11. Difference Between Dataset Labels and DQN Actions

An important distinction was identified during Day 29.

The CICIDS2017 ground-truth labels describe the type of network traffic, such as:

```text
BENIGN
DoS
DDoS
Brute Force
Web Attack
Infiltration
Botnet
Port Scan
```

These are **traffic labels**.

The DQN actions are:

```text
BENIGN
ATTACK
BLOCK
MONITOR
```

These represent **agent decisions**.

Therefore:

```text
Ground Truth
     ≠
DQN Action
```

This distinction is important for designing the environment and reward mechanism on Day 30.

---

## 12. Complete DQN Learning Flow

The planned DQN learning process is:

```text
Network State
     ↓
DQN Network
     ↓
Q-values
     ↓
Epsilon-Greedy Action Selection
     ↓
Action
     ↓
Environment
     ↓
Reward + Next State
     ↓
Experience Replay Buffer
     ↓
Sample Mini-batch
     ↓
DQN Training
     ↓
Update Online Network
     ↓
Periodically Update Target Network
```

---

## 13. Day 29 Outcome

The following concepts were studied and connected to the FDRL-IDS project:

* DQN
* Q-value approximation
* Neural-network-based Q-learning
* Experience replay
* Target network
* Epsilon-greedy exploration
* DQN input and output
* Four proposed IDS actions
* Proposed DQN architecture

The initial DQN architecture was defined as:

```text
20 → 256 → 128 → 64 → 4
```

---

## 14. Day 29 Conclusion

Day 29 established the DQN architecture and explained how it will be used as the decision-making component of the FDRL-IDS.

The next step is not to train the DQN immediately.

First, the IDS environment must be formally defined.

On Day 30, the following must be specified:

* State
* Action
* Timestep
* Episode
* Ground truth
* Reward calculation
* Transition to the next state
* Episode termination

This design must be clear before DQN implementation begins.
