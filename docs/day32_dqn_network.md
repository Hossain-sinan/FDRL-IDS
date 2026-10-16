# DAY 32 — DQN NETWORK

## Project

**Federated Deep Reinforcement Learning Intrusion Detection System (FDRL-IDS)**

## 1. Objective

The main objective of Day 32 was to implement the neural network used by the Deep Q-Network (DQN) agent.

The network takes the 20 selected network traffic features as input and produces four Q-values corresponding to the four actions defined for the IDS environment:

* `0` — BENIGN
* `1` — ATTACK
* `2` — BLOCK
* `3` — MONITOR

This network will later be trained using the DQN learning process.

---

## 2. Network Architecture

The implemented architecture is:

```text
20 Input Features
       ↓
Linear Layer: 20 → 256
       ↓
ReLU
       ↓
Linear Layer: 256 → 128
       ↓
ReLU
       ↓
Linear Layer: 128 → 64
       ↓
ReLU
       ↓
Linear Layer: 64 → 4
       ↓
4 Q-values
```

The architecture follows the initial design specified for the project.

The final layer produces four raw Q-values rather than probabilities.

---

## 3. Implementation

The DQN network was implemented using PyTorch.

The main class is:

```python
class DQNNetwork(nn.Module):
```

The network uses:

* `nn.Linear` for fully connected layers
* `nn.ReLU` for hidden-layer activation
* A final linear layer for the four Q-value outputs

The main source file is:

```text
src/day32_dqn_network.py
```

---

## 4. Input

The network expects a state containing 20 features.

For testing, one sample state was created with shape:

```text
torch.Size([1, 20])
```

This means:

* `1` = one sample
* `20` = twenty input features

---

## 5. Output

The network successfully produced an output with shape:

```text
torch.Size([1, 4])
```

The four outputs represent:

```text
Output 0 → BENIGN
Output 1 → ATTACK
Output 2 → BLOCK
Output 3 → MONITOR
```

These outputs are Q-values.

The network does not use Softmax because DQN requires raw Q-values rather than class probabilities.

---

## 6. Test Result

The network was executed successfully.

Test output included:

```text
[INPUT]
Shape: torch.Size([1, 20])

[OUTPUT Q-VALUES]
tensor([[ 0.0047, -0.1712, -0.0024,  0.1499]])

[OUTPUT SHAPE]
torch.Size([1, 4])

[SELECTED ACTION]
Action number: 3
Action name: MONITOR
```

The highest initial Q-value was:

```text
0.1499
```

which corresponded to action:

```text
3 — MONITOR
```

This does not mean that the model has learned that MONITOR is the correct action. The network is currently untrained, so the initial Q-values are based on randomly initialized weights.

---

## 7. Parameter Verification

The network contains trainable parameters in each fully connected layer.

The parameter counts are:

```text
20 → 256     = 5,376
256 → 128    = 32,896
128 → 64     = 8,256
64 → 4       = 260
```

Total:

```text
46,788 trainable parameters
```

This is a relatively small neural network and is appropriate as an initial DQN architecture for the project.

---

## 8. Relationship with Day 31

Day 31 implemented the IDS environment.

The relationship between the two components is:

```text
Day 31

Network State
     ↓
IDS Environment
     ↓
Reward
```

Day 32 adds the DQN network:

```text
Network State
     ↓
DQN Network
     ↓
4 Q-values
     ↓
Select Action
     ↓
IDS Environment
     ↓
Reward
```

The DQN network will therefore act as the decision-making component of the reinforcement learning system.

---

## 9. Challenges

### Challenge 1 — Understanding Q-values

The network output is not a probability distribution.

Instead, each output represents an estimated value for taking a particular action.

This was clarified by testing the four outputs and selecting the action with the highest Q-value.

### Challenge 2 — Understanding the untrained output

The initial Q-values do not represent a learned IDS decision because the network has not been trained yet.

The values are only useful for confirming that the network can accept the state and produce four outputs.

---

## 10. Accomplishments

* Implemented the DQN neural network using PyTorch.
* Implemented the proposed `20 → 256 → 128 → 64 → 4` architecture.
* Added ReLU activation functions to the hidden layers.
* Tested a 20-feature input state.
* Confirmed the output shape is `[1, 4]`.
* Confirmed all four IDS actions are represented.
* Tested automatic action selection using the highest Q-value.
* Verified the network contains 46,788 trainable parameters.

---

## 11. Day 32 Status

**Status: COMPLETE**

The DQN neural network is successfully implemented and tested.

The network is ready to be combined with the additional components required for DQN learning.

---

## 12. Next Step — Day 33

Day 33 will implement the main DQN supporting mechanisms:

```text
Experience Replay
Target Network
Epsilon-Greedy Exploration
```

The project proposal specifies:

```text
Replay Buffer Size = 10,000
Target Network Update = Every 100 Steps
```

These components will allow the network to learn from previous experiences while improving training stability.
