# DAY 28 — NEURAL NETWORK LEARNING AND PYTORCH IMPLEMENTATION

## Project

**Federated Deep Reinforcement Learning Intrusion Detection System (FDRL-IDS)**

## Phase

**Phase 5 — DQN / Deep Reinforcement Learning IDS**

## Day 28 Objective

The main objective of Day 28 was to understand the basic components of a neural network and build a small PyTorch neural network that can later be used as the Q-network of the DQN-based IDS.

The work focused on:

* Input layer
* Hidden layers
* Activation function
* Output layer
* Loss function
* Optimizer
* Backpropagation
* Basic Q-value generation

No CICIDS2017 training was performed on Day 28. The purpose was to verify that the neural-network component works independently before connecting it to the DQN and IDS environment.

---

## 1. Neural Network Concept

A neural network receives input data, processes it through hidden layers, and produces an output.

For the FDRL-IDS project, the future DQN will receive the selected network features as its state.

The basic flow is:

```text
Network State
     ↓
Neural Network
     ↓
Q-values
     ↓
Action Selection
```

---

## 2. Input

The current FDRL-IDS baseline uses **20 selected network features**.

Therefore, the demonstration neural network was configured with:

```text
Input size = 20
```

A single test state was created with:

```text
torch.Size([1, 20])
```

This represents:

```text
1 state
×
20 features
```

---

## 3. Neural Network Architecture

The starting architecture follows the project proposal:

```text
20
 ↓
256
 ↓
128
 ↓
64
 ↓
4
```

Implemented using PyTorch:

```text
Linear(20 → 256)
      ↓
    ReLU
      ↓
Linear(256 → 128)
      ↓
    ReLU
      ↓
Linear(128 → 64)
      ↓
    ReLU
      ↓
Linear(64 → 4)
```

The four output values will eventually represent the Q-values for the four IDS actions:

| Output | Action  |
| -----: | ------- |
|      0 | BENIGN  |
|      1 | ATTACK  |
|      2 | BLOCK   |
|      3 | MONITOR |

This architecture is being used as the initial proposal-based design. Its effectiveness will be evaluated later rather than assuming that it is optimal.

---

## 4. Activation Function

The hidden layers use the **ReLU** activation function.

ReLU helps the neural network learn nonlinear relationships between the input features and its outputs.

The implementation used:

```python
nn.ReLU()
```

---

## 5. Test Input

A randomly generated test state was used:

```python
state = torch.randn(1, 20)
```

This was intentionally used instead of CICIDS2017 data.

The purpose was to test whether the neural network could process the correct input shape without introducing data-processing or dataset-related issues.

---

## 6. Network Output

The network successfully produced four output values:

```text
[0.0536, 0.1052, -0.0068, 0.0769]
```

The output shape was:

```text
torch.Size([1, 4])
```

The largest output was:

```text
0.1052
```

at index:

```text
1
```

Therefore, the selected action index in this demonstration was:

```text
1 = ATTACK
```

This does **not** mean that the network learned that the test state was an attack.

The test state was randomly generated and the network was not trained. The result only confirms that the network can produce four outputs and select the output with the highest value.

---

## 7. Loss Function

Mean Squared Error loss was used to test the loss-calculation process:

```python
nn.MSELoss()
```

The test produced:

```text
Loss = 7.22947883605957
```

The purpose was to verify that the network output could be compared with target values.

The loss value is not treated as a model-performance result because the network was not being trained on real IDS data.

---

## 8. Backpropagation

Backpropagation was successfully executed using:

```python
loss.backward()
```

This confirms that PyTorch was able to calculate gradients through the neural network.

The result was:

```text
Backpropagation: Completed successfully.
```

---

## 9. Optimizer

The Adam optimizer was used for the test:

```python
torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

An optimizer step was successfully completed.

This confirms that the model parameters can be updated based on the calculated gradients.

---

## 10. Day 28 Implementation

The implementation was saved as:

```text
src/day28_neural_network.py
```

The script successfully tested:

```text
Network creation
       ↓
Input processing
       ↓
Forward pass
       ↓
Q-value generation
       ↓
Action selection
       ↓
Loss calculation
       ↓
Backpropagation
       ↓
Adam optimizer update
```

---

## 11. Environment

PyTorch was installed successfully in the project virtual environment.

Environment used:

```text
Python: 3.13.5
PyTorch: 2.14.0+cpu
Device: CPU
CUDA: False
```

CPU execution is sufficient for this small educational neural-network experiment.

---

## 12. Result

Day 28 was successfully completed.

The following components were verified:

* PyTorch installation
* Neural network creation
* 20-feature input
* 256 → 128 → 64 hidden architecture
* ReLU activation
* Four-output layer
* Q-value generation
* Action selection
* Loss calculation
* Backpropagation
* Adam optimizer

The neural network is now ready to become the neural-network component of the DQN.

---

## 13. Important Limitation

The Day 28 experiment did **not** train an IDS model.

The input was randomly generated and no CICIDS2017 samples were used.

Therefore, the Day 28 output should not be interpreted as:

* IDS accuracy
* attack-detection performance
* DQN performance
* classification performance
* trained Q-values

Those experiments will be performed later after the DQN and IDS environment are implemented.

---

## 14. Connection to the Next Day

Day 27 introduced Q-learning.

Day 28 introduced neural networks.

The next step is to combine them:

```text
Q-learning
    +
Neural Network
    ↓
DQN
```

Therefore, **Day 29 will focus on understanding and implementing the basic DQN architecture, including the relationship between states, actions, Q-values, experience replay, and the target network.**

---

## Day 28 Status

**COMPLETED**

The neural-network component successfully passed the initial technical test and is ready for integration into the DQN stage.
