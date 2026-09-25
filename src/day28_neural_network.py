"""
Day 28 — Learn Neural Networks

Purpose:
Build and test a small PyTorch neural network
that will later be used as the Q-network for DQN.
"""

import torch
import torch.nn as nn


# =========================================================
# 1. Define the neural network
# =========================================================

class DQNNetwork(nn.Module):

    def __init__(self, input_size=20, num_actions=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_actions)
        )

    def forward(self, state):
        return self.network(state)


# =========================================================
# 2. Create the network
# =========================================================

model = DQNNetwork()

print("=" * 60)
print("DAY 28 — NEURAL NETWORK TEST")
print("=" * 60)

print("\n[1] Network architecture:")
print(model)


# =========================================================
# 3. Create one fake IDS state
# =========================================================

state = torch.randn(1, 20)

print("\n[2] Input state:")
print(state)

print("\nInput state shape:")
print(state.shape)


# =========================================================
# 4. Pass the state through the network
# =========================================================

q_values = model(state)

print("\n[3] Output Q-values:")
print(q_values)

print("\nQ-values shape:")
print(q_values.shape)


# =========================================================
# 5. Select the action with highest Q-value
# =========================================================

action = torch.argmax(q_values, dim=1)

print("\n[4] Selected action:")
print(action.item())


# =========================================================
# 6. Calculate loss
# =========================================================

target_q_values = torch.tensor(
    [[1.0, 2.0, 3.0, 4.0]]
)

loss_function = nn.MSELoss()

loss = loss_function(q_values, target_q_values)

print("\n[5] Loss:")
print(loss.item())


# =========================================================
# 7. Test backpropagation and optimizer
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

optimizer.zero_grad()

loss.backward()

optimizer.step()

print("\n[6] Backpropagation:")
print("Completed successfully.")

print("\n[7] Optimizer:")
print("Adam optimizer step completed.")


# =========================================================
# 8. Finish
# =========================================================

print("\n" + "=" * 60)
print("DAY 28 TEST COMPLETED SUCCESSFULLY")
print("=" * 60)