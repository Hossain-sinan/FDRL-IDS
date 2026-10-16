import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """
    Neural network used by the DQN agent.

    Input:
        20 network traffic features

    Output:
        4 Q-values, one for each IDS action
    """

    def __init__(self, input_size=20, output_size=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, output_size)
        )

    def forward(self, state):
        return self.network(state)


if __name__ == "__main__":

    print("=" * 70)
    print("DAY 32 — DQN NETWORK TEST")
    print("=" * 70)

    # Create the DQN network
    model = DQNNetwork()

    print("\n[MODEL ARCHITECTURE]")
    print(model)

    # Create one fake state with 20 features
    sample_state = torch.tensor(
        [[
            0.10, 0.20, 0.30, 0.40, 0.50,
            0.60, 0.70, 0.80, 0.90, 0.15,
            0.25, 0.35, 0.45, 0.55, 0.65,
            0.75, 0.85, 0.95, 0.05, 0.12
        ]],
        dtype=torch.float32
    )

    print("\n[INPUT]")
    print("Shape:", sample_state.shape)

    # Pass the state through the network
    q_values = model(sample_state)

    print("\n[OUTPUT Q-VALUES]")
    print(q_values)

    print("\n[OUTPUT SHAPE]")
    print(q_values.shape)

    # Select action with highest Q-value
    action = torch.argmax(q_values, dim=1).item()

    action_names = {
        0: "BENIGN",
        1: "ATTACK",
        2: "BLOCK",
        3: "MONITOR"
    }

    print("\n[SELECTED ACTION]")
    print("Action number:", action)
    print("Action name:", action_names[action])

    print("\n" + "=" * 70)
    print("DAY 32 NETWORK TEST COMPLETE")
    print("=" * 70)