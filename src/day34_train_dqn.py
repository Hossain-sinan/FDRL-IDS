

"""
Day 34 - First Small DQN Training Experiment

Step 2:
Create a small balanced training dataset from CICIDS2017.

We use:
    500 BENIGN samples
    500 DDoS samples

Total:
    1000 samples
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# 1. Project path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# 2. File paths
# ---------------------------------------------------------

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

FEATURE_FILE = PROJECT_ROOT / "docs" / "day16_features_20.csv"


# ---------------------------------------------------------
# 3. Small experiment settings
# ---------------------------------------------------------

BENIGN_SAMPLES = 500
ATTACK_SAMPLES = 500


print("=" * 70)
print("DAY 34 - DQN TRAINING EXPERIMENT")
print("=" * 70)


# ---------------------------------------------------------
# 4. Load selected 20 features
# ---------------------------------------------------------

features_df = pd.read_csv(FEATURE_FILE)

selected_features = (
    features_df["feature"]
    .dropna()
    .astype(str)
    .tolist()
)

print()
print("Number of selected features:", len(selected_features))


# ---------------------------------------------------------
# 5. Load CSV data
# ---------------------------------------------------------

print()
print("Loading required data...")

df = pd.read_csv(RAW_FILE)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("Loaded rows:", len(df))
print("Loaded columns:", len(df.columns))


# ---------------------------------------------------------
# 6. Check required columns
# ---------------------------------------------------------

missing_features = [
    feature
    for feature in selected_features
    if feature not in df.columns
]

if missing_features:
    print()
    print("ERROR: Some selected features are missing:")
    print(missing_features)
    raise ValueError("Required features were not found.")


if "Label" not in df.columns:
    raise ValueError("Label column was not found.")


# Keep only the 20 features and Label
df = df[selected_features + ["Label"]]

# ---------------------------------------------------------
# 7. Select BENIGN samples
# ---------------------------------------------------------

benign_df = df[df["Label"] == "BENIGN"].sample(
    n=BENIGN_SAMPLES,
    random_state=42
)


# ---------------------------------------------------------
# 8. Select DDoS samples
# ---------------------------------------------------------

attack_df = df[df["Label"] == "DDoS"].sample(
    n=ATTACK_SAMPLES,
    random_state=42
)


# ---------------------------------------------------------
# 9. Combine both classes
# ---------------------------------------------------------

small_df = pd.concat(
    [benign_df, attack_df],
    ignore_index=True
)

# Shuffle the final dataset
small_df = small_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


print()
print("Small training dataset created.")
print("Dataset shape:", small_df.shape)


# ---------------------------------------------------------
# 10. Convert labels to binary values
# ---------------------------------------------------------

small_df["label_id"] = (
    small_df["Label"] != "BENIGN"
).astype(int)


print()
print("Binary label distribution:")
print(small_df["label_id"].value_counts().sort_index())


# ---------------------------------------------------------
# 11. Create state data
# ---------------------------------------------------------

states = small_df[selected_features].copy()

labels = small_df["label_id"].to_numpy()


# ---------------------------------------------------------
# 12. Convert feature values to numeric
# ---------------------------------------------------------

states = states.apply(
    pd.to_numeric,
    errors="coerce"
)


# ---------------------------------------------------------
# 13. Replace invalid values
# ---------------------------------------------------------

states = states.replace(
    [np.inf, -np.inf],
    np.nan
)

states = states.fillna(0)


# ---------------------------------------------------------
# 14. Scale features between 0 and 1
# ---------------------------------------------------------

scaler = MinMaxScaler()

states = scaler.fit_transform(states)


# ---------------------------------------------------------
# 15. Final information
# ---------------------------------------------------------

print()
print("Final state shape:", states.shape)
print("Final label shape:", labels.shape)

print()
print("First state:")
print(states[0])

print()
print("First label:")
print(labels[0])

print()
print("Minimum feature value:", states.min())
print("Maximum feature value:", states.max())

print()
print("STEP 2 COMPLETED")
print("Small balanced training data is ready.")

# ---------------------------------------------------------
# 16. Create IDS environment
# ---------------------------------------------------------

from src.day31_ids_environment import IDSEnvironment

print()
print("Creating IDS environment...")

env = IDSEnvironment(
    states=states,
    labels=labels
)


# ---------------------------------------------------------
# 17. Test environment
# ---------------------------------------------------------

state = env.reset()

print()
print("Initial state shape:", state.shape)

next_state, reward, done, info = env.step(0)

print("Action:", 0)
print("Reward:", reward)
print("Done:", done)
print("Info:", info)

print()
print("STEP 3 COMPLETED")
print("DQN training data is connected to the IDS environment.")

# ---------------------------------------------------------
# 18. Create DQN network
# ---------------------------------------------------------

from src.day32_dqn_network import DQNNetwork
from src.day33_dqn_components import (
    DQNAgentNetworks,
    EpsilonGreedyPolicy,
    ReplayBuffer,
)
import torch

print()
print("Creating DQN network...")

networks = DQNAgentNetworks(
    state_size=20,
    action_size=4
)

network = networks.online_network
network.train()

optimizer = torch.optim.Adam(
    network.parameters(),
    lr=0.001
)

loss_function = torch.nn.MSELoss()


# ---------------------------------------------------------
# 19. Test network with one state
# ---------------------------------------------------------

state_tensor = torch.tensor(
    state,
    dtype=torch.float32
).unsqueeze(0)

with torch.no_grad():
    q_values = network(state_tensor)


print()
print("Q-values:")
print(q_values)

print()
print("Q-values shape:", q_values.shape)


# ---------------------------------------------------------
# 20. Select action
# ---------------------------------------------------------

action = torch.argmax(
    q_values,
    dim=1
).item()

print()
print("Selected action:", action)

print()
print("STEP 4 COMPLETED")
print("DQN network can process an IDS state.")

# ---------------------------------------------------------
# 21. Create replay buffer
# ---------------------------------------------------------

print()
print("Creating replay buffer...")

replay_buffer = ReplayBuffer(
    capacity=10_000
)


# ---------------------------------------------------------
# 22. Store one experience
# ---------------------------------------------------------

replay_buffer.push(
    state,
    action,
    reward,
    next_state,
    done
)

print()
print("Replay buffer size:", len(replay_buffer))

print()
print("STEP 5 COMPLETED")
print("Experience can be stored in the replay buffer.")


        
        
# ---------------------------------------------------------
# 23. Collect experiences using epsilon-greedy
# ---------------------------------------------------------

print()
print("Collecting experiences with epsilon-greedy...")

policy = EpsilonGreedyPolicy(
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    action_size=4
)

print("Initial epsilon:", policy.epsilon)

state = env.reset()

for step in range(50):

    # Choose action using epsilon-greedy
    current_action = policy.select_action(
        state,
        network
    )

    # Take action in environment
    new_state, new_reward, new_done, info = env.step(
        current_action
    )

    # Store experience
    replay_buffer.push(
        state,
        current_action,
        new_reward,
        new_state,
        new_done
    )

    # Move to next state
    state = new_state

    # Reduce epsilon
    policy.decay_epsilon()

    if new_done:
        state = env.reset()

print()
print("Experiences collected:", len(replay_buffer))
print("Final epsilon:", policy.epsilon)

print()
print("STEP 6 COMPLETED")
print("DQN agent collected training experiences using epsilon-greedy.")

# ---------------------------------------------------------
# 24. Sample training batch
# ---------------------------------------------------------

BATCH_SIZE = 32

print()
print("Sampling training batch...")

(
    batch_states,
    batch_actions,
    batch_rewards,
    batch_next_states,
    batch_dones
) = replay_buffer.sample(BATCH_SIZE)

print()
print("Batch states shape:", batch_states.shape)
print("Batch actions shape:", batch_actions.shape)
print("Batch rewards shape:", batch_rewards.shape)
print("Batch next states shape:", batch_next_states.shape)
print("Batch dones shape:", batch_dones.shape)

print()
print("STEP 7 COMPLETED")
print("Training batch is ready.")

# ---------------------------------------------------------
# 25. Calculate target Q-values
# ---------------------------------------------------------

GAMMA = 0.99

print()
print("Calculating target Q-values...")

with torch.no_grad():

    # Get Q-values for the next states from the target network
    next_q_values = networks.target_network(
        batch_next_states
    )

    # Select the best future Q-value
    max_next_q_values = torch.max(
        next_q_values,
        dim=1
    ).values

    # Calculate the DQN target
    target_q_values = (
        batch_rewards
        + GAMMA
        * max_next_q_values
        * (1 - batch_dones)
    )


print()
print("Target Q-values shape:", target_q_values.shape)

print()
print("First 5 target Q-values:")
print(target_q_values[:5])

print()
print("STEP 8 COMPLETED")
print("Target Q-values calculated.") 

# ---------------------------------------------------------
# 26. Calculate current Q-values
# ---------------------------------------------------------

print()
print("Calculating current Q-values...")

# Get Q-values for all actions
current_q_values = network(
    batch_states
)

# Select only the action that was actually taken
current_q_values = current_q_values.gather(
    1,
    batch_actions.unsqueeze(1)
).squeeze(1)


print()
print("Current Q-values shape:", current_q_values.shape)

print()
print("First 5 current Q-values:")
print(current_q_values[:5])

print()
print("STEP 9 COMPLETED")
print("Current Q-values calculated.")

# ---------------------------------------------------------
# 27. Calculate DQN loss
# ---------------------------------------------------------

print()
print("Calculating DQN loss...")

loss = loss_function(
    current_q_values,
    target_q_values
)

print()
print("DQN Loss:", loss.item())

print()
print("STEP 10 COMPLETED")
print("DQN loss calculated.")

# ---------------------------------------------------------
# 28. Update DQN network
# ---------------------------------------------------------

print()
print("Updating DQN network...")

# Q-value before updating weights
q_before = network(batch_states[0:1]).detach()

optimizer.zero_grad()
loss.backward()
optimizer.step()

# Update the target network at the configured interval
training_step = step + 1

if networks.should_update_target(training_step):
    networks.update_target_network()
    print("Target network updated at step:", training_step)

# Q-value after updating weights
q_after = network(batch_states[0:1]).detach()

print()
print("Q-value before update:")
print(q_before)

print()
print("Q-value after update:")
print(q_after)

print()
print("STEP 11 COMPLETED")
print("DQN network learned from the training loss.")

print()
print("STEP 12 COMPLETED")
print("DQN network output changed after the weight update.")


# ---------------------------------------------------------
# 29. Multiple DQN training updates
# ---------------------------------------------------------

print()
print("Starting multiple DQN training updates...")

TRAINING_STEPS = 10

loss_history = []

for step in range(TRAINING_STEPS):

    # Sample a new batch
    (
        batch_states,
        batch_actions,
        batch_rewards,
        batch_next_states,
        batch_dones
    ) = replay_buffer.sample(BATCH_SIZE)

    # Calculate target Q-values
    with torch.no_grad():

        next_q_values = networks.target_network(
            batch_next_states
        )

        max_next_q_values = torch.max(
            next_q_values,
            dim=1
        ).values

        target_q_values = (
            batch_rewards
            + GAMMA
            * max_next_q_values
            * (1 - batch_dones)
        )

    # Calculate current Q-values
    current_q_values = network(
        batch_states
    )

    current_q_values = current_q_values.gather(
        1,
        batch_actions.unsqueeze(1)
    ).squeeze(1)

    # Calculate loss
    loss = loss_function(
        current_q_values,
        target_q_values
    )

  
    # Update online network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update target network at the configured interval
    training_step = step + 1

    if networks.should_update_target(training_step):
        networks.update_target_network()
        print("Target network updated at step:", training_step)


    loss_history.append(loss.item())

    print(
        f"Training step {step + 1}: "
        f"Loss = {loss.item():.4f}"
    )

print()
print("STEP 14 COMPLETED")
print("DQN completed multiple training updates.")

# ---------------------------------------------------------
# 30. Test target network update
# ---------------------------------------------------------

print()
print("Testing target network update...")

training_step = 100

if networks.should_update_target(training_step):

    networks.update_target_network()

    print()
    print("Target network updated.")
    print("Training step:", training_step)

else:

    print()
    print("Target network was not updated.")
    print("Training step:", training_step)

print()
print("STEP 15 COMPLETED")
print("Target network update mechanism is working.")


# ---------------------------------------------------------
# 31. Training result summary
# ---------------------------------------------------------

print()
print("Training result summary...")

print("Number of training updates:", TRAINING_STEPS)

print(
    "First training loss:",
    round(loss_history[0], 4)
)

print(
    "Last training loss:",
    round(loss_history[-1], 4)
)

print(
    "Experience collection epsilon:",
    round(policy.epsilon, 4)
)

print()
print("Loss history:")
print([round(loss, 4) for loss in loss_history])

print()
print("STEP 16 COMPLETED")
print("DQN training results were recorded.")
