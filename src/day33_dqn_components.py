
"""
DAY 33 — DQN Components

Federated Deep Reinforcement Learning Intrusion Detection System (FDRL-IDS)

This file implements:
1. Experience Replay Buffer
2. Target Network
3. Epsilon-Greedy Action Selection

Day 32 provides the DQN neural network.
Day 33 adds the supporting components needed for DQN training.
"""

# ============================================================
# IMPORT PATH FIX
# ============================================================

import sys
from pathlib import Path

# Get the FDRL-IDS project root directory.
# __file__ = D:\Projects\FDRL-IDS\src\day33_dqn_components.py
# parents[1] = D:\Projects\FDRL-IDS
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add project root to Python's import path.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import random
from collections import deque

import numpy as np
import torch

from src.day32_dqn_network import DQNNetwork


# ============================================================
# 1. EXPERIENCE REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    """
    Stores DQN experiences and randomly samples training batches.

    Each experience contains:

        state
        action
        reward
        next_state
        done
    """

    def __init__(self, capacity=10_000):
        """
        Create the replay buffer.

        Parameters:
            capacity: Maximum number of experiences stored.
        """

        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        """
        Store one experience in the replay buffer.
        """

        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences.

        Returns:
            states
            actions
            rewards
            next_states
            dones
        """

        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Not enough experiences in buffer. "
                f"Have {len(self.buffer)}, need {batch_size}."
            )

        batch = random.sample(
            self.buffer,
            batch_size
        )

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            np.array(states),
            dtype=torch.float32
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32
        )

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def __len__(self):
        """
        Return the current number of stored experiences.
        """

        return len(self.buffer)


# ============================================================
# 2. ONLINE + TARGET NETWORKS
# ============================================================

class DQNAgentNetworks:
    """
    Manages the online network and target network.

    Online network:
        Learns continuously.

    Target network:
        Provides more stable Q-value targets.
    """

    def __init__(
        self,
        state_size=20,
        action_size=4,
        target_update_frequency=100
    ):
        """
        Create online and target DQN networks.
        """

        self.state_size = state_size
        self.action_size = action_size
        self.target_update_frequency = target_update_frequency

        # ----------------------------------------------------
        # Online Network
        # ----------------------------------------------------

        self.online_network = DQNNetwork(
            input_size=state_size,
            output_size=action_size
        )

        # ----------------------------------------------------
        # Target Network
        # ----------------------------------------------------

        self.target_network = DQNNetwork(
            input_size=state_size,
            output_size=action_size
        )

        # Initially synchronize both networks.
        self.update_target_network()

    def update_target_network(self):
        """
        Copy the online network's parameters
        to the target network.
        """

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

    def should_update_target(self, step):
        """
        Check whether the target network
        should be updated at this step.
        """

        return (
            step > 0
            and step % self.target_update_frequency == 0
        )


# ============================================================
# 3. EPSILON-GREEDY POLICY
# ============================================================

class EpsilonGreedyPolicy:
    """
    Implements epsilon-greedy action selection.

    With probability epsilon:
        choose a random action.

    Otherwise:
        choose the action with the highest Q-value.
    """

    def __init__(
        self,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        action_size=4
    ):
        """
        Create epsilon-greedy policy.
        """

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.action_size = action_size

    def select_action(self, state, network):
        """
        Select an action using epsilon-greedy strategy.
        """

        # ----------------------------------------------------
        # Exploration
        # ----------------------------------------------------

        if random.random() < self.epsilon:

            return random.randrange(
                self.action_size
            )

        # ----------------------------------------------------
        # Exploitation
        # ----------------------------------------------------

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = network(
                state_tensor
            )

        action = torch.argmax(
            q_values,
            dim=1
        ).item()

        return int(action)

    def decay_epsilon(self):
        """
        Reduce epsilon after a training step.
        """

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )


# ============================================================
# 4. DAY 33 TEST
# ============================================================

def main():

    print("=" * 70)
    print("DAY 33 — DQN COMPONENTS TEST")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    # ========================================================
    # TEST 1 — REPLAY BUFFER
    # ========================================================

    print("\n[1] TESTING EXPERIENCE REPLAY BUFFER")

    replay_buffer = ReplayBuffer(
        capacity=10_000
    )

    print(
        "Buffer capacity:",
        replay_buffer.capacity
    )

    # --------------------------------------------------------
    # Add 32 sample experiences
    # --------------------------------------------------------

    for _ in range(32):

        state = np.random.rand(
            20
        ).astype(np.float32)

        next_state = np.random.rand(
            20
        ).astype(np.float32)

        action = random.randint(
            0,
            3
        )

        reward = random.choice(
            [-20, -5, 2, 10]
        )

        done = random.choice(
            [False, True]
        )

        replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    print(
        "Stored experiences:",
        len(replay_buffer)
    )

    # --------------------------------------------------------
    # Sample batch
    # --------------------------------------------------------

    batch_size = 8

    (
        states,
        actions,
        rewards,
        next_states,
        dones
    ) = replay_buffer.sample(
        batch_size
    )

    print("\nSampled batch:")
    print(
        "States shape:",
        states.shape
    )

    print(
        "Actions shape:",
        actions.shape
    )

    print(
        "Rewards shape:",
        rewards.shape
    )

    print(
        "Next states shape:",
        next_states.shape
    )

    print(
        "Dones shape:",
        dones.shape
    )

    # ========================================================
    # TEST 2 — TARGET NETWORK
    # ========================================================

    print("\n[2] TESTING ONLINE + TARGET NETWORK")

    networks = DQNAgentNetworks(
        state_size=20,
        action_size=4,
        target_update_frequency=100
    )

    print("Online network created.")
    print("Target network created.")

    # --------------------------------------------------------
    # Check initial synchronization
    # --------------------------------------------------------

    networks_synchronized = all(
        torch.equal(
            online_parameter,
            target_parameter
        )
        for online_parameter,
        target_parameter in zip(
            networks.online_network.parameters(),
            networks.target_network.parameters()
        )
    )

    print(
        "Networks initially synchronized:",
        networks_synchronized
    )

    print(
        "Target update frequency:",
        networks.target_update_frequency
    )

    # --------------------------------------------------------
    # Test target update schedule
    # --------------------------------------------------------

    print("\nTarget update schedule:")

    test_steps = [
        50,
        99,
        100,
        101,
        200
    ]

    for step in test_steps:

        print(
            f"Step {step}: "
            f"update target = "
            f"{networks.should_update_target(step)}"
        )

    # ========================================================
    # TEST 3 — EPSILON-GREEDY
    # ========================================================

    print("\n[3] TESTING EPSILON-GREEDY POLICY")

    policy = EpsilonGreedyPolicy(
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        action_size=4
    )

    test_state = np.random.rand(
        20
    ).astype(np.float32)

    print(
        "Initial epsilon:",
        policy.epsilon
    )

    # --------------------------------------------------------
    # Select several actions while epsilon = 1.0
    # --------------------------------------------------------

    actions = []

    for _ in range(10):

        action = policy.select_action(
            test_state,
            networks.online_network
        )

        actions.append(action)

    print(
        "Actions during exploration:",
        actions
    )

    # --------------------------------------------------------
    # Decay epsilon
    # --------------------------------------------------------

    policy.decay_epsilon()

    print(
        "Epsilon after one decay:",
        policy.epsilon
    )

    # ========================================================
    # TEST 4 — ACTION VALIDATION
    # ========================================================

    print("\n[4] TESTING ACTION RANGE")

    valid_actions = all(
        action in [0, 1, 2, 3]
        for action in actions
    )

    print(
        "All actions are valid:",
        valid_actions
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 70)

    if (
        networks_synchronized
        and valid_actions
        and len(replay_buffer) == 32
    ):
        print("DAY 33 TEST PASSED")
        print("Replay Buffer: PASS")
        print("Target Network: PASS")
        print("Epsilon-Greedy: PASS")
        print("Action Validation: PASS")
    else:
        print("DAY 33 TEST NEEDS ATTENTION")

    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

