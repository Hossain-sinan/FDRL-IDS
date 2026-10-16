"""
Day 31 — DQN IDS Environment

Purpose:
Create a simple reinforcement-learning environment for the
FDRL-IDS project.

The environment provides:

State:
    Network traffic features

Actions:
    0 = BENIGN
    1 = ATTACK
    2 = BLOCK
    3 = MONITOR

Rewards:
    Correct BENIGN decision = +2
    Correct attack-related decision = +10
    False positive = -5
    False negative = -20
"""


import numpy as np


class IDSEnvironment:
    """
    Simple IDS environment for the DQN agent.
    """

    ACTION_BENIGN = 0
    ACTION_ATTACK = 1
    ACTION_BLOCK = 2
    ACTION_MONITOR = 3

    def __init__(self, states, labels):
        """
        Initialize the environment.

        Parameters
        ----------
        states : numpy.ndarray
            Network traffic feature vectors.

        labels : numpy.ndarray
            Ground-truth labels.
            0 = BENIGN
            1 = ATTACK
        """

        self.states = np.asarray(states, dtype=np.float32)
        self.labels = np.asarray(labels)

        if len(self.states) != len(self.labels):
            raise ValueError(
                "Number of states and labels must be the same."
            )

        if len(self.states) == 0:
            raise ValueError(
                "Environment cannot be created with empty data."
            )

        self.current_index = 0

    def reset(self):
        """
        Reset the environment.

        Returns
        -------
        numpy.ndarray
            Initial state.
        """

        self.current_index = 0

        return self.states[self.current_index]

    def step(self, action):
        """
        Perform one environment step.

        Parameters
        ----------
        action : int
            Agent's selected action.

        Returns
        -------
        next_state : numpy.ndarray
            Next network state.

        reward : int
            Reward received by the agent.

        done : bool
            Whether the episode has finished.

        info : dict
            Additional information about the step.
        """

        valid_actions = [
            self.ACTION_BENIGN,
            self.ACTION_ATTACK,
            self.ACTION_BLOCK,
            self.ACTION_MONITOR,
        ]

        if action not in valid_actions:
            raise ValueError(
                f"Invalid action: {action}. "
                f"Valid actions are {valid_actions}."
            )

        ground_truth = self.labels[self.current_index]

        # Convert the four actions into a binary
        # security decision for reward calculation.
        predicted_benign = action == self.ACTION_BENIGN
        actual_benign = ground_truth == 0

        if actual_benign and predicted_benign:
            # True Negative
            reward = 2
            result = "TN"

        elif not actual_benign and not predicted_benign:
            # True Positive
            reward = 10
            result = "TP"

        elif actual_benign and not predicted_benign:
            # False Positive
            reward = -5
            result = "FP"

        else:
            # False Negative
            reward = -20
            result = "FN"

        # Move to the next sample.
        self.current_index += 1

        done = self.current_index >= len(self.states)

        if done:
            next_state = np.zeros_like(self.states[0])
        else:
            next_state = self.states[self.current_index]

        info = {
            "ground_truth": int(ground_truth),
            "action": int(action),
            "result": result,
        }

        return next_state, reward, done, info