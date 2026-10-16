"""
Day 31 — IDS Environment Test

Tests the basic state -> action -> reward flow.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.day31_ids_environment import IDSEnvironment


# Three simple network states.
states = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
], dtype=np.float32)


# 0 = BENIGN
# 1 = ATTACK
labels = np.array([
    0,
    1,
    0,
])


env = IDSEnvironment(states, labels)


print("=" * 60)
print("DAY 31 — IDS ENVIRONMENT TEST")
print("=" * 60)


# Reset environment.
state = env.reset()

print("\nInitial state:")
print(state)


# Step 1:
# Ground truth = BENIGN
# Action = BENIGN
# Expected reward = +2
next_state, reward, done, info = env.step(
    IDSEnvironment.ACTION_BENIGN
)

print("\nStep 1")
print("Action:", info["action"])
print("Ground truth:", info["ground_truth"])
print("Result:", info["result"])
print("Reward:", reward)
print("Done:", done)


# Step 2:
# Ground truth = ATTACK
# Action = ATTACK
# Expected reward = +10
next_state, reward, done, info = env.step(
    IDSEnvironment.ACTION_ATTACK
)

print("\nStep 2")
print("Action:", info["action"])
print("Ground truth:", info["ground_truth"])
print("Result:", info["result"])
print("Reward:", reward)
print("Done:", done)


# Step 3:
# Ground truth = BENIGN
# Action = ATTACK
# Expected reward = -5
next_state, reward, done, info = env.step(
    IDSEnvironment.ACTION_ATTACK
)

print("\nStep 3")
print("Action:", info["action"])
print("Ground truth:", info["ground_truth"])
print("Result:", info["result"])
print("Reward:", reward)
print("Done:", done)


print("\nEnvironment test completed.")