# Day 33 — Experience Replay, Target Network and Epsilon-Greedy

## Project

**Federated Deep Reinforcement Learning Intrusion Detection System (FDRL-IDS)**

## Phase

**Phase 5 — DQN / Deep Reinforcement Learning IDS**

## Day

**Day 33**

---

## 1. Objective

The main goal of Day 33 was to implement and test three important components of the DQN agent:

1. Experience Replay Buffer
2. Target Network
3. Epsilon-Greedy Action Selection

These components are needed before starting the actual DQN training on Day 34.

---

## 2. What I Learned

### Experience Replay

In DQN, the agent collects experiences while interacting with the environment.

Each experience contains:

```text
State
Action
Reward
Next State
Done
```

Instead of training directly from the latest experience, the experiences are stored in a replay buffer.

Later, a random batch of experiences is selected for training.

This helps the model learn from different previous experiences instead of only learning from consecutive samples.

---

### Target Network

DQN uses two networks:

- Online Network
- Target Network

The online network is updated during training.

The target network is kept more stable and is updated from the online network after a fixed number of steps.

For this project, the target network update frequency is:

```text
100 steps
```

---

### Epsilon-Greedy Policy

The agent needs to balance:

- Exploration — trying random actions
- Exploitation — selecting the action with the highest Q-value

For this project, the initial epsilon value is:

```text
1.0
```

The minimum epsilon is:

```text
0.01
```

The decay value is:

```text
0.995
```

At the beginning, the agent mostly explores random actions. As training continues, epsilon becomes smaller and the agent starts using the learned Q-values more often.

---

## 3. Replay Buffer Implementation

I created the `ReplayBuffer` class in:

```text
src/day33_dqn_components.py
```

The buffer uses:

```python
deque(maxlen=10000)
```

So the maximum number of stored experiences is:

```text
10,000
```

The buffer was tested by storing 32 sample experiences.

The sampled batch size was:

```text
8
```

The test produced:

```text
States shape: torch.Size([8, 20])
Actions shape: torch.Size([8])
Rewards shape: torch.Size([8])
Next states shape: torch.Size([8, 20])
Dones shape: torch.Size([8])
```

The shapes are correct for the current DQN input size of 20 features.

---

## 4. Online and Target Network

I created both networks using the DQN architecture from Day 32.

The two networks were initially synchronized.

The test result was:

```text
Networks initially synchronized: True
```

The target network update frequency was also checked.

### Update Test

| Step | Update Target |
|---:|:---:|
| 50 | False |
| 99 | False |
| 100 | True |
| 101 | False |
| 200 | True |

This confirms that the target network is scheduled to update every 100 steps.

---

## 5. Epsilon-Greedy Test

I tested the epsilon-greedy policy with:

```text
Initial epsilon = 1.0
Epsilon decay = 0.995
Minimum epsilon = 0.01
```

During the test, random actions were generated because the initial epsilon was 1.0.

Example output:

```text
Actions during exploration: [1, 3, 3, 0, 0, 3, 0, 1, 3, 3]
```

After one epsilon decay:

```text
Epsilon after one decay: 0.995
```

The actions were also checked to make sure they were within the valid action range:

```text
0, 1, 2, 3
```

---

## 6. Day 33 Test Result

The complete test finished successfully.

```text
DAY 33 TEST PASSED

Replay Buffer: PASS
Target Network: PASS
Epsilon-Greedy: PASS
Action Validation: PASS
```

So all three DQN components are working correctly for the current implementation.

---

## 7. Challenge Faced

At first, I got this error when running the file from the project parent directory:

```text
ModuleNotFoundError: No module named 'src'
```

The problem was related to Python's module search path when running the script using its full path.

I fixed it by adding the project root to `sys.path` before importing the DQN network.

After the change, the complete Day 33 test ran successfully.

---

## 8. File Created

The main file created for Day 33 is:

```text
src/day33_dqn_components.py
```

This file contains:

- `ReplayBuffer`
- `DQNAgentNetworks`
- `EpsilonGreedyPolicy`
- Component testing code

---

## 9. Day 33 Workflow

The components can be understood as:

```text
Environment
     |
     v
  State
     |
     v
Epsilon-Greedy Policy
     |
     v
   Action
     |
     v
Reward + Next State
     |
     v
Replay Buffer
     |
     v
Random Batch
     |
     v
DQN Training
     |
     v
Online Network
     |
     v
Target Network
```

The actual training connection will be implemented on Day 34.

---

## 10. Time Allocation

Approximate time spent:

| Activity | Time |
|---|---:|
| Understanding replay buffer | 35 min |
| Implementing replay buffer | 40 min |
| Implementing target network | 30 min |
| Implementing epsilon-greedy | 30 min |
| Testing and fixing import issue | 30 min |
| Documentation | 10 min |
| **Total** | **~2 hr 55 min** |

---

## 11. Pending Work

### High Priority — Day 34

The next step is to start a small DQN training experiment.

The training flow will connect:

```text
IDS Environment
      ↓
Current State
      ↓
DQN Network
      ↓
Epsilon-Greedy Action
      ↓
Action
      ↓
Reward + Next State
      ↓
Replay Buffer
      ↓
Sample Batch
      ↓
Calculate Loss
      ↓
Optimizer Update
      ↓
Update Target Network
```

The first training experiment should be small. I should not start with the complete CICIDS2017 dataset immediately.

---

## 12. Key Takeaways

- Experience Replay stores previous agent experiences for training.
- Random sampling from the replay buffer helps avoid learning only from consecutive experiences.
- DQN uses an online network and a target network.
- The target network is updated every 100 steps in this project.
- Epsilon-Greedy allows the agent to explore and later use learned actions.
- The four valid actions are `0, 1, 2, 3`.
- All Day 33 component tests passed.
- The project is now ready for the first small DQN training experiment.

---

## 13. Completion Checklist

- [x] Implement Replay Buffer
- [x] Test storing experiences
- [x] Test random batch sampling
- [x] Implement Online Network
- [x] Implement Target Network
- [x] Test target synchronization
- [x] Test 100-step target update schedule
- [x] Implement Epsilon-Greedy Policy
- [x] Test epsilon decay
- [x] Validate action range
- [x] Fix Python import issue
- [x] Run complete Day 33 test
- [x] All tests passed

---

## 14. Conclusion

Day 33 is completed successfully.

I implemented the main supporting components required for DQN training: replay buffer, target network, and epsilon-greedy policy. I also tested each component and fixed the Python import problem found during testing.

The next step is **Day 34 — First Small-Scale DQN Training**, where these components will be connected with the IDS environment and used for actual training.
