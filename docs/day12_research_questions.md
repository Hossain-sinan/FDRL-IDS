# Day 12 — Research Questions and Research Gap

## 1. Objective

The goal of Day 12 was to understand the research questions,
research gap, and research contribution of the FDRL-IDS project.

## 2. Research Questions

### RQ1
How effectively can Federated Learning detect network attacks
in the FDRL-IDS system?

### RQ2
How does the performance of the Federated Learning model compare
with a centralized learning model?

### RQ3
How quickly and consistently does the Federated Learning model
converge during training?

### RQ4
How does Non-IID client data affect Federated Learning performance,
and can FedProx improve the results?

### RQ5
How vulnerable is the Federated Learning system to poisoning attacks,
and how much do they affect model performance?

### RQ6
How does the proposed FDRL-IDS approach compare with traditional
supervised machine learning methods?

## 3. Research Gap

Existing research has studied intrusion detection, machine learning,
Federated Learning, and Deep Reinforcement Learning separately or in
different combinations.

This project focuses on evaluating their combination in a federated
intrusion detection environment with heterogeneous client data.

A major focus is the effect of Non-IID data across different simulated
organizations and the possible use of FedProx to improve federated
training.

The project also considers poisoning attacks and compares the proposed
approach with centralized learning and traditional supervised
machine learning methods.

## 4. Proposed Approach

The project proposes an FDRL-IDS framework using multiple simulated
organizations based on the CICIDS2017 dataset.

The four clients are:

- Hospital
- Bank
- University
- ISP

The client distributions were verified during Day 11 and were found to
be different, creating a Non-IID federated environment.

## 5. Research Contribution

The main contribution is the development and evaluation of an
FDRL-IDS framework combining Federated Learning and Deep Reinforcement
Learning for intrusion detection.

The study evaluates:

- Federated vs centralized learning
- Model convergence
- Non-IID data effects
- FedProx
- Poisoning attacks
- Supervised machine learning comparison

## 6. Research Gap Flow

Existing Research
        ↓
What is Missing?
        ↓
Our Approach
        ↓
What Will We Test?

## 7. Conclusion

Day 12 identified the six research questions and defined the research
gap and contribution of the FDRL-IDS project.

The project now has a clear connection between the research problem,
dataset, Non-IID client design, proposed approach, and experiments.