# IDS Concepts

## 1. What is IDS?

IDS stands for Intrusion Detection System.

An IDS is a security system that monitors network or computer activity and detects possible attacks.

Basic flow:

Network Traffic
      ↓
     IDS
      ↓
BENIGN / ATTACK

---

## 2. Signature-Based IDS

A signature-based IDS detects attacks by comparing network activity with known attack patterns or signatures.

Example:

If the IDS already knows the pattern of a specific attack, it can identify similar traffic.

Advantages:
- Good at detecting known attacks.
- Usually produces fewer false alarms for known patterns.

Disadvantage:
- May not detect new or unknown attacks effectively.

Simple idea:

Signature-Based IDS = "I know this attack pattern."

---

## 3. Anomaly-Based IDS

An anomaly-based IDS detects behavior that is different from normal behavior.

It first learns or defines what normal network behavior looks like. If new traffic is significantly different, it may be considered suspicious.

Example:

Normal traffic:
10–50 requests per minute

New traffic:
10,000 requests per minute

The IDS may identify this as suspicious.

Advantages:
- Can detect unusual or previously unknown attacks.

Disadvantage:
- Can produce false positives.

Simple idea:

Anomaly-Based IDS = "This behavior looks unusual."

---

## 4. Network-Based IDS

Network-Based IDS (NIDS) monitors network traffic to detect suspicious or malicious activity.

Basic flow:

Network Traffic
      ↓
     NIDS
      ↓
BENIGN / ATTACK

Our project is related to network-based intrusion detection because we work with network traffic data from CICIDS2017.

---

## 5. ML-Based IDS

A machine-learning-based IDS uses machine learning algorithms to learn patterns from network traffic data.

Example:

Network Features
      ↓
   ML Model
      ↓
BENIGN / ATTACK

The model learns from examples of benign and attack traffic.

ML-based IDS can reduce the need to manually define every attack pattern.

---

## 6. Connection to Our FDRL-IDS Project

Our project is a Federated Deep Reinforcement Learning Intrusion Detection System (FDRL-IDS).

The project uses network traffic data from CICIDS2017.

The basic idea is:

Network Traffic
      ↓
CICIDS2017 Data
      ↓
Data Preprocessing
      ↓
Multiple Federated Clients
      ↓
Federated Learning
      ↓
Global Knowledge
      ↓
DRL Agent
      ↓
Security Decision

The DRL agent can make decisions such as:

- BENIGN
- ATTACK
- BLOCK
- MONITOR