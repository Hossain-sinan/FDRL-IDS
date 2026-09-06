# CICIDS2017 Notes

## 1. What is CICIDS2017?

CICIDS2017 is a network intrusion detection dataset created by the Canadian Institute for Cybersecurity.

It contains realistic network traffic, including both normal traffic and malicious attack traffic.

The dataset is used to develop and evaluate Intrusion Detection Systems (IDS) and machine learning models.

---

## 2. Network Flow

A network flow is a summary of communication between network endpoints.

Instead of looking only at individual packets, flow-based data describes the communication using many measurements.

One row in the dataset represents a network flow.

Example information in a flow includes:

* Source information
* Destination information
* Source port
* Destination port
* Protocol
* Flow duration
* Number of packets
* Packet lengths

---

## 3. Features

Features are measurements that describe a network flow.

CICIDS2017 provides many traffic features generated using CICFlowMeter.

Examples include:

* Flow Duration
* Total Forward Packets
* Total Backward Packets
* Forward Packet Length
* Backward Packet Length
* Flow Bytes/s
* Flow Packets/s
* Packet Length Mean
* Packet Length Standard Deviation

The original dataset contains 78 traffic features.

---

## 4. Labels

A label tells us what type of traffic a network flow represents.

The dataset contains normal traffic labeled as BENIGN and malicious traffic with attack labels.

Examples:

* BENIGN
* DoS
* DDoS
* PortScan
* Bot
* Brute Force
* Web Attack
* Infiltration
* Heartbleed

---

## 5. Attack Categories

Important attack categories in CICIDS2017 include:

* DoS
* DDoS
* PortScan
* Bot
* Brute Force
* Web Attack
* Infiltration
* Heartbleed

Some categories contain more specific attack labels.

For example:

* DoS Hulk
* DoS GoldenEye
* DoS Slowloris
* DoS Slowhttptest
* FTP-Patator
* SSH-Patator

---

## 6. Five-Day Traffic Structure

CICIDS2017 traffic was collected over five days:

* Monday
* Tuesday
* Wednesday
* Thursday
* Friday

Different network activities and attack scenarios were performed on different days.

The five-day structure is important for this project because the data can later be divided into different federated clients.

---

## 7. Connection to FDRL-IDS

The CICIDS2017 dataset will be used as the network traffic source for the FDRL-IDS project.

The dataset can be divided into different portions to simulate different organizations or federated clients.

For example:

```text
CICIDS2017
     |
     +---- Client 1
     |
     +---- Client 2
     |
     +---- Client 3
     |
     +---- Client 4
```

Each client can receive different traffic distributions.

This can simulate a non-IID federated learning environment.

---

## 8. Important Understanding

The basic structure is:

```text
Network Traffic
       ↓
Network Flow
       ↓
Features
       ↓
IDS/ML/DRL Model
       ↓
Prediction / Decision
       ↓
BENIGN or ATTACK
```

The main purpose of understanding CICIDS2017 before processing it is to know what each row, feature, and label represents.

---

## 9. Day 7 Summary

I learned that:

1. CICIDS2017 is an intrusion detection dataset.
2. A network flow represents summarized network communication.
3. Features describe the network flow.
4. Labels describe whether the traffic is normal or malicious.
5. CICIDS2017 contains multiple attack categories.
6. The traffic was collected across five days.
7. The five-day structure can be useful for simulating federated clients.
8. The dataset must be understood before preprocessing and model training.
