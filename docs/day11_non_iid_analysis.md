# Day 11 — Non-IID Client Data Distribution

## 1. Objective

The main goal of Day 11 was to understand **Non-IID data** and prepare the CICIDS2017 dataset for the federated learning part of the FDRL-IDS project.

In Federated Learning, different clients normally do not have exactly the same type of data. This is called **Non-IID (Non-Independent and Identically Distributed) data**.

For this project, the CICIDS2017 dataset was divided into four simulated organizations:

* **Hospital** → Monday + Tuesday
* **Bank** → Wednesday
* **University** → Thursday
* **ISP** → Friday

The purpose was to check whether each client has a different distribution of network traffic and attack classes.

---

## 2. What is Non-IID Data?

Non-IID means that the data available at different clients is not distributed in the same way.

For example, if one client has mostly BENIGN traffic and another client has a large amount of DoS traffic, their data distributions are different.

In this project:

* Hospital has mostly BENIGN traffic and some Brute Force traffic.
* Bank has BENIGN and a large amount of DoS traffic.
* University has mostly BENIGN traffic with Web Attack and Infiltration traffic.
* ISP has BENIGN, DDoS, Port Scan and Botnet traffic.

Therefore, the four clients have different data distributions.

This makes the experiment more realistic for Federated Learning.

---

## 3. Federated Client Design

The original CICIDS2017 files were grouped according to the proposed organization/client structure.

### Client 1 — Hospital

Dataset:

* Monday-WorkingHours.pcap_ISCX.csv
* Tuesday-WorkingHours.pcap_ISCX.csv

Reason:

The Hospital client represents network traffic collected during Monday and Tuesday.

### Client 2 — Bank

Dataset:

* Wednesday-workingHours.pcap_ISCX.csv

Reason:

The Bank client uses Wednesday traffic.

### Client 3 — University

Dataset:

* Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
* Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv

Reason:

The University client uses Thursday traffic, including Web Attack and Infiltration traffic.

### Client 4 — ISP

Dataset:

* Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
* Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
* Friday-WorkingHours-Morning.pcap_ISCX.csv

Reason:

The ISP client uses Friday traffic, which contains DDoS, Port Scan and Botnet traffic.

---

## 4. Initial Data Processing

The analysis was performed using Python and Pandas.

Because the complete CICIDS2017 dataset is large, the CSV files were processed in chunks instead of loading everything into memory at once.

The `Label` column was selected because Day 11 focused on understanding the distribution of attack classes between clients.

The same label mapping from Day 10 was used:

| Original Label             | Project Category |
| -------------------------- | ---------------- |
| BENIGN                     | BENIGN           |
| DoS Hulk                   | DoS              |
| DoS GoldenEye              | DoS              |
| DoS slowloris              | DoS              |
| DoS Slowhttptest           | DoS              |
| DDoS                       | DDoS             |
| FTP-Patator                | Brute Force      |
| SSH-Patator                | Brute Force      |
| Web Attack - Brute Force   | Web Attack       |
| Web Attack - XSS           | Web Attack       |
| Web Attack - Sql Injection | Web Attack       |
| Infiltration               | Infiltration     |
| Bot                        | Botnet           |
| PortScan                   | Port Scan        |

Heartbleed was excluded from the experimental dataset as decided during Day 10.

---

## 5. Mistake During the First Analysis

During the first Day 11 processing, the client distributions were calculated successfully, but the total number of records did not match the verified Day 10 total.

The first result was:

**2,828,552 records**

However, Day 10 had already established that the experimental dataset contained:

**2,830,732 records**

So there was a difference of:

**2,180 records**

At first, the distribution appeared reasonable, but the total-record check showed that something was missing.

This check was important because it prevented us from creating an incorrect final diagram.

---

## 6. Finding the Mistake

The missing number, **2,180**, was exactly equal to the total number of Web Attack records found during Day 10.

Therefore, the Web Attack records were investigated.

The actual labels in the Thursday Web Attack file were found to be:

* `Web Attack � Brute Force` → 1,507 records
* `Web Attack � XSS` → 652 records
* `Web Attack � Sql Injection` → 21 records

Total:

**1,507 + 652 + 21 = 2,180**

The problem was caused by a difference in the character used inside the Web Attack label. The mapping code was expecting a different dash/character, so those labels were not matched.

As a result, the Web Attack records were dropped during the mapping step.

---

## 7. Fixing the Mistake

The label mapping was corrected to use the exact labels present in the CSV files:

```python
label_mapping = {
    "BENIGN": "BENIGN",

    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",

    "DDoS": "DDoS",

    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",

    "Web Attack � Brute Force": "Web Attack",
    "Web Attack � XSS": "Web Attack",
    "Web Attack � Sql Injection": "Web Attack",

    "Infiltration": "Infiltration",

    "Bot": "Botnet",

    "PortScan": "Port Scan"
}
```

The client distribution was then calculated again.

The University client changed from:

**456,788 → 458,968 records**

because the missing 2,180 Web Attack records were correctly included.

---

## 8. Final Verification

After fixing the mapping, the total number of records from all four clients was checked.

The final total was:

**2,830,732 records**

This exactly matched the verified experimental dataset total from Day 10.

Therefore:

**Client Total = Day 10 Experimental Total**

The final verification result was:

```text
Grand Total: 2830732
Expected Total: 2,830,732
✓ VERIFIED: Client distribution matches Day 10.
```

This confirmed that no records were accidentally lost during the client distribution process.

---

## 9. Final Client Distribution

The final client distribution is shown below.

| Client     | Total Records | BENIGN (%) | DoS (%) | DDoS (%) | Brute Force (%) | Web Attack (%) | Infiltration (%) | Botnet (%) | Port Scan (%) |
| ---------- | ------------: | ---------: | ------: | -------: | --------------: | -------------: | ---------------: | ---------: | ------------: |
| Hospital   |       975,827 |    98.5822 |  0.0000 |   0.0000 |          1.4178 |         0.0000 |           0.0000 |     0.0000 |        0.0000 |
| Bank       |       692,692 |    63.5248 | 36.4752 |   0.0000 |          0.0000 |         0.0000 |           0.0000 |     0.0000 |        0.0000 |
| University |       458,968 |    99.5172 |  0.0000 |   0.0000 |          0.0000 |         0.4750 |           0.0078 |     0.0000 |        0.0000 |
| ISP        |       703,245 |    58.9157 |  0.0000 |  18.2052 |          0.0000 |         0.0000 |           0.0000 |     0.2796 |       22.5995 |
| **Total**  | **2,830,732** |            |         |          |                 |                |                  |            |               |

---

## 10. Important Observations

### Hospital

The Hospital client contains mostly BENIGN traffic:

* BENIGN = 98.5822%
* Brute Force = 1.4178%

This means the Hospital client has a very different distribution from the other clients.

### Bank

The Bank client contains:

* BENIGN = 63.5248%
* DoS = 36.4752%

DoS traffic is much more common in the Bank client's data compared with the Hospital and University clients.

### University

The University client contains:

* BENIGN = 99.5172%
* Web Attack = 0.4750%
* Infiltration = 0.0078%

The University client has the Web Attack and Infiltration classes.

### ISP

The ISP client contains:

* BENIGN = 58.9157%
* DDoS = 18.2052%
* Port Scan = 22.5995%
* Botnet = 0.2796%

The ISP client has a much larger amount of DDoS and Port Scan traffic.

---

## 11. Why This is Non-IID

The results clearly show that the clients do not have the same class distribution.

For example:

* Hospital has Brute Force traffic.
* Bank has a large amount of DoS traffic.
* University has Web Attack and Infiltration traffic.
* ISP has DDoS, Port Scan and Botnet traffic.

Therefore, the data distribution is different across clients.

This creates a **Non-IID federated learning environment**, which is one of the important conditions we want to study in the FDRL-IDS project.

---

## 12. Client Distribution Diagram

A stacked bar chart was created to visually represent the percentage distribution of traffic classes across the four federated clients.

The figure is saved at:

```text
docs/figures/day11_client_distribution.png
```

The diagram shows the different traffic distributions of:

**Hospital | Bank | University | ISP**

This makes the Non-IID property easier to understand visually.

---

## 13. Files Produced

The following files were created during Day 11:

```text
notebooks/
└── non_iid_client_distribution.ipynb

docs/
├── day11_client_distribution.csv
├── day11_non_iid_analysis.md
└── figures/
    └── day11_client_distribution.png
```

### Notebook

`non_iid_client_distribution.ipynb`

Contains the Python code used to:

* Define the four clients
* Check dataset files
* Read labels in chunks
* Map attack labels
* Calculate client distributions
* Verify the total records
* Create the distribution table
* Generate the diagram

### CSV

`day11_client_distribution.csv`

Contains the final client-level distribution percentages and record counts.

### Figure

`day11_client_distribution.png`

Contains the visual representation of the Non-IID client distributions.

---

## 14. Conclusion

Day 11 successfully demonstrated the Non-IID property required for the FDRL-IDS project.

The CICIDS2017 dataset was divided into four simulated organizations:

* Hospital
* Bank
* University
* ISP

Each client has a different distribution of network traffic and attack classes.

During the analysis, an error was found where 2,180 Web Attack records were not included because of a label-character mismatch. The problem was identified by comparing the Day 11 total with the verified Day 10 total. The labels were inspected, the mapping was corrected, and the analysis was run again.

The final client total became **2,830,732**, exactly matching the Day 10 experimental dataset.

Therefore, the client distribution is verified and ready for the next stage of the FDRL-IDS project.

**Day 11 Status: COMPLETED ✓**
