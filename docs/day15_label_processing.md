# Day 15 — Label Processing

## Goal

The goal of Day 15 was to create a proper label strategy for the FDRL-IDS project and convert the original CICIDS2017 attack labels into the classes that will be used in our project.

I decided to use an **8-class multi-class attack representation as the primary label system**. I also created a **binary label** for future baseline comparison.

---

## 1. Label Strategy

The project uses two types of labels.

### Primary label — 8-class multi-class

| Label ID | Attack Category |
| -------: | --------------- |
|        0 | BENIGN          |
|        1 | DoS             |
|        2 | DDoS            |
|        3 | Brute Force     |
|        4 | Web Attack      |
|        5 | Infiltration    |
|        6 | Botnet          |
|        7 | Port Scan       |

I selected this as the primary label because our project needs to understand different types of network attacks instead of only deciding whether traffic is normal or malicious.

### Secondary label — Binary

The binary label is:

| Binary ID | Meaning |
| --------: | ------- |
|         0 | BENIGN  |
|         1 | ATTACK  |

This label will be useful later when comparing our system with simpler binary intrusion detection models.

---

## 2. Original CICIDS2017 Labels

The original dataset contains more detailed attack names such as:

* BENIGN
* DoS Hulk
* DoS GoldenEye
* DoS slowloris
* DoS Slowhttptest
* DDoS
* FTP-Patator
* SSH-Patator
* Web Attack Brute Force
* Web Attack XSS
* Web Attack SQL Injection
* Infiltration
* Bot
* PortScan
* Heartbleed

These labels are more detailed than the categories required for our project, so I grouped related attacks together.

---

## 3. Label Mapping

The mapping used in the project is:

| Original Label           | Project Category |
| ------------------------ | ---------------- |
| BENIGN                   | BENIGN           |
| DoS Hulk                 | DoS              |
| DoS GoldenEye            | DoS              |
| DoS slowloris            | DoS              |
| DoS Slowhttptest         | DoS              |
| DDoS                     | DDoS             |
| FTP-Patator              | Brute Force      |
| SSH-Patator              | Brute Force      |
| Web Attack Brute Force   | Web Attack       |
| Web Attack XSS           | Web Attack       |
| Web Attack SQL Injection | Web Attack       |
| Infiltration             | Infiltration     |
| Bot                      | Botnet           |
| PortScan                 | Port Scan        |

---

## 4. Web Attack Mapping Problem

During the first implementation, the Web Attack labels were not mapped correctly.

The reason was that the actual labels in my cleaned dataset contained a special replacement character:

```text
Web Attack � Brute Force
Web Attack � XSS
Web Attack � Sql Injection
```

My first mapping expected different strings, so these records were temporarily treated as unmapped.

I checked the actual dataset values and corrected the mapping to match the real labels.

After fixing the mapping, the Web Attack records were processed correctly.

Final Web Attack records:

```text
2143
```

This was an important debugging step because it showed that label mapping should always be checked against the actual dataset values.

---

## 5. Heartbleed Handling

After fixing the Web Attack mapping, only one label remained unmapped:

```text
Heartbleed    11
```

There were only 11 Heartbleed records after data cleaning.

Heartbleed is not included in the predefined 8-class strategy of this project. Therefore, these 11 records were excluded from the processed dataset.

This decision will also be mentioned in the final project documentation so that the dataset processing remains transparent.

---

## 6. Final Dataset Result

Before label processing:

```text
Cleaned rows: 2,520,798
```

Unmapped Heartbleed rows:

```text
11
```

After label processing:

```text
Processed rows: 2,520,787
```

Therefore:

```text
2,520,798 - 11 = 2,520,787
```

---

## 7. Final Multi-Class Distribution

The final 8-class distribution is:

| Class        |   Records |
| ------------ | --------: |
| BENIGN       | 2,095,057 |
| DoS          |   193,745 |
| DDoS         |   128,014 |
| Port Scan    |    90,694 |
| Brute Force  |     9,150 |
| Web Attack   |     2,143 |
| Botnet       |     1,948 |
| Infiltration |        36 |

Total:

```text
2,520,787
```

---

## 8. Numeric Label Distribution

The numeric label IDs are:

```text
0    2,095,057
1      193,745
2      128,014
3        9,150
4        2,143
5           36
6        1,948
7       90,694
```

The IDs correctly represent all 8 project classes.

---

## 9. Binary Label Distribution

The binary labels are:

```text
0    2,095,057
1      425,730
```

Here:

```text
0 = BENIGN
1 = ATTACK
```

The binary attack count is the sum of all seven attack categories.

---

## 10. Important Difference: Labels vs RL Actions

The dataset labels and the reinforcement learning actions are different things.

The dataset label tells us:

```text
What type of traffic is this?
```

For example:

```text
BENIGN
DoS
DDoS
Port Scan
```

The RL agent will later decide what action to take based on the traffic.

The planned RL actions are:

```text
BENIGN
ATTACK
BLOCK
MONITOR
```

Therefore, I should not treat the dataset label and RL action as the same concept.

---

## 11. Files Created

The main implementation file is:

```text
src/label_processing.py
```

The testing file is:

```text
src/test_label_processing.py
```

The final documentation is:

```text
docs/day15_label_processing.md
```

---

## 12. Final Verification

I created an automated test to verify the label processing.

The test confirmed:

```text
PASS: All 8 expected classes are present.
PASS: Label IDs are exactly 0-7.
PASS: Binary labels contain only 0 and 1.
PASS: No missing processed labels.
PASS: Final processed row count is correct.
```

Final result:

```text
DAY 15 LABEL PROCESSING TEST PASSED
```

---

## Conclusion

Day 15 is completed successfully.

I now have a reproducible label-processing step that converts the original CICIDS2017 labels into the 8 classes required for the FDRL-IDS project.

The processed dataset contains **2,520,787 records**, with all 8 classes available and no missing processed labels.

The binary label is also available for future baseline experiments.

The next step is to continue with the next stage of the data preprocessing pipeline.
