# Day 10 — Attack Label Analysis

## 1. Objective

Today my main goal was to understand the attack labels of the CICIDS2017 dataset.

In the project proposal, I planned to use 8 categories:

* BENIGN
* DoS
* DDoS
* Brute Force
* Web Attack
* Infiltration
* Botnet
* Port Scan

So, I checked the original labels and grouped them according to these 8 categories.

---

## 2. Dataset Information

I used the 8 CICIDS2017 CSV files that I collected for this project.

After reading the label column from all files, I found:

**Total records = 2,830,743**

Each record has 78 network traffic features and 1 label column.

One problem I found was that the actual label column name was:

```text
' Label'
```

There was a space before `Label`.

I solved this by removing the extra spaces from the column names before doing the analysis.

---

## 3. Original Labels

After checking all the files, I found 15 different labels:

```text
BENIGN
DDoS
PortScan
Bot
Infiltration
Web Attack – Brute Force
Web Attack – XSS
Web Attack – Sql Injection
FTP-Patator
SSH-Patator
DoS slowloris
DoS Slowhttptest
DoS Hulk
DoS GoldenEye
Heartbleed
```

So, the dataset has more labels than the 8 categories that I planned to use in my project.

---

## 4. Label Grouping

I grouped the original labels into the 8 project categories.

| Original Label             | New Category |
| -------------------------- | ------------ |
| BENIGN                     | BENIGN       |
| DoS Hulk                   | DoS          |
| DoS GoldenEye              | DoS          |
| DoS slowloris              | DoS          |
| DoS Slowhttptest           | DoS          |
| DDoS                       | DDoS         |
| FTP-Patator                | Brute Force  |
| SSH-Patator                | Brute Force  |
| Web Attack – Brute Force   | Web Attack   |
| Web Attack – XSS           | Web Attack   |
| Web Attack – Sql Injection | Web Attack   |
| Infiltration               | Infiltration |
| Bot                        | Botnet       |
| PortScan                   | Port Scan    |
| Heartbleed                 | Excluded     |

This grouping makes the dataset match the 8 categories used in my FDRL-IDS project.

---

## 5. Heartbleed Problem

During the label mapping, I found that 11 records were not mapped.

The unmapped label was:

```text
Heartbleed
```

I decided not to put Heartbleed into any of the 8 categories because it is a different type of vulnerability and does not properly belong to DoS, DDoS, Brute Force, Web Attack, Infiltration, Botnet, or Port Scan.

So, I excluded these 11 records only from the experimental dataset.

I did not delete them from the original CSV files.

The record numbers are:

```text
Original records       = 2,830,743
Heartbleed records     = 11
Experimental records   = 2,830,732
```

Only 11 records were removed, so this has a very small effect on the total dataset.

---

## 6. Final Class Distribution

After applying the mapping and excluding Heartbleed, I got the following 8 categories:

| Class        |       Records | Percentage |
| ------------ | ------------: | ---------: |
| BENIGN       |     2,273,097 |   80.3007% |
| DoS          |       252,661 |    8.9256% |
| DDoS         |       128,027 |    4.5228% |
| Brute Force  |        13,835 |    0.4887% |
| Web Attack   |         2,180 |    0.0770% |
| Infiltration |            36 |    0.0013% |
| Botnet       |         1,966 |    0.0695% |
| Port Scan    |       158,930 |    5.6144% |
| **Total**    | **2,830,732** |   **100%** |

---

## 7. Class Imbalance

From this result, I can clearly see that the dataset is highly imbalanced.

BENIGN is the largest class:

```text
BENIGN = 2,273,097 records
        = 80.3007%
```

On the other hand, Infiltration has only:

```text
Infiltration = 36 records
             = 0.0013%
```

This is a very large difference.

This is important for my FDRL-IDS project because if I train a model directly on this data, the model may become more biased toward the BENIGN class.

So, I need to consider this class imbalance in the future preprocessing and model training stages.

---

## 8. Class Distribution Graph

I created a graph to visualize the class distribution.

Because the number of records between the largest and smallest classes is very different, I used a logarithmic scale in the graph.

The graph shows clearly that BENIGN traffic is much larger than the attack categories.

The graph is saved here:

```text
docs/figures/day10_class_distribution.png
```

---

## 9. What I Learned Today

From today's work, I learned:

* CICIDS2017 has 15 original labels.
* My project will use 8 main categories.
* Different attack names can be grouped into one common category.
* Heartbleed is not suitable for my 8-class experiment, so I excluded its 11 records.
* The final experimental dataset contains 2,830,732 records.
* BENIGN traffic makes up about 80.30% of the dataset.
* The dataset has a very serious class imbalance.
* Class imbalance will be an important issue in the next stages of my project.

---

## 10. Files Created

I created the following files for Day 10:

### Notebook

```text
notebooks/attack_label_analysis.ipynb
```

This notebook contains my label analysis, mapping, class counting, percentage calculation, and graph generation.

### Graph

```text
docs/figures/day10_class_distribution.png
```

### Report

```text
docs/day10_attack_label_analysis.md
```

---

## 11. Conclusion

Day 10 is completed successfully.

I analyzed the original CICIDS2017 labels and converted them into the 8 categories required for my FDRL-IDS project.

The final experimental dataset contains **2,830,732 records**.

The most important finding from today's work is that the dataset has a **very high class imbalance**, with BENIGN traffic making up **80.3007%** of the data.

Now I have a clear understanding of the labels and their distribution. This will help me in the next stage, where I will start preparing the dataset for preprocessing and federated client simulation.
