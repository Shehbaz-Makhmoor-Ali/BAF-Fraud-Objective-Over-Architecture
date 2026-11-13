# 🏦 Bank Account Fraud Detection: Replicating "Objective over Architecture"

**Researcher:** Shehbaz Makhmoor Ali (BS Data Science)  
**Institution:** Air University, Islamabad  

## 📌 Project Overview
This repository contains the independent replication of the baseline methodology proposed in the December 2025 MDPI research paper: *"Objective over Architecture: Fraud Detection Under Extreme Imbalance in Bank Account Opening."* 

**Reference Paper:** The original MDPI research paper (`MDPI_Objective_over_Architecture_2025.pdf`) is included in the root directory of this repository for academic reference and comparative review.

The project utilizes the **NeurIPS 2022 Bank Account Fraud (BAF) Dataset**, which contains 1,000,000 synthetic banking applications with an extreme **1.1% fraud prevalence** (99-to-1 class imbalance). The objective of this initial milestone (`v1.0`) is to mathematically validate the paper's core thesis: when dealing with extreme tabular imbalance, applying a static cost-sensitive penalty to the objective function is highly effective, yielding a strong Recall rate for detecting fraudulent applications.

## ⚙️ Methodology & Preprocessing Pipeline
To accurately replicate the paper's findings, the following strict preprocessing pipeline was implemented:

1. **Data Partitioning:** Stratified 80/20 split (800,000 train / 200,000 test) preserving the exact 1.1% fraud ratio.
2. **Imputation:** 
   - *Numerical:* Median imputation (replacing `-1` missing indicators).
   - *Categorical:* Constant imputation (replacing missing values with `__missing__`).
3. **Encoding & Scaling:** One-Hot Encoding for categorical variables (`device_os`, `housing_status`, etc.) and Standard Scaling for numerical variance.
4. **The Objective (Class Weighting):** A calculated static penalty factor of **89.67** was applied to the loss function to heavily penalize False Negatives (missed fraud).

## 📊 Replication Results (`v1.0` Baseline)

The primary evaluation metric prioritized is **PR-AUC (Precision-Recall Area Under Curve)**, as standard Accuracy is misleading on 99-to-1 imbalanced data.

| Architecture | Objective Penalty | PR-AUC | Fraud Recall | Precision |
| :--- | :--- | :---: | :---: | :---: |
| **Deep Learning (GRU)** | Static (89.67) | **0.1548** | **81.0%** | 4.0% |
| **LightGBM** | Static (89.67) | 0.1482 | 79.0% | 4.0% |

### 🏆 Key Findings
The baseline replication was completely successful. By applying the `89.67` static penalty to a Gated Recurrent Unit (GRU) neural network, the model successfully identified **81% of all synthetic fraudsters**. The precision rate (4%) mirrors the original paper's findings, reflecting the fundamental precision-recall trade-off required to prevent severe financial losses in an operational banking environment.

## 🔮 Future Work & Next Research Phases
Having established and validated the supervised replication baseline (`0.1548` PR-AUC), upcoming phases of this project will investigate novel strategies to overcome extreme class imbalance without sacrificing operational precision:

* **Alternative Learning Paradigms:** Exploring methods that do not rely solely on heavily weighted supervised binary classification.
* **Synthetic Minority Modeling:** Investigating data-level techniques to better represent rare fraud patterns during training.
* **Feature & Threshold Optimization:** Evaluating domain-specific interaction features and dynamic decision thresholds to improve the precision-recall trade-off.

## 📂 Repository Structure
* `BAF_Replication.ipynb` — Complete Jupyter Notebook with EDA visualizations, preprocessing, and model evaluation outputs.
* `baf_replication.py` — Clean, standalone Python script for local or cloud execution.
* `MDPI_Objective_over_Architecture_2025.pdf` — Original reference research paper.
* `requirements.txt` — Required Python dependencies.

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Shehbaz-Makhmoor-Ali/BAF-Extreme-Imbalance-Fraud-Detection.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the NeurIPS 2022 BAF dataset (`Base.csv`) and place it in the root directory (or update the file path in the script), then run `BAF_Replication.ipynb` or `baf_replication.py`.