# DeepAudit: Advanced Credit Card Fraud Detection Pipeline

DeepAudit is an end-to-end machine learning and deep learning repository designed to detect highly anomalous, fraudulent credit card transactions. Operating on a highly imbalanced dataset (~0.17% fraud rate), this project implements a hybrid architecture utilizing classical tree-based ensembles alongside a 1D Convolutional Neural Network (Conv1D) to extract spatial localized features across transaction patterns.

## Core Architecture & Methodology

Detecting credit card fraud presents a massive class imbalance challenge. A naive model predicting all transactions as legitimate would achieve an accuracy of 99.83% while completely failing its objective. This project addresses this through a robust, multi-tiered pipeline:

1. **Synthetic Minority Over-sampling Technique (SMOTE):** To prevent data leakage, the pipeline splits data strictly into training and evaluation sets before synthesizing artificial fraudulent records. SMOTE balances the training distribution so the neural network can map minority boundaries effectively.
2. **Robust Scaler Preprocessing:** Outliers in transaction `Amount` and temporal patterns in `Time` are transformed using robust statistics (median and quantiles) to prevent exploding gradients in dense and spatial layers.
3. **Random Forest Baseline:** An ensemble of decision trees establishes a rigorous baseline for feature importance and non-linear threshold partitions across the PCA-transformed features ($V_1$ to $V_{28}$).
4. **1D Convolutional Neural Network (Conv1D):** By treating the tabular transaction features as a contiguous sequence, the 1D CNN slides receptive fields across the dimensions to pick up complex, localized coordinate relationships that traditional multi-layer perceptrons miss.

---

## Dataset Specifications

The framework utilizes the classic **MLG-ULB Credit Card Fraud Detection Dataset**, containing transactions made by European cardholders.
- **Total Transactions:** 284,807
- **Target Variable (`Class`):** `0` for Legitimate, `1` for Fraudulent
- **Total Fraud Events:** 492 transactions (0.172%)
- **Features:** `Time`, `Amount`, and 28 principal components ($V_1$ through $V_{28}$) obtained via PCA for privacy preservation.

---

## Repository Structure

```text
├── Credit_Card_Fraud_Detection.ipynb   # Core project notebook with complete pipeline
├── README.md                           # Detailed project documentation
