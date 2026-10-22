# Explainable and Cost-Sensitive E-Commerce Fraud Detection Using Machine Learning

**CSE303 Machine Learning Capstone Project**  
**Dataset Source:** [https://www.kaggle.com/competitions/ieee-fraud-detection/data](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

---

## 1. Project Overview & Problem Framing

In online e-commerce transactions, identifying fraudulent payments is a critical challenge. Unlike standard, balanced classification problems encountered in introductory machine learning, real-world fraud detection involves three foundational complexities:

1. **Extreme Class Imbalance:**  
   In production payment streams, legitimate transactions vastly outnumber fraudulent ones. A naive classifier predicting "Legitimate" for 100% of transactions can achieve a deceptively high raw accuracy score while failing to detect a single fraudulent transaction. As a result, standard accuracy is an inappropriate metric. Evaluation must rely on metrics designed for imbalanced data, such as Precision, Recall, Confusion Matrices, the F1-Score, and Precision-Recall Area Under the Curve (PR-AUC).

2. **Asymmetric Error Costs (Cost-Sensitivity):**  
   Standard classification algorithms typically treat false alarms and missed detections as having identical consequences. In payment fraud, the business impact is fundamentally asymmetric:
   - **False Negative (FN):** The system fails to detect an actual fraudulent transaction, resulting in direct financial loss, dispute fees, and chargeback penalties.
   - **False Positive (FP):** The system falsely flags a legitimate customer transaction, leading to customer friction, cart abandonment, and operational manual review costs.
   - *Hypothetical Illustration:* To conceptualize this asymmetry, one might hypothetically consider an undetected fraud incurring a $500 loss versus a false alarm review costing $5. These figures are **strictly hypothetical illustrations** for conceptual clarity, not verified business facts. In this project, formal cost-sensitive optimization will only be claimed once an explicit, domain-justified cost matrix is formulated in later stages.

3. **Interpretability & Model Explainability:**  
   Payment fraud systems operate in highly regulated environments where automated decisions must be explainable to risk analysts, compliance officers, and customers. While initial baseline models (such as shallow Decision Trees) provide basic split rules and feature importances, **Decision Tree feature importance alone does not fulfill the project's explainability requirement**. Our full workflow will incorporate **SHAP (SHapley Additive exPlanations)** to provide rigorous, game-theoretic local explanations (e.g., individual transaction waterfall/force plots) and global feature attribution summaries.

---

## 2. CSE303 Capstone Requirements & Overall Workflow

To satisfy the academic objectives of the CSE303 Machine Learning capstone, the project encompasses a comprehensive end-to-end machine learning lifecycle:

1. **Exploratory Data Analysis (EDA):** Memory-conscious inspection of shapes, data types, empirical class distributions, missingness profiles, and univariate/bivariate feature relationships.
2. **Data Cleaning & Preprocessing:** Thoughtful missing-value treatment (treating high missingness as exploratory flags rather than blunt automatic drops), frequency/target encoding for high-cardinality categoricals, and memory optimization (downcasting numerical types).
3. **Interpretable Baseline Modeling:** Training an initial baseline model (e.g., Decision Tree) to establish benchmark performance on imbalanced data.
4. **Multiple Syllabus-Aligned Model Comparisons:** Comparing several distinct families of algorithms studied in the curriculum (e.g., Logistic Regression, Decision Trees, Random Forest, and Gradient Boosted Trees).
5. **Rigorous Evaluation on Suitable Metrics:** Evaluating all models using Confusion Matrices, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
6. **Robust Benchmarking:** Evaluating stability across multiple random seeds or cross-validation where computationally feasible.
7. **Hyperparameter Tuning:** Systematic hyperparameter optimization for the top two performing models.
8. **In-Depth Error Analysis:** Inspecting patterns in False Positives and False Negatives to identify failure modes and boundary cases.
9. **Cost-Sensitive Threshold Analysis:** Finding optimal decision probability thresholds under explicit, justified cost assumptions rather than defaulting to standard 0.5 cutoffs.
10. **SHAP Explainability:** Implementing SHAP to explain both global feature contributions and local, transaction-level predictions.
11. **Deployment & Demonstration:** Building an interactive demo application (e.g., a lightweight web interface) for testing transaction inputs.
12. **Capstone Deliverables:** Compiling the final academic project report, presentation poster, and live demonstration.

---

## 3. Dataset Information & File Preparation

This project uses data from the **IEEE-CIS Fraud Detection Competition**:  
[https://www.kaggle.com/competitions/ieee-fraud-detection/data](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

### Dataset Architecture
The dataset is partitioned across two main tables, joined by the unique primary key `TransactionID`:

1. **Transaction Table (`train_transaction.csv`):**  
   Contains transaction-level details including `TransactionID`, `isFraud` (ground-truth binary target), `TransactionDT` (timedelta from a reference point), `TransactionAmt` (payment amount in USD), `ProductCD` (product code), card information (`card1`–`card6`), address information (`addr1`, `addr2`), distance (`dist1`, `dist2`), email domains (`P_emaildomain`, `R_emaildomain`), and masked numeric features (`C1`–`C14`, `D1`–`D15`, `M1`–`M9`, `V1`–`V339`).

2. **Identity Table (`train_identity.csv`):**  
   Contains identity, network, and device attributes collected for transactions where identity tracking was available or triggered: network connection data (IP, proxy), device types, operating systems, browsers, and masked variables (`id_01`–`id_38`, `DeviceType`, `DeviceInfo`). Not all transactions have corresponding identity records.

### Important Data Placement Notice
- **Files Required for Initial EDA and Supervised Modeling:**  
  Only **`train_transaction.csv`** and **`train_identity.csv`** are required.  
  The competition test files (`test_transaction.csv` and `test_identity.csv`) are unlabeled sets intended strictly for Kaggle leaderboard submissions. They are **not required** for our project's exploratory data analysis, supervised training, validation, or holdout testing (which will be partitioned from the labeled training data).
- **Manual Download:**  
  Download `train_transaction.csv` and `train_identity.csv` manually from Kaggle and place them into `data/raw/`:
  ```text
  data/raw/
  ├── train_transaction.csv
  └── train_identity.csv
  ```
- **Memory Consideration:**  
  The raw CSV files are large tabular datasets that expand significantly in RAM when parsed by `pandas`. Exploration scripts should use chunking, sampling (`nrows`), or explicit downcasting to avoid memory exhaustion.
- **Git Exclusion:**  
  All raw and processed dataset files (`*.csv`, `*.zip`, `*.parquet`, `*.feather`) are strictly excluded from version control via `.gitignore` and must never be committed to Git.

---

## 4. Project Directory Structure

```text
ML_FRAUD_DETECTION/
│
├── .gitignore                   # Excludes raw/processed data, model artifacts, caches, and venvs
├── README.md                    # Project documentation, capstone workflow, and setup instructions
├── requirements.txt             # Initial Python dependencies for data manipulation and EDA
│
├── data/
│   ├── raw/                     # Original IEEE-CIS CSV files (local only, untracked)
│   └── processed/               # Cleaned or sampled datasets for experiments (local only, untracked)
│
├── notebooks/
│   └── 01_eda.ipynb             # Stage 1: Memory-conscious starter EDA notebook
│
└── src/
    └── __init__.py              # Python package for modular scripts in later stages
```

---

## 5. Technical Implementation Roadmap

> **Note on Scope:**  
> The following six-stage engineering roadmap outlines the technical progression of the codebase. It represents a **simplified technical roadmap** designed for code staging and **does not replace the full CSE303 capstone requirements** outlined in Section 2.

| Stage | Engineering Milestone | Key Technical Activities |
| :--- | :--- | :--- |
| **Stage 1 (Current)** | **Project Setup & EDA Planning** | Establish repository structure, `.gitignore`, dependency configuration, and a memory-conscious starter notebook with file verification. |
| **Stage 2** | **Exploratory Data Analysis (EDA)** | Deep dive into individual table shapes, dynamic `isFraud` distribution, exploratory missingness flags (50% and 80%), and transaction amount / product distributions. |
| **Stage 3** | **Data Cleaning & Feature Engineering** | Thoughtful missing-value treatment (avoiding automatic dropping), frequency encoding for high-cardinality categoricals, temporal feature extraction, and memory downcasting. |
| **Stage 4** | **Baseline Models & Multi-Model Comparison** | Train interpretable Decision Tree baseline; benchmark syllabus-aligned models (Logistic Regression, Random Forest, etc.); formulate explicit cost matrix and evaluate PR-AUC. |
| **Stage 5** | **Tuning, Error Analysis & SHAP Explainability** | Hyperparameter tuning of top two models; in-depth false-positive and false-negative error analysis; local and global SHAP explainability. |
| **Stage 6** | **Deployment & Final Capstone Defense** | Build interactive demo interface; complete project report, evaluation summary, and presentation poster. |

---

## 6. Getting Started (Windows Setup Guide)

### Step 1: Open Terminal in Project Directory
Open PowerShell or your preferred terminal in the project root:
```powershell
cd d:\ML_FRAUD_DETECTION
```

### Step 2: Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell script execution is restricted on your system, enable it for the current session with: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)*

**Linux / macOS (Reference):**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Baseline Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Add the Labeled Dataset Files
Download `train_transaction.csv` and `train_identity.csv` from [https://www.kaggle.com/competitions/ieee-fraud-detection/data](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and place them into:
```text
data/raw/
```
*(Do not download test files; they are unlabeled and not used in this project.)*

### Step 5: Launch the EDA Notebook
```powershell
jupyter notebook notebooks/01_eda.ipynb
```

---

## 7. Academic Integrity & Project Standards

- **No Fabricated Data or Metrics:** All experimental results, confusion matrices, and performance scores must be generated dynamically from actual model execution in later stages.
- **Hypothetical Cost Basis:** Any monetary cost figures used in early stages are strictly pedagogical illustrations until a formal, domain-justified cost matrix is explicitly defined in Stage 4.
- **Explainability Standard:** Explainability will not be reduced to basic Decision Tree feature importance; it will be formally analyzed using SHAP game-theoretic attributions.
