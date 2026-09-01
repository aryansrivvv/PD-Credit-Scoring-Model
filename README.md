# Retail Credit Risk: Probability of Default (PD) Model Validation

This repository contains an end-to-end Probability of Default (PD) modeling framework built to industry and regulatory standards. It contrasts a traditional **Logistic Regression Scorecard (Champion)** built with strictly monotonic Weight of Evidence (WOE) bins against a **Gradient Boosting Machine (Challenger)** built with XGBoost.

The project evaluates the critical trade-off between absolute predictive power and regulatory explainability required by the Equal Credit Opportunity Act (ECOA) and the Fair Credit Reporting Act (FCRA).

## Executive Summary

* **Champion Model (Logistic + WOE):** Achieved an **ROC-AUC of 0.8178** and a **KS Statistic of 0.5004**. Selected for production due to strict feature monotonicity, $p < 0.05$ transparency, and clear adverse action extraction.
* **Challenger Model (XGBoost):** Achieved an **ROC-AUC of 0.8694** and a **KS Statistic of 0.5846**. Outperformed the baseline by capturing non-linear feature interactions, but violated regulatory interpretability requirements.
* **Adverse Action Automation:** Implemented SHAP (SHapley Additive exPlanations) game theory values to programmatically extract the top 4 decline reasons for high-risk profiles.

---

## Technical Architecture

### 1. Data Preprocessing & Feature Engineering

* **Dataset:** Kaggle's "Give Me Some Credit" (150,000 borrowers, 6.7% default rate).
* **Missing Data Strategy:** Missing financial indicators (e.g., `MonthlyIncome`) were imputed using the median from the training fold to prevent data leakage, and accompanied by explicit `is_missing` boolean flags, as missingness is highly predictive of credit distress.
* **Information Value (IV) & Weight of Evidence (WOE):** Continuous variables were mapped to monotonic quantile bins to neutralize extreme outliers and enforce linear relationships. Features with an $IV < 0.02$ were dropped from the scorecard to prevent target leakage and overfitting.

### 2. Champion Model (Statsmodels Logistic Regression)

Built using `statsmodels.api` to satisfy risk committee requirements for full statistical transparency.

* All retained features demonstrated a $p$-value of $0.000$, ensuring statistical significance.
* WOE-transformed variables strictly enforced monotonic "common sense" constraints (e.g., as debt-to-income increases, the assigned risk score strictly increases).

### 3. Challenger Model (XGBoost)

Trained on the raw, un-binned data to benchmark the maximum theoretical predictive power of the dataset.

* Handled the 6.7% class imbalance using the `scale_pos_weight` parameter rather than synthetic oversampling (SMOTE) to preserve the true distribution of the portfolio.
* Captured complex risk interactions (e.g., low income paired with a specific utilization threshold) that a linear scorecard misses.

---

## Model Validation Metrics

| Metric | Champion (Logistic + WOE) | Challenger (XGBoost) | Delta |
| --- | --- | --- | --- |
| **ROC-AUC** | 0.8178 | 0.8694 | + 0.0516 |
| **KS Statistic** | 0.5004 | 0.5846 | + 0.0842 |
| **Monotonicity** | Strictly Enforced | Unconstrained | N/A |
| **Explainability** | High (Coefficients) | Low (Requires SHAP) | N/A |

---

## Regulatory Independent Validation Report

While the XGBoost challenger demonstrates a superior KS statistic and captures roughly 8% more separation between Goods and Bads, deploying unconstrained machine learning poses operational risks:

1. **Adverse Action Non-Compliance:** Lenders must issue clear decline reasons. XGBoost lacks intrinsic coefficients, relying instead on post-hoc explainers like SHAP. SHAP can generate unstable local explanations across similar credit profiles, risking ECOA violations.
2. **Violation of Monotonicity:** Unconstrained trees can create jagged relationships. The Challenger may logically infer that a 40% debt ratio is risky, 42% is safe, and 45% is risky again due solely to training data noise.
3. **Macroeconomic Overfitting:** Because XGBoost relies heavily on fine-grained historical interactions, its predictive power (measured by Population Stability Index) is expected to degrade much faster than a simpler, generalized Logistic Scorecard during macroeconomic shifts.

**Recommendation:** The Logistic Regression scorecard should be deployed as the primary decision engine. The XGBoost model should be retained as a parallel shadow-benchmark to monitor scorecard degradation and identify overlapping segments where the primary model historically underperforms.

---

## Installation & Usage

### Prerequisites

* Python 3.9+
* Jupyter Notebook

### Setup

1. Clone the repository:
```bash
git clone https://github.com/aryansrivvv/PD-Credit-Scoring-Model.git
cd PD-Credit-Scoring-Model

```


2. Install dependencies:
```bash
pip install pandas numpy scikit-learn statsmodels xgboost shap

```


3. Download the dataset `cs-training.csv` from [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place it in the root directory.
4. Launch the notebook:
```bash
jupyter notebook

```
