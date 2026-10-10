# PD Credit Scoring Model

An end-to-end Probability of Default (PD) scorecard pipeline built on the "Give Me Some Credit" dataset.

This repository holds a production-grade workflow for evaluating retail borrower risk. It contrasts a traditional **Logistic Regression Scorecard (Champion)** built with strictly monotonic Weight of Evidence (WOE) bins against an unconstrained **Gradient Boosting Machine (Challenger)** built with XGBoost.

The project evaluates the critical trade-off between absolute predictive power and regulatory explainability required by laws like the Equal Credit Opportunity Act (ECOA) and the Fair Credit Reporting Act (FCRA).

## Executive Summary
* **Champion Model (Logistic + WOE):** Achieved an **ROC-AUC of 0.8574** and a **KS Statistic of 0.5633**. Selected for production due to strict feature monotonicity, statistical transparency (all $p$-values < 0.05), and robust calibration.
* **Challenger Model (XGBoost):** Achieved an **ROC-AUC of 0.8664** and a **KS Statistic of 0.5767**. While it slightly outperformed the baseline by capturing non-linear interactions, it lacks the intrinsic explainability required for automated adverse action reporting.
* **Outcome:** The 0.009 AUC gap was deemed sufficiently small to justify deploying the interpretable Champion model, retaining the Challenger strictly as a shadow benchmark.

## Results & Performance Metrics (Test Set)

| Metric | Champion (Logistic Scorecard) | Challenger (XGBoost) | Delta |
| --- | --- | --- | --- |
| **ROC-AUC** | 0.8574 | 0.8664 | + 0.0090 |
| **Gini** | 0.7148 | 0.7328 | + 0.0180 |
| **KS Statistic** | 0.5633 | 0.5767 | + 0.0134 |
| **Monotonicity** | Strictly Enforced | Unconstrained | N/A |
| **Explainability** | High (Coefficients) | Low (Requires SHAP) | N/A |

*For more detailed insights into the methodology, validation splits, and calibration, please see the [MODEL_CARD.md](MODEL_CARD.md).*

## Pipeline Highlights
- **Interpretability via WOE**: Converts continuous signals into step-wise risk scores. All bins strictly adhere to monotonic constraints to ensure logical business rules (e.g., higher debt ratio strictly increases risk).
- **Deeper Analysis & Data Quality**: Addressed complex overlaps in missing data attributes and anomalous sentinel codes, culminating in robust exclusion rules and neutral penalty mapping.
- **Intrinsic Calibration**: Avoids complex post-processing calibration (like Isotonic regression) by validating native predicted log-odds slopes on the test set.
- **Manual Routing**: Identifies structural outliers (e.g., 96/98 past-due records) and dynamically diverts them out of the automated ML decisioning flow for manual review.

## Project Structure
- `Data_Preprocessing.ipynb`: The master end-to-end pipeline handling data ingestion, exploratory deeper analysis, monotonic binning constraints, missing-data overrides, and comprehensive model evaluations.
- `model_config.json`: The fully serialized WOE dictionary and intercept/coefficient weights for standalone deployment.
- `MODEL_CARD.md`: A detailed report on model methodology, performance metrics, calibration limits, and deployment guidelines.
- `cs-training.csv` & `cs-test.csv`: The historical credit data files.
- `requirements.txt`: The strict Python dependency map to recreate the isolated environment.

## Running Locally

To run the pipeline in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Data_Preprocessing.ipynb
```
