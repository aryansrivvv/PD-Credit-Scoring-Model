# PD Credit Scoring Model

An end-to-end Probability of Default (PD) scorecard pipeline built on the "Give Me Some Credit" dataset.

This repository holds the production-grade workflow for evaluating retail borrower risk using monotonic Weight of Evidence (WOE) binning and Logistic Regression, heavily evaluated against an unconstrained XGBoost challenger.

## Project Structure
- `Data_Preprocessing.ipynb`: The master end-to-end pipeline handling data ingestion, exploratory deeper analysis, monotonic binning constraints, missing-data overrides, and comprehensive model evaluations.
- `model_config.json`: The fully serialized WOE dictionary and intercept/coefficient weights for standalone deployment.
- `cs-training.csv` & `cs-test.csv`: The historical credit data files.
- `MODEL_CARD.md`: A detailed report on model methodology, performance metrics, calibration limits, and deployment guidelines.
- `requirements.txt`: The strict Python dependency map to recreate the isolated environment.

## Pipeline Highlights
- **Interpretability via WOE**: Converts continuous signals into step-wise risk scores. All bins strictly adhere to monotonic constraints to ensure logical business rules (e.g. higher utilization strictly increases risk).
- **Deeper Analysis**: Following extensive exploratory analysis, we identified complex overlaps in missing data attributes and anomalous sentinel codes, culminating in robust exclusion rules and neutral penalty mapping. 
- **Calibration Check**: Avoids complex post-processing calibration (like Isotonic regression) by validating native predicted log-odds slopes.
- **Manual Routing**: Identifies structural outliers (e.g., 96/98 past-due records) and dynamically diverts them out of the automated ML decisioning flow.

## Running Locally

To run the pipeline in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Data_Preprocessing.ipynb
```
