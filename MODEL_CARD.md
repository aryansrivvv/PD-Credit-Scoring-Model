# Model Card: Give Me Some Credit (PD Scorecard)

## 1. Purpose
This model predicts the Probability of Default (PD) for retail credit customers—specifically, the likelihood that a customer will experience financial distress (90 days past due or worse) within the next two years. It is designed to act as an interpretable logistic regression scorecard for risk pricing, provisioning, and automated decision-making.

## 2. Data
- **Target:** `SeriousDlqin2yrs` (Binary: 1 = Default, 0 = Non-Default).
- **Population:** Retail borrowers with credit and demographic data. 
- **Training Set:** 150,000 historical records (split into 95,823 Train / 23,963 Validation / 29,945 Test).
- **Features Analyzed:** 10 continuous numerical features (utilization, late-payment counts, age, debt ratio, income, and open credit lines/loans).

## 3. Methodology
- **Preprocessing:** Deeper analysis revealed data quality issues requiring imputation, binning, and outlier handling. Features were mapped into Weight of Evidence (WOE) monotonic bins to ensure business-compliant scorecards. 
- **Missing Data Policy:** Missing bins for `MonthlyIncome`, `DebtRatio`, and `NumberOfDependents` are scored neutrally (WOE = 0). Missing values in utilization, late-payment counts, or age are routed to manual review.
- **Model Selection:** A highly interpretable Logistic Regression WOE Scorecard was chosen as the Champion over a gradient-boosted Challenger (XGBoost).
- **Feature Selection:** Iterative backward pruning based on p-values (p ≤ 0.05) and positive coefficient penalties. The final model retains 8 independent features with strict negative parameter signs.

## 4. Performance (Test Set)
- **Champion (Logistic Scorecard):** AUC = 0.8574 | Gini = 0.7148 | KS = 0.5633
- **Challenger (XGBoost):** AUC = 0.8664 | Gini = 0.7328 | KS = 0.5767
- **Gap:** The 0.009 AUC gap between the constrained scorecard and the unconstrained benchmark is statistically significant (95% CI: 0.0068 - 0.0112) but sufficiently small to justify deploying the interpretable scorecard.
- **5-Fold Cross Validation:** Champion AUC 0.8516 ± 0.0083 vs Challenger 0.8610 ± 0.0077.
- **Risk Ranking:** Score bands are strictly monotonic. The riskiest 10% of customers exhibit a 35.6% default rate, capturing over 54% of all bad loans.

## 5. Calibration
- **Platt Scaling Slopes:** 0.985 (Validation) / 1.023 (Test), proving strong intrinsic calibration without post-processing calibration algorithms (isotonic recalibration proved computationally redundant).
- **Portfolio Averages:** Mean predicted PD = 6.56%, closely matching the 6.60% observed bad rate on the hold-out test set. 
- **Conservative Tails:** The four safest score bands overpredict default rates by roughly 1.7x (expected 165 bads vs. observed 97), creating a conservative safety margin for prime risk evaluation.

## 6. Limitations & Considerations
- **No Out-of-Time Validation:** Performance was tested on a random stratified holdout rather than an out-of-time sample. Real-world performance might degrade slightly.
- **Regulatory Compliance (`age` and `dependents`):** The inclusion of applicant age and dependent counts necessitates rigorous compliance reviews (e.g., ECOA / Regulation B in the US) to prevent illegal discrimination.
- **Data Skew:** The middle bin for `DebtRatio` is quite wide, absorbing roughly 40% of the applicant distribution. 

## 7. Operational Monitoring
- **Manual Review Pipeline:** Sentinel codes (values of 96/98 in past-due features) constitute ~0.18% of applicants but hold a severe 54.5% default rate. These are flagged for immediate manual underwriting (a policy rule should dictate how to handle them manually).
- **PSI Tracking:** Initial Population Stability Index (PSI) between the train and test splits is near-zero (0.0006). Ongoing monitoring is required to detect feature drift in production data.
