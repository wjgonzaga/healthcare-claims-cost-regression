# healthcare-claims-cost-regression

📊 Healthcare Claims Cost Regression

End-to-end machine learning pipeline to estimate healthcare claim cost using demographic, plan, and utilization features, with business-focused evaluation and interpretation.

I. Business Context:

Accurate healthcare cost forecasting supports:

1.Financial planning & reserves
2.Risk stratification and care management
3.Operational resource allocation
4.Value-based contract performance

This project demonstrates how a baseline predictive model can be built, evaluated, and interpreted in a way that aligns with real healthcare analytics decision-making.

II. Objective:
Build a reproducible baseline regression model to estimate allowed claim cost and evaluate predictive performance on held-out data, while translating results into meaningful operational insight.

III. Data Description:
1. Synthetic claims-like dataset (no PHI; safe for public use)
2. ~20,000 records
3. Feature groups:
- Demographics: age, sex, region
- Plan attributes: plan type
- Utilization: ER visits, inpatient admits, outpatient visits, prescription fills, chronic condition count
4. Right-skewed cost distribution with multiplicative noise to reflect real healthcare spending behavior

IV. Methodology

1. Generate synthetic claims dataset
2. Apply log transformation to stabilize heavy-tailed cost distribution
3. Encode categorical variables via one-hot encoding
4. Train Ridge regression baseline within a scikit-learn Pipeline
5. Evaluate using:
     - MAE (average dollar error)
     - RMSE (sensitivity to large misses)
     - R² (variance explained)
6. Produce predicted vs. actual visualization as a diagnostic artifact

V. Results:

Baseline Ridge regression on the synthetic dataset:
MAE: $1,089
RMSE: $4,604
R²: –1.31
The negative R² shows that a simple linear model cannot adequately explain the large variability in healthcare costs.
This is common in medical spending, where a small number of very high-cost cases drive overall totals.
Future improvements would evaluate nonlinear models to better capture these complex cost patterns.

VI. Key Insights

Healthcare costs exhibit strong skew and tail risk, challenging simple linear models.

Baseline regression provides interpretability and benchmarking, even when performance is limited.

Error Analysis and Tail Sensitivity (analysis.py)

Healthcare costs were highly uneven, with the largest claim about 27× the median.

Most large prediction errors occurred in a small number of extreme high-cost cases, where the linear baseline significantly overestimated spending. These few tail observations dominated squared error and led to the overall negative R².

When the top 1% highest-cost members were removed for diagnostic analysis, model performance improved to:

R² ≈ 0.28

This suggests the baseline model captures meaningful cost variation for the majority of members, but struggles with catastrophic outliers, a common challenge in healthcare cost modeling.

VII. Real-world improvement likely requires:

Nonlinear models
Risk stratification classification
Temporal utilization features
