# Customer Lifetime Value (CLV) Analytics Platform

A production-grade, end-to-end Machine Learning pipeline and interactive dashboard for predicting Customer Lifetime Value. This project transitions CLV from a theoretical exercise into an actionable business system designed to optimize Customer Acquisition Cost (CAC) and drive proactive retention.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat-square&logo=streamlit)
![Lifetimes](https://img.shields.io/badge/Lifetimes-0.11+-purple?style=flat-square)

---

## Authors

| Name | Role |
|------|------|
| **Sanman** | Data Analyst |
| **Varsha** | Data Analyst |

---

## The Business Problem

E-commerce companies often struggle to allocate marketing budgets efficiently. By predicting **Customer Lifetime Value (CLV)** over a 6-month horizon, businesses can:

- **Optimize Customer Acquisition Cost (CAC):** Reallocate ad spend toward channels that acquire high-LTV users.
- **Proactive Retention:** Identify "At-Risk" segments before churn occurs.
- **Focus Retention Spend:** Implement the 80/20 rule by offering VIP perks exclusively to the "Champions" segment.

> A 5% improvement in the retention of top-decile customers can increase overall profits by **25-95%** (Harvard Business Review).

---

## Dataset and Architecture

### The Data
The pipeline generates a highly realistic synthetic dataset comprising **50,000 customers** and over **350,000 transactions** across a 3-year observation window.
- **Realism Factors:** Pareto-distributed spending (whale behavior), seasonal spikes (Q4 peaks), and age/category correlations.

### Project Structure
```
clv-prediction/
├── README.md
├── requirements.txt
├── run_pipeline.py             # End-to-end orchestrator
├── data/raw/                   # Generated datasets and features
├── src/                        # Modular source code
│   ├── data_generator.py
│   ├── feature_engineering.py  # RFM + 15 advanced features (leakage-free)
│   ├── segmentation.py         # K-Means customer segmentation
│   ├── models.py               # Optuna-tuned ML + Probabilistic models
│   ├── eda.py
│   ├── explainability.py
│   └── monitoring.py
├── models/                     # Serialized artifacts (XGBoost, BG/NBD, etc.)
├── reports/                    # Model results and visualizations
│   └── figures/
├── notebooks/                  # Jupyter notebooks
│   └── CLV_Analysis.ipynb
└── app/
    └── dashboard.py            # Streamlit multi-tab business dashboard
```

---

## Methodology and Rigor

Most beginner CLV projects suffer from massive **Data Leakage** by predicting historical revenue using historical order values. This project strictly mitigates this issue and implements industry-standard evaluation.

### 1. Data Leakage Mitigation
- **Calibration / Holdout Split:** The dataset is split temporally at `reference_date - 180 days`.
- **Target Variable:** CLV is defined strictly as the revenue generated *only* during the 6-month holdout period.
- **Features:** RFM and behavioral features are calculated *only* on the calibration period, guaranteeing the model predicts future behavior without peeking at the target.

### 2. Model Comparison
We benchmark modern Machine Learning algorithms against traditional Statistical/Probabilistic baselines:

| Model | Type | Tuning / Approach |
|-------|------|--------|
| **XGBoost** | Advanced ML | Optuna (30 trials) |
| **LightGBM** | Advanced ML | Optuna (30 trials) |
| **Random Forest** | Intermediate ML | Sklearn Default |
| **Ridge Regression**| Baseline ML | L2 Regularization |
| **BG/NBD + Gamma-Gamma** | Probabilistic | `lifetimes` / Buy-Till-You-Die (BTYD) |

### 3. Evaluation Metrics
Models are evaluated on the holdout period using:
- **RMSE** (Root Mean Squared Error) - Heavily penalizes large prediction errors on "whales".
- **MAE** (Mean Absolute Error) - Business-friendly interpretation of average dollar error.
- **R-squared** (Coefficient of Determination) - Explains variance captured by the model.

---

## Actionable Business Insights

The Streamlit dashboard surfaces direct recommendations based on the data:

1. **High-Value Cohorts:** The "Champions" segment represents a disproportionate amount of future revenue. Prioritize retention budgets for this tier.
2. **Churn Prevention:** Customers who cross the 90-day inactivity threshold rarely return. Deploy automated win-back campaigns at day 60.
3. **Channel Optimization:** "Referral" and "Organic Search" yield higher lifetime values. Reallocate CAC to these top-performing channels.

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd clv-prediction
pip install -r requirements.txt

# 2. Run the full pipeline
python run_pipeline.py

# 3. Launch the business dashboard
streamlit run app/dashboard.py
```

---

## Future Work

To further evolve this system into an enterprise-grade platform:

- **Deep Learning Integration:** Implement LSTM or Transformer architectures to capture sequential purchase patterns rather than relying solely on aggregated RFM features.
- **Real-Time Inference:** Containerize the prediction API using FastAPI/Docker to serve predictions at the point of checkout.
- **Cloud Deployment:** Migrate the data generation and storage to AWS S3 / Snowflake, and schedule pipeline retraining using Apache Airflow.
- **Causal Inference:** Integrate uplift modeling to determine if a marketing intervention actually *caused* a purchase, rather than just predicting organic behavior.

---

