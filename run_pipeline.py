"""
CLV Prediction - End-to-End Pipeline
======================================
Orchestrates: Data > EDA > Features > Segments > Models > Explainability

Authors: Sanman, Varsha
"""

import os, sys, time
import pandas as pd

# Ensure src is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data_generator import generate_and_save
from src.eda import run_eda
from src.feature_engineering import build_feature_matrix, get_feature_columns
from src.segmentation import segment_customers
from src.models import train_and_evaluate
from src.explainability import explain_model
from src.monitoring import monitoring_strategy


def main():
    start = time.time()
    print("=" * 60)
    print("  CLV PREDICTION PIPELINE")
    print("=" * 60)

    data_dir = os.path.join(ROOT, "data", "raw")
    models_dir = os.path.join(ROOT, "models")
    reports_dir = os.path.join(ROOT, "reports")

    # -- Step 1: Generate Data -------------------------------------------------
    print("\n[STEP 1/7] Data Generation")
    customers, transactions = generate_and_save(data_dir)

    # -- Step 2: Feature Engineering -------------------------------------------
    print("\n[STEP 2/7] Feature Engineering")
    customers = pd.read_csv(os.path.join(data_dir, "customers.csv"), parse_dates=["signup_date"])
    transactions = pd.read_csv(os.path.join(data_dir, "transactions.csv"), parse_dates=["date"])
    features = build_feature_matrix(customers, transactions)

    # -- Step 3: EDA -----------------------------------------------------------
    print("\n[STEP 3/7] Exploratory Data Analysis")
    run_eda(customers, transactions, features, os.path.join(reports_dir, "figures"))

    # -- Step 4: Customer Segmentation -----------------------------------------
    print("\n[STEP 4/7] Customer Segmentation")
    features = segment_customers(features, os.path.join(reports_dir, "figures"))
    features.to_csv(os.path.join(data_dir, "features_segmented.csv"), index=False)

    # -- Step 5: Model Training ------------------------------------------------
    print("\n[STEP 5/7] Model Training & Evaluation")
    feature_cols = get_feature_columns(features)
    models, results_df, best_model, best_key, splits = train_and_evaluate(
        features, feature_cols, models_dir=models_dir, reports_dir=reports_dir, n_tune_trials=30
    )
    results_df.to_csv(os.path.join(reports_dir, "model_results.csv"), index=False)

    # -- Step 6: Explainability ------------------------------------------------
    print("\n[STEP 6/7] SHAP Explainability")
    X_train, X_test, y_train, y_test = splits
    explain_model(best_model, X_train, X_test, feature_cols,
                  os.path.join(reports_dir, "figures"), model_type="tree")

    # -- Step 7: Monitoring Strategy -------------------------------------------
    print("\n[STEP 7/7] Monitoring Strategy")
    print(monitoring_strategy())

    elapsed = time.time() - start
    print("=" * 60)
    print(f"  PIPELINE COMPLETE - {elapsed:.1f}s")
    print(f"  Outputs: {models_dir}/, {reports_dir}/")
    print(f"  Launch dashboard: streamlit run app/dashboard.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
