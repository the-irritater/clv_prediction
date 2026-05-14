"""
Feature Engineering
====================
Transforms raw customer + transaction data into ML-ready features.
Includes RFM core, advanced behavioral features, and encoding.

Authors: Sanman, Varsha
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def compute_rfm(transactions: pd.DataFrame, reference_date: pd.Timestamp = None) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary features per customer."""
    if reference_date is None:
        reference_date = transactions["date"].max() + pd.Timedelta(days=1)

    rfm = transactions.groupby("customer_id").agg(
        recency=("date", lambda x: (reference_date - x.max()).days),
        frequency=("transaction_id", "nunique"),
        monetary=("amount", "sum"),
    ).reset_index()

    return rfm


def compute_advanced_features(customers: pd.DataFrame,
                                transactions: pd.DataFrame,
                                reference_date: pd.Timestamp = None) -> pd.DataFrame:
    """Build the full feature matrix with domain-specific features."""
    if reference_date is None:
        reference_date = transactions["date"].max() + pd.Timedelta(days=1)
        
    # FIX: Data Leakage Prevention (Split into calibration and holdout)
    cutoff_date = reference_date - pd.Timedelta(days=180) # 6 months holdout
    calib_txns = transactions[transactions["date"] < cutoff_date]
    holdout_txns = transactions[transactions["date"] >= cutoff_date]

    # -- RFM -------------------------------------------------------------------
    rfm = compute_rfm(calib_txns, cutoff_date)

    # -- Transaction-level aggregations ----------------------------------------
    txn_agg = calib_txns.groupby("customer_id").agg(
        avg_order_value=("amount", "mean"),
        max_order_value=("amount", "max"),
        min_order_value=("amount", "min"),
        order_std=("amount", "std"),
        total_quantity=("quantity", "sum"),
        avg_quantity=("quantity", "mean"),
        avg_discount=("discount_pct", "mean"),
        max_discount=("discount_pct", "max"),
        n_transactions=("transaction_id", "count"),
    ).reset_index()
    txn_agg["order_std"] = txn_agg["order_std"].fillna(0)

    # -- Category diversity ----------------------------------------------------
    cat_diversity = calib_txns.groupby("customer_id")["product_category"].nunique().reset_index()
    cat_diversity.columns = ["customer_id", "category_diversity"]

    # -- Temporal features -----------------------------------------------------
    first_last = calib_txns.groupby("customer_id")["date"].agg(["min", "max"]).reset_index()
    first_last.columns = ["customer_id", "first_purchase", "last_purchase"]
    first_last["days_since_first_purchase"] = (cutoff_date - first_last["first_purchase"]).dt.days
    first_last["customer_lifespan"] = (first_last["last_purchase"] - first_last["first_purchase"]).dt.days
    first_last["customer_lifespan"] = first_last["customer_lifespan"].clip(lower=1)

    # -- Inter-purchase time ---------------------------------------------------
    sorted_txns = calib_txns.sort_values(["customer_id", "date"])
    sorted_txns["prev_date"] = sorted_txns.groupby("customer_id")["date"].shift(1)
    sorted_txns["inter_purchase_days"] = (sorted_txns["date"] - sorted_txns["prev_date"]).dt.days

    ipt = sorted_txns.groupby("customer_id")["inter_purchase_days"].agg(
        inter_purchase_mean="mean",
        inter_purchase_std="std",
    ).reset_index()
    ipt["inter_purchase_std"] = ipt["inter_purchase_std"].fillna(0)

    # -- Weekend purchase ratio ------------------------------------------------
    transactions_with_dow = calib_txns.copy()
    transactions_with_dow["is_weekend"] = transactions_with_dow["date"].dt.dayofweek >= 5
    weekend_ratio = transactions_with_dow.groupby("customer_id")["is_weekend"].mean().reset_index()
    weekend_ratio.columns = ["customer_id", "weekend_purchase_ratio"]

    # -- Preferred month (mode) ------------------------------------------------
    transactions_with_month = calib_txns.copy()
    transactions_with_month["month"] = transactions_with_month["date"].dt.month
    month_mode = transactions_with_month.groupby("customer_id")["month"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 6
    ).reset_index()
    month_mode.columns = ["customer_id", "preferred_month"]

    # -- Rolling revenue (3m and 6m lookback from cutoff_date) -----------------
    cutoff_3m = cutoff_date - pd.Timedelta(days=90)
    cutoff_6m = cutoff_date - pd.Timedelta(days=180)

    rev_3m = calib_txns[calib_txns["date"] >= cutoff_3m].groupby("customer_id")["amount"].sum().reset_index()
    rev_3m.columns = ["customer_id", "rolling_3m_revenue"]

    rev_6m = calib_txns[calib_txns["date"] >= cutoff_6m].groupby("customer_id")["amount"].sum().reset_index()
    rev_6m.columns = ["customer_id", "rolling_6m_revenue"]

    # -- Merge all features ----------------------------------------------------
    features = rfm.merge(txn_agg, on="customer_id", how="left")
    features = features.merge(cat_diversity, on="customer_id", how="left")
    features = features.merge(first_last[["customer_id", "days_since_first_purchase", "customer_lifespan"]],
                               on="customer_id", how="left")
    features = features.merge(ipt, on="customer_id", how="left")
    features = features.merge(weekend_ratio, on="customer_id", how="left")
    features = features.merge(month_mode, on="customer_id", how="left")
    features = features.merge(rev_3m, on="customer_id", how="left")
    features = features.merge(rev_6m, on="customer_id", how="left")

    # Fill missing rolling revenue (customers with no recent purchases)
    features["rolling_3m_revenue"] = features["rolling_3m_revenue"].fillna(0)
    features["rolling_6m_revenue"] = features["rolling_6m_revenue"].fillna(0)

    # -- Add customer demographics ---------------------------------------------
    features = features.merge(
        customers[["customer_id", "age", "gender", "region", "acquisition_channel", "signup_date"]],
        on="customer_id", how="left"
    )

    # Tenure
    features["tenure_days"] = (cutoff_date - features["signup_date"]).dt.days
    features = features.drop(columns=["signup_date"])

    # Purchase velocity (orders per active month)
    features["purchase_velocity"] = features["frequency"] / (features["customer_lifespan"] / 30).clip(lower=1)

    # Discount sensitivity flag
    features["is_discount_sensitive"] = (features["avg_discount"] > 10).astype(int)

    # Churn flag (no purchase in last 90 days)
    features["is_churned"] = (features["recency"] > 90).astype(int)

    # -- Fill any remaining NaNs -----------------------------------------------
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].fillna(0)

    # -- Target variable: CLV = revenue in holdout period (6 months) -----------
    holdout_rev = holdout_txns.groupby("customer_id")["amount"].sum().reset_index()
    holdout_rev.columns = ["customer_id", "clv"]
    
    # Merge target and fill customers with 0 future revenue
    features = features.merge(holdout_rev, on="customer_id", how="left")
    features["clv"] = features["clv"].fillna(0)

    return features


def encode_features(features: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features for ML."""
    df = features.copy()

    # One-hot encode low-cardinality categoricals
    cat_cols = ["gender", "acquisition_channel"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # Label encode region (ordinal-ish in terms of revenue)
    le = LabelEncoder()
    df["region_encoded"] = le.fit_transform(df["region"])
    df = df.drop(columns=["region"])

    return df


def get_feature_columns(encoded_df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns (excludes target and ID)."""
    exclude = {"customer_id", "clv", "monetary"}
    return [c for c in encoded_df.columns if c not in exclude and encoded_df[c].dtype in [np.float64, np.int64, np.int32, np.uint8, np.float32]]


def build_feature_matrix(customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """End-to-end feature engineering pipeline."""
    print("\n[INFO] Engineering features...")
    features = compute_advanced_features(customers, transactions)
    encoded = encode_features(features)

    feature_cols = get_feature_columns(encoded)
    print(f"   [OK] {len(feature_cols)} features created for {len(encoded):,} customers")
    print(f"   [FEATURES] {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")

    return encoded


if __name__ == "__main__":
    customers = pd.read_csv("data/raw/customers.csv", parse_dates=["signup_date"])
    transactions = pd.read_csv("data/raw/transactions.csv", parse_dates=["date"])
    features = build_feature_matrix(customers, transactions)
    features.to_csv("data/raw/features.csv", index=False)
    print(f"\n[OK] Feature matrix saved: {features.shape}")
