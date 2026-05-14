"""
Synthetic E-Commerce Data Generator
====================================
Generates realistic customer + transaction data for CLV modeling.

Key realism features:
- Pareto-distributed spending (few whales, many low-spenders)
- Seasonal purchase patterns (holiday spikes in Nov–Dec)
- Realistic churn (~30% inactive after 6 months)
- Correlated features (age ↔ category, channel ↔ CLV)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Configuration ──────────────────────────────────────────────────────────────
SEED = 42
N_CUSTOMERS = 50_000
DATE_START = "2021-01-01"
DATE_END = "2023-12-31"

REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
REGION_WEIGHTS = [0.35, 0.30, 0.20, 0.10, 0.05]

CHANNELS = ["Organic Search", "Paid Ads", "Social Media", "Referral", "Direct"]
CHANNEL_WEIGHTS = [0.25, 0.30, 0.20, 0.15, 0.10]

CATEGORIES = ["Electronics", "Fashion", "Home & Kitchen", "Books", "Sports", "Beauty", "Grocery"]

GENDER_OPTIONS = ["Male", "Female", "Non-Binary"]
GENDER_WEIGHTS = [0.48, 0.48, 0.04]


def _seasonal_multiplier(month: int) -> float:
    """Returns a purchase-probability multiplier based on month of year."""
    seasonal = {
        1: 0.85, 2: 0.80, 3: 0.90, 4: 0.95, 5: 1.00, 6: 1.05,
        7: 1.00, 8: 0.95, 9: 1.00, 10: 1.10, 11: 1.40, 12: 1.50,
    }
    return seasonal.get(month, 1.0)


def generate_customers(n: int = N_CUSTOMERS, seed: int = SEED) -> pd.DataFrame:
    """Generate a customer demographics table."""
    rng = np.random.default_rng(seed)

    signup_start = datetime.strptime(DATE_START, "%Y-%m-%d")
    signup_end = datetime.strptime("2023-06-30", "%Y-%m-%d")
    signup_range = (signup_end - signup_start).days

    signup_dates = [signup_start + timedelta(days=int(d)) for d in rng.integers(0, signup_range, n)]

    ages = rng.normal(loc=35, scale=12, size=n).clip(18, 75).astype(int)
    genders = rng.choice(GENDER_OPTIONS, size=n, p=GENDER_WEIGHTS)
    regions = rng.choice(REGIONS, size=n, p=REGION_WEIGHTS)
    channels = rng.choice(CHANNELS, size=n, p=CHANNEL_WEIGHTS)

    # Base propensity (Pareto-like): determines how much a customer spends
    # Higher propensity → higher CLV (whale behavior)
    propensity = rng.pareto(a=2.5, size=n) + 0.1

    # Channel influences propensity: referrals have higher value
    channel_boost = np.where(
        np.isin(channels, ["Referral"]), 1.3,
        np.where(np.isin(channels, ["Organic Search"]), 1.1, 1.0)
    )
    propensity *= channel_boost

    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:06d}" for i in range(n)],
        "signup_date": signup_dates,
        "age": ages,
        "gender": genders,
        "region": regions,
        "acquisition_channel": channels,
        "propensity": propensity,  # internal — used for transaction generation
    })
    return df


def generate_transactions(customers: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Generate transactional data based on customer propensity."""
    rng = np.random.default_rng(seed)
    end_date = datetime.strptime(DATE_END, "%Y-%m-%d")

    records = []
    txn_id = 0

    for _, cust in customers.iterrows():
        cid = cust["customer_id"]
        signup = cust["signup_date"]
        prop = cust["propensity"]
        age = cust["age"]

        # Number of transactions scales with propensity
        # Mean ~10 transactions, whales can have 50+
        n_txns = max(1, int(rng.poisson(lam=prop * 4)))

        # Churn probability — ~30% of customers become inactive
        churn_prob = 0.30 if prop < 0.5 else 0.10
        churned = rng.random() < churn_prob
        if churned:
            # Churned customers only have transactions in the first 30–180 days
            active_window = int(rng.integers(30, 180))
            churn_date = min(signup + timedelta(days=active_window), end_date)
        else:
            churn_date = end_date

        available_days = (churn_date - signup).days
        if available_days <= 0:
            available_days = 1

        for _ in range(n_txns):
            # Transaction date — weighted toward seasonal peaks
            day_offset = int(rng.integers(0, available_days))
            txn_date = signup + timedelta(days=day_offset)
            month = txn_date.month

            # Seasonal gating: skip some transactions in low-season months
            if rng.random() > _seasonal_multiplier(month) * 0.75:
                continue

            # Amount: log-normal distribution scaled by propensity
            base_amount = rng.lognormal(mean=3.5, sigma=0.8)
            amount = round(base_amount * (0.5 + prop * 0.5), 2)
            amount = max(5.0, min(amount, 5000.0))  # clip extremes

            # Category — age-correlated preferences
            if age < 25:
                cat_weights = [0.30, 0.25, 0.05, 0.10, 0.15, 0.10, 0.05]
            elif age < 40:
                cat_weights = [0.25, 0.15, 0.20, 0.10, 0.10, 0.10, 0.10]
            elif age < 55:
                cat_weights = [0.15, 0.10, 0.30, 0.10, 0.10, 0.10, 0.15]
            else:
                cat_weights = [0.10, 0.08, 0.25, 0.20, 0.07, 0.10, 0.20]

            category = rng.choice(CATEGORIES, p=cat_weights)
            quantity = int(rng.integers(1, 6))
            discount_pct = round(rng.choice(
                [0, 0, 0, 5, 10, 15, 20, 25, 30],
                p=[0.35, 0.15, 0.10, 0.10, 0.10, 0.08, 0.05, 0.04, 0.03]
            ), 1)

            records.append({
                "transaction_id": f"TXN_{txn_id:08d}",
                "customer_id": cid,
                "date": txn_date,
                "amount": amount,
                "product_category": category,
                "quantity": quantity,
                "discount_pct": discount_pct,
            })
            txn_id += 1

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["customer_id", "date"]).reset_index(drop=True)
    return df


def generate_and_save(output_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate datasets and save to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    print("🔧 Generating customers...")
    customers = generate_customers()
    customers_public = customers.drop(columns=["propensity"])
    customers_public.to_csv(os.path.join(output_dir, "customers.csv"), index=False)
    print(f"   ✅ {len(customers_public):,} customers saved")

    print("🔧 Generating transactions...")
    transactions = generate_transactions(customers)
    transactions.to_csv(os.path.join(output_dir, "transactions.csv"), index=False)
    print(f"   ✅ {len(transactions):,} transactions saved")

    return customers_public, transactions


if __name__ == "__main__":
    generate_and_save()
