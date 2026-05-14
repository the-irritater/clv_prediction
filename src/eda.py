"""
Exploratory Data Analysis
==========================
Generates publication-quality EDA plots and data quality reports.
All plots are saved to reports/figures/.

Authors: Sanman, Varsha
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# -- Style configuration -------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#1a1d23",
    "axes.edgecolor": "#2d3139",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#a0a0a0",
    "ytick.color": "#a0a0a0",
    "grid.color": "#2d3139",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})

# Curated color palette
PALETTE = ["#6C63FF", "#FF6584", "#43E97B", "#FFD93D", "#00C9FF", "#F97316", "#A78BFA"]
GRADIENT_CMAP = sns.color_palette("blend:#6C63FF,#FF6584", as_cmap=True)


def _save_fig(fig, name: str, output_dir: str):
    """Save figure with tight layout."""
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"   [SAVED] {name}.png")


def data_quality_report(customers: pd.DataFrame, transactions: pd.DataFrame) -> dict:
    """Run data quality checks and return a summary dict."""
    report = {}

    # Missing values
    report["customers_missing"] = customers.isnull().sum().to_dict()
    report["transactions_missing"] = transactions.isnull().sum().to_dict()

    # Duplicates
    report["customer_duplicates"] = customers.duplicated(subset=["customer_id"]).sum()
    report["transaction_duplicates"] = transactions.duplicated(subset=["transaction_id"]).sum()

    # Outliers (amount)
    q1 = transactions["amount"].quantile(0.25)
    q3 = transactions["amount"].quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (transactions["amount"] < q1 - 1.5 * iqr) | (transactions["amount"] > q3 + 1.5 * iqr)
    report["amount_outliers_count"] = int(outlier_mask.sum())
    report["amount_outliers_pct"] = round(outlier_mask.mean() * 100, 2)

    # Basic stats
    report["n_customers"] = len(customers)
    report["n_transactions"] = len(transactions)
    report["date_range"] = f"{transactions['date'].min().date()} to {transactions['date'].max().date()}"
    report["avg_txns_per_customer"] = round(len(transactions) / len(customers), 1)

    print("\n[REPORT] Data Quality Report:")
    for k, v in report.items():
        print(f"   {k}: {v}")

    return report


def plot_revenue_distribution(transactions: pd.DataFrame, output_dir: str):
    """Revenue per customer - distribution on log scale."""
    rev_per_cust = transactions.groupby("customer_id")["amount"].sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram (log scale)
    axes[0].hist(rev_per_cust, bins=80, color=PALETTE[0], alpha=0.85, edgecolor="none")
    axes[0].set_xlabel("Total Revenue per Customer ($)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Revenue Distribution (Log Scale)")
    axes[0].set_yscale("log")
    axes[0].axvline(rev_per_cust.median(), color=PALETTE[1], linestyle="--", linewidth=2,
                     label=f"Median: ${rev_per_cust.median():,.0f}")
    axes[0].legend(facecolor="#1a1d23", edgecolor="#2d3139")

    # Box plot
    bp = axes[1].boxplot(rev_per_cust, vert=True, patch_artist=True,
                          boxprops=dict(facecolor=PALETTE[0], alpha=0.7),
                          medianprops=dict(color=PALETTE[1], linewidth=2),
                          whiskerprops=dict(color="#a0a0a0"),
                          capprops=dict(color="#a0a0a0"),
                          flierprops=dict(marker="o", markerfacecolor=PALETTE[1], markersize=3, alpha=0.3))
    axes[1].set_title("Revenue Box Plot")
    axes[1].set_ylabel("Total Revenue ($)")

    _save_fig(fig, "01_revenue_distribution", output_dir)


def plot_monthly_revenue_trend(transactions: pd.DataFrame, output_dir: str):
    """Monthly and quarterly revenue trends with seasonality."""
    monthly = transactions.set_index("date").resample("M")["amount"].sum().reset_index()
    monthly.columns = ["month", "revenue"]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(monthly["month"], monthly["revenue"], alpha=0.3, color=PALETTE[0])
    ax.plot(monthly["month"], monthly["revenue"], color=PALETTE[0], linewidth=2.5)

    # Highlight holiday peaks
    for _, row in monthly.iterrows():
        if row["month"].month in [11, 12]:
            ax.axvspan(row["month"] - pd.Timedelta(days=15), row["month"] + pd.Timedelta(days=15),
                       alpha=0.08, color=PALETTE[3])

    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax.grid(True, alpha=0.3)
    _save_fig(fig, "02_monthly_revenue_trend", output_dir)


def plot_correlation_heatmap(features_df: pd.DataFrame, output_dir: str):
    """Correlation heatmap of numerical features."""
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if "id" not in c.lower()]
    corr = features_df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap=GRADIENT_CMAP, center=0,
                annot=True, fmt=".2f", linewidths=0.5,
                ax=ax, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix")
    _save_fig(fig, "03_correlation_heatmap", output_dir)


def plot_category_mix(transactions: pd.DataFrame, output_dir: str):
    """Product category distribution - overall and by revenue."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # By count
    cat_counts = transactions["product_category"].value_counts()
    axes[0].barh(cat_counts.index, cat_counts.values, color=PALETTE[:len(cat_counts)], edgecolor="none")
    axes[0].set_title("Transactions by Category")
    axes[0].set_xlabel("Count")
    axes[0].invert_yaxis()

    # By revenue
    cat_revenue = transactions.groupby("product_category")["amount"].sum().sort_values(ascending=True)
    axes[1].barh(cat_revenue.index, cat_revenue.values, color=PALETTE[:len(cat_revenue)], edgecolor="none")
    axes[1].set_title("Revenue by Category")
    axes[1].set_xlabel("Revenue ($)")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    axes[1].invert_yaxis()

    _save_fig(fig, "04_category_mix", output_dir)


def plot_channel_performance(customers: pd.DataFrame, transactions: pd.DataFrame, output_dir: str):
    """Revenue and customer count by acquisition channel."""
    rev_per_cust = transactions.groupby("customer_id")["amount"].sum().reset_index()
    rev_per_cust.columns = ["customer_id", "total_revenue"]
    merged = customers.merge(rev_per_cust, on="customer_id", how="left").fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Customer count
    ch_counts = merged["acquisition_channel"].value_counts()
    axes[0].bar(range(len(ch_counts)), ch_counts.values, color=PALETTE[:len(ch_counts)], edgecolor="none")
    axes[0].set_xticks(range(len(ch_counts)))
    axes[0].set_xticklabels(ch_counts.index, rotation=30, ha="right")
    axes[0].set_title("Customers by Channel")
    axes[0].set_ylabel("Count")

    # Avg revenue by channel
    ch_rev = merged.groupby("acquisition_channel")["total_revenue"].mean().sort_values(ascending=False)
    axes[1].bar(range(len(ch_rev)), ch_rev.values, color=PALETTE[:len(ch_rev)], edgecolor="none")
    axes[1].set_xticks(range(len(ch_rev)))
    axes[1].set_xticklabels(ch_rev.index, rotation=30, ha="right")
    axes[1].set_title("Avg Revenue by Channel")
    axes[1].set_ylabel("Avg Revenue ($)")

    _save_fig(fig, "05_channel_performance", output_dir)


def plot_age_vs_clv(customers: pd.DataFrame, transactions: pd.DataFrame, output_dir: str):
    """Scatter plot of customer age vs CLV with regression line."""
    rev_per_cust = transactions.groupby("customer_id")["amount"].sum().reset_index()
    rev_per_cust.columns = ["customer_id", "clv"]
    merged = customers.merge(rev_per_cust, on="customer_id", how="left").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Subsample for readability
    sample = merged.sample(n=min(5000, len(merged)), random_state=42)
    ax.scatter(sample["age"], sample["clv"], alpha=0.15, s=10, color=PALETTE[0])

    # Regression line
    z = np.polyfit(merged["age"], merged["clv"], 2)
    p = np.poly1d(z)
    age_range = np.linspace(18, 75, 100)
    ax.plot(age_range, p(age_range), color=PALETTE[1], linewidth=3, label="Polynomial fit (deg=2)")

    ax.set_title("Age vs Customer Lifetime Value")
    ax.set_xlabel("Age")
    ax.set_ylabel("CLV ($)")
    ax.legend(facecolor="#1a1d23", edgecolor="#2d3139")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, "06_age_vs_clv", output_dir)


def plot_cohort_retention(customers: pd.DataFrame, transactions: pd.DataFrame, output_dir: str):
    """Monthly cohort retention heatmap."""
    txn = transactions.merge(customers[["customer_id", "signup_date"]], on="customer_id")
    txn["signup_cohort"] = txn["signup_date"].dt.to_period("Q").astype(str)
    txn["txn_month"] = txn["date"].dt.to_period("M")
    txn["signup_month"] = txn["signup_date"].dt.to_period("M")
    txn["months_since_signup"] = (txn["txn_month"].astype(int) - txn["signup_month"].astype(int))

    # Limit to first 12 months
    txn = txn[txn["months_since_signup"].between(0, 11)]

    cohort_data = txn.groupby(["signup_cohort", "months_since_signup"])["customer_id"].nunique().reset_index()
    cohort_sizes = txn.groupby("signup_cohort")["customer_id"].nunique().reset_index()
    cohort_sizes.columns = ["signup_cohort", "cohort_size"]

    cohort_data = cohort_data.merge(cohort_sizes, on="signup_cohort")
    cohort_data["retention_rate"] = cohort_data["customer_id"] / cohort_data["cohort_size"]

    pivot = cohort_data.pivot_table(index="signup_cohort", columns="months_since_signup",
                                     values="retention_rate")
    # Take last 8 cohorts for readability
    pivot = pivot.tail(8)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="YlGnBu", linewidths=0.5, ax=ax,
                cbar_kws={"format": mticker.PercentFormatter(1.0)})
    ax.set_title("Cohort Retention Rates")
    ax.set_xlabel("Months Since Signup")
    ax.set_ylabel("Signup Cohort")
    _save_fig(fig, "07_cohort_retention", output_dir)


def plot_region_revenue(customers: pd.DataFrame, transactions: pd.DataFrame, output_dir: str):
    """Revenue distribution by geographic region."""
    rev_per_cust = transactions.groupby("customer_id")["amount"].sum().reset_index()
    rev_per_cust.columns = ["customer_id", "total_revenue"]
    merged = customers.merge(rev_per_cust, on="customer_id", how="left").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    region_data = []
    regions = merged["region"].unique()
    for region in sorted(regions):
        region_data.append(merged.loc[merged["region"] == region, "total_revenue"].values)

    bp = ax.boxplot(region_data, labels=sorted(regions), vert=True, patch_artist=True,
                     showfliers=False,
                     medianprops=dict(color=PALETTE[1], linewidth=2),
                     whiskerprops=dict(color="#a0a0a0"),
                     capprops=dict(color="#a0a0a0"))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.7)

    ax.set_title("Revenue Distribution by Region")
    ax.set_ylabel("Total Revenue ($)")
    ax.tick_params(axis="x", rotation=20)
    _save_fig(fig, "08_region_revenue", output_dir)


def run_eda(customers: pd.DataFrame, transactions: pd.DataFrame,
            features_df: pd.DataFrame = None, output_dir: str = "reports/figures"):
    """Run the full EDA pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n[INFO] Running Exploratory Data Analysis...")

    report = data_quality_report(customers, transactions)
    plot_revenue_distribution(transactions, output_dir)
    plot_monthly_revenue_trend(transactions, output_dir)
    plot_category_mix(transactions, output_dir)
    plot_channel_performance(customers, transactions, output_dir)
    plot_age_vs_clv(customers, transactions, output_dir)
    plot_cohort_retention(customers, transactions, output_dir)
    plot_region_revenue(customers, transactions, output_dir)

    if features_df is not None:
        plot_correlation_heatmap(features_df, output_dir)

    print(f"\n[OK] EDA complete - {len(os.listdir(output_dir))} plots saved to {output_dir}/")
    return report


if __name__ == "__main__":
    customers = pd.read_csv("data/raw/customers.csv", parse_dates=["signup_date"])
    transactions = pd.read_csv("data/raw/transactions.csv", parse_dates=["date"])
    run_eda(customers, transactions)
