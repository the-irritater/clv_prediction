"""
Customer Segmentation
======================
K-Means clustering on RFM features to create actionable customer segments.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt.rcParams.update({
    "figure.facecolor": "#0e1117", "axes.facecolor": "#1a1d23",
    "axes.edgecolor": "#2d3139", "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0", "xtick.color": "#a0a0a0",
    "ytick.color": "#a0a0a0", "grid.color": "#2d3139",
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
})
PALETTE = ["#6C63FF", "#FF6584", "#43E97B", "#FFD93D", "#00C9FF", "#F97316"]
SEGMENT_NAMES = {
    0: "Champions", 1: "Loyal Customers", 2: "At-Risk",
    3: "Lost / Hibernating", 4: "New Customers", 5: "Potential Loyalists",
}


def _save_fig(fig, name, output_dir):
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   📊 Saved {name}.png")


def find_optimal_k(rfm_scaled, k_range=range(3, 8)):
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(rfm_scaled)
        scores[k] = silhouette_score(rfm_scaled, labels, sample_size=min(10000, len(rfm_scaled)))
    best_k = max(scores, key=scores.get)
    print(f"   🔍 Silhouette scores: {', '.join(f'k={k}: {v:.3f}' for k, v in scores.items())}")
    print(f"   ✅ Optimal k = {best_k}")
    return best_k


def segment_customers(features, output_dir="outputs/figures"):
    os.makedirs(output_dir, exist_ok=True)
    print("\n🎯 Running Customer Segmentation...")

    rfm_cols = ["recency", "frequency", "monetary"]
    rfm_data = features[rfm_cols].copy()
    rfm_data["monetary"] = np.log1p(rfm_data["monetary"])
    rfm_data["frequency"] = np.log1p(rfm_data["frequency"])

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)
    best_k = find_optimal_k(rfm_scaled)

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    features = features.copy()
    features["segment_id"] = kmeans.fit_predict(rfm_scaled)

    segment_order = features.groupby("segment_id")["monetary"].mean().sort_values(ascending=False).index.tolist()
    rank_map = {old: new for new, old in enumerate(segment_order)}
    features["segment_id"] = features["segment_id"].map(rank_map)
    features["segment"] = features["segment_id"].map(lambda x: SEGMENT_NAMES.get(x, f"Segment {x}"))

    summary = features.groupby("segment").agg(
        count=("customer_id", "count"), avg_clv=("clv", "mean"),
        avg_recency=("recency", "mean"), avg_frequency=("frequency", "mean"),
    ).round(1)
    print(f"\n📋 Segment Summary:\n{summary.to_string()}")

    # Plot segment distribution
    seg_counts = features["segment"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].pie(seg_counts.values, labels=seg_counts.index, colors=PALETTE[:len(seg_counts)],
                autopct="%1.1f%%", startangle=90, pctdistance=0.85,
                wedgeprops=dict(width=0.4, edgecolor="#0e1117", linewidth=2),
                textprops=dict(color="#e0e0e0", fontsize=9))
    axes[0].set_title("Customer Segment Distribution")
    seg_rev = features.groupby("segment")["clv"].sum().sort_values(ascending=True)
    axes[1].barh(seg_rev.index, seg_rev.values, color=PALETTE[:len(seg_rev)])
    axes[1].set_title("Total Revenue by Segment")
    axes[1].set_xlabel("Total Revenue ($)")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    _save_fig(fig, "09_segment_distribution", output_dir)

    # Radar chart
    segments = sorted(features["segment"].unique())
    radar_data = features.groupby("segment")[rfm_cols].mean()
    radar_data["recency"] = radar_data["recency"].max() - radar_data["recency"]
    for col in rfm_cols:
        cmin, cmax = radar_data[col].min(), radar_data[col].max()
        radar_data[col] = (radar_data[col] - cmin) / (cmax - cmin) if cmax > cmin else 0.5
    angles = np.linspace(0, 2 * np.pi, len(rfm_cols), endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#1a1d23")
    for i, seg in enumerate(segments):
        vals = radar_data.loc[seg].tolist() + [radar_data.loc[seg].tolist()[0]]
        ax.plot(angles, vals, "o-", linewidth=2, label=seg, color=PALETTE[i % len(PALETTE)])
        ax.fill(angles, vals, alpha=0.15, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.capitalize() for c in rfm_cols], color="#e0e0e0")
    ax.set_title("RFM Radar by Segment", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), facecolor="#1a1d23", edgecolor="#2d3139")
    _save_fig(fig, "10_segment_radar", output_dir)

    # CLV boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    data = [features.loc[features["segment"] == s, "clv"].values for s in segments]
    bp = ax.boxplot(data, labels=segments, vert=True, patch_artist=True, showfliers=False,
                     medianprops=dict(color="#FF6584", linewidth=2),
                     whiskerprops=dict(color="#a0a0a0"), capprops=dict(color="#a0a0a0"))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.7)
    ax.set_title("CLV Distribution by Segment")
    ax.set_ylabel("CLV ($)")
    ax.tick_params(axis="x", rotation=15)
    _save_fig(fig, "11_segment_clv_boxplot", output_dir)

    print(f"\n✅ Segmentation complete — {best_k} segments")
    return features
