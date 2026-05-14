"""
CLV Prediction Dashboard
=========================
Streamlit multi-tab dashboard with Overview, EDA, Segments, Predictions, and Model Performance.
"""

import os, sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib
import shap

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CLV Prediction", page_icon="💎", layout="wide",
                   initial_sidebar_state="expanded")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.main { background: #0e1117; }
.stMetric { background: linear-gradient(135deg, #1a1d23 0%, #252830 100%);
    border: 1px solid #2d3139; border-radius: 16px; padding: 20px; }
.stMetric label { color: #a0a0a0 !important; font-size: 0.85rem !important; }
.stMetric [data-testid="stMetricValue"] { color: #e0e0e0 !important;
    font-size: 1.8rem !important; font-weight: 700 !important; }
h1, h2, h3 { color: #e0e0e0 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background: #1a1d23; border-radius: 10px;
    color: #a0a0a0; padding: 10px 24px; border: 1px solid #2d3139; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #6C63FF, #FF6584);
    color: white !important; border: none; }
div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0e1117 0%, #1a1d23 100%);
    border-right: 1px solid #2d3139; }
.gradient-text { background: linear-gradient(135deg, #6C63FF, #FF6584);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2rem; font-weight: 700; }
.card { background: linear-gradient(135deg, #1a1d23, #252830);
    border: 1px solid #2d3139; border-radius: 16px; padding: 24px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

COLORS = ["#6C63FF", "#FF6584", "#43E97B", "#FFD93D", "#00C9FF", "#F97316"]


@st.cache_data
def load_data():
    data_dir = os.path.join(ROOT, "data", "raw")
    customers = pd.read_csv(os.path.join(data_dir, "customers.csv"), parse_dates=["signup_date"])
    transactions = pd.read_csv(os.path.join(data_dir, "transactions.csv"), parse_dates=["date"])
    features = pd.read_csv(os.path.join(data_dir, "features_segmented.csv"))
    return customers, transactions, features


@st.cache_resource
def load_models():
    model_dir = os.path.join(ROOT, "outputs", "models")
    models = {}
    for name in ["ridge", "random_forest", "xgboost", "lightgbm"]:
        path = os.path.join(model_dir, f"{name}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.joblib"))
    return models, feature_cols


def plotly_dark_layout(fig, title=""):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#1a1d23",
        title=dict(text=title, font=dict(size=18, color="#e0e0e0")),
        font=dict(family="Inter", color="#a0a0a0"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# ── Load data ─────────────────────────────────────────────────────────────────
try:
    customers, transactions, features = load_data()
    models, feature_cols = load_models()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"⚠️ Run `python run_pipeline.py` first to generate data and models.\n\nError: {e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="gradient-text">💎 CLV Prediction</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Customer Lifetime Value** prediction system with segmentation & explainability.")
    st.markdown("---")
    st.markdown(f"👥 **{len(customers):,}** customers")
    st.markdown(f"🧾 **{len(transactions):,}** transactions")
    st.markdown(f"📊 **{len(feature_cols)}** features")
    st.markdown(f"🤖 **{len(models)}** models trained")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🔍 EDA", "🎯 Segments", "🔮 Predictions", "🏆 Model Performance"])

# ═══════════════════════════════ TAB 1: OVERVIEW ══════════════════════════════
with tab1:
    st.markdown("## Business Overview")
    col1, col2, col3, col4 = st.columns(4)
    total_rev = features["clv"].sum()
    avg_clv = features["clv"].mean()
    churn_rate = features["is_churned"].mean() * 100 if "is_churned" in features.columns else 0
    col1.metric("Total Customers", f"{len(features):,}")
    col2.metric("Total Revenue", f"${total_rev:,.0f}")
    col3.metric("Avg CLV", f"${avg_clv:,.0f}")
    col4.metric("Churn Rate", f"{churn_rate:.1f}%")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        monthly = transactions.set_index("date").resample("M")["amount"].sum().reset_index()
        fig = px.area(monthly, x="date", y="amount", title="Monthly Revenue Trend",
                      color_discrete_sequence=[COLORS[0]])
        fig.update_traces(fill="tozeroy", fillcolor="rgba(108,99,255,0.2)")
        st.plotly_chart(plotly_dark_layout(fig, "Monthly Revenue Trend"), use_container_width=True)
    with c2:
        top10 = features.nlargest(10, "clv")[["customer_id", "clv", "frequency", "recency"]]
        fig = px.bar(top10, x="customer_id", y="clv", title="Top 10 Customers by CLV",
                     color="clv", color_continuous_scale=["#6C63FF", "#FF6584"])
        st.plotly_chart(plotly_dark_layout(fig, "Top 10 Customers by CLV"), use_container_width=True)

# ═══════════════════════════════ TAB 2: EDA ═══════════════════════════════════
with tab2:
    st.markdown("## Exploratory Data Analysis")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(features, x="clv", nbins=80, title="CLV Distribution",
                           color_discrete_sequence=[COLORS[0]], log_y=True)
        st.plotly_chart(plotly_dark_layout(fig, "CLV Distribution (Log Scale)"), use_container_width=True)
    with c2:
        cat_rev = transactions.groupby("product_category")["amount"].sum().sort_values(ascending=True).reset_index()
        fig = px.bar(cat_rev, x="amount", y="product_category", orientation="h",
                     color_discrete_sequence=[COLORS[2]])
        st.plotly_chart(plotly_dark_layout(fig, "Revenue by Category"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        numeric_feats = features[["recency", "frequency", "monetary", "avg_order_value",
                                   "category_diversity", "tenure_days"]].dropna()
        corr = numeric_feats.corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        aspect="auto", zmin=-1, zmax=1)
        st.plotly_chart(plotly_dark_layout(fig, "Feature Correlations"), use_container_width=True)
    with c4:
        ch_rev = features.groupby("acquisition_channel" if "acquisition_channel" in features.columns
                                   else features.columns[0])["clv"].mean().sort_values(ascending=False).reset_index()
        if "acquisition_channel" in features.columns:
            fig = px.bar(ch_rev, x="acquisition_channel", y="clv", color_discrete_sequence=[COLORS[4]])
            st.plotly_chart(plotly_dark_layout(fig, "Avg CLV by Channel"), use_container_width=True)

# ═══════════════════════════════ TAB 3: SEGMENTS ══════════════════════════════
with tab3:
    st.markdown("## Customer Segments")
    if "segment" in features.columns:
        seg_summary = features.groupby("segment").agg(
            Customers=("customer_id", "count"),
            Avg_CLV=("clv", "mean"), Total_Revenue=("clv", "sum"),
            Avg_Recency=("recency", "mean"), Avg_Frequency=("frequency", "mean"),
        ).round(1).reset_index()

        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(seg_summary, values="Customers", names="segment",
                         color_discrete_sequence=COLORS, hole=0.45)
            st.plotly_chart(plotly_dark_layout(fig, "Segment Distribution"), use_container_width=True)
        with c2:
            fig = px.bar(seg_summary.sort_values("Avg_CLV", ascending=True),
                         x="Avg_CLV", y="segment", orientation="h",
                         color="segment", color_discrete_sequence=COLORS)
            st.plotly_chart(plotly_dark_layout(fig, "Avg CLV by Segment"), use_container_width=True)

        # Radar chart
        rfm_cols = ["recency", "frequency", "monetary"]
        radar_data = features.groupby("segment")[rfm_cols].mean()
        radar_data["recency"] = radar_data["recency"].max() - radar_data["recency"]
        for col in rfm_cols:
            cmin, cmax = radar_data[col].min(), radar_data[col].max()
            if cmax > cmin:
                radar_data[col] = (radar_data[col] - cmin) / (cmax - cmin)
        fig = go.Figure()
        for i, seg in enumerate(radar_data.index):
            vals = radar_data.loc[seg].tolist() + [radar_data.loc[seg].tolist()[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=[c.capitalize() for c in rfm_cols] + [rfm_cols[0].capitalize()],
                fill="toself", name=seg, line=dict(color=COLORS[i % len(COLORS)])))
        st.plotly_chart(plotly_dark_layout(fig, "RFM Radar by Segment"), use_container_width=True)

        st.markdown("### Segment Details")
        st.dataframe(seg_summary.style.format({
            "Avg_CLV": "${:,.0f}", "Total_Revenue": "${:,.0f}",
            "Avg_Recency": "{:.0f} days", "Avg_Frequency": "{:.1f}",
        }), use_container_width=True)
    else:
        st.warning("No segment data found. Run the pipeline first.")

# ═══════════════════════════════ TAB 4: PREDICTIONS ═══════════════════════════
with tab4:
    st.markdown("## CLV Predictor")
    best_key = "xgboost" if "xgboost" in models else list(models.keys())[-1]
    best_model = models[best_key]

    st.markdown(f"**Active model:** `{best_key}` | **Features:** {len(feature_cols)}")
    st.markdown("---")

    st.markdown("### Single Customer Prediction")
    cust_id = st.selectbox("Select Customer", features["customer_id"].head(200).tolist())
    if cust_id:
        cust_row = features[features["customer_id"] == cust_id]
        if not cust_row.empty:
            valid_cols = [c for c in feature_cols if c in cust_row.columns]
            X_single = cust_row[valid_cols].values
            pred = best_model.predict(X_single)[0]
            actual = cust_row["clv"].values[0]

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted CLV", f"${pred:,.0f}")
            c2.metric("Actual CLV", f"${actual:,.0f}")
            c3.metric("Error", f"${abs(pred - actual):,.0f}")

            seg = cust_row["segment"].values[0] if "segment" in cust_row.columns else "N/A"
            st.markdown(f"**Segment:** {seg} | **Recency:** {cust_row['recency'].values[0]:.0f} days | "
                        f"**Frequency:** {cust_row['frequency'].values[0]:.0f} orders")

    # Batch section
    st.markdown("---")
    st.markdown("### Future Revenue Forecast")
    if "segment" in features.columns:
        months_ahead = st.slider("Forecast months ahead", 3, 24, 12)
        seg_forecast = features.groupby("segment").agg(
            customers=("customer_id", "count"),
            avg_monthly_rev=("clv", lambda x: x.mean() / 12),
        ).reset_index()
        seg_forecast["forecast_revenue"] = seg_forecast["customers"] * seg_forecast["avg_monthly_rev"] * months_ahead
        total_forecast = seg_forecast["forecast_revenue"].sum()
        st.metric(f"Projected Revenue ({months_ahead}mo)", f"${total_forecast:,.0f}")
        fig = px.bar(seg_forecast, x="segment", y="forecast_revenue",
                     color="segment", color_discrete_sequence=COLORS)
        st.plotly_chart(plotly_dark_layout(fig, f"{months_ahead}-Month Revenue Forecast by Segment"),
                        use_container_width=True)

# ═══════════════════════════════ TAB 5: MODEL PERF ════════════════════════════
with tab5:
    st.markdown("## Model Performance")
    results_path = os.path.join(ROOT, "outputs", "model_results.csv")
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        st.dataframe(results_df.style.highlight_max(subset=["R²"], color="#43E97B")
                     .highlight_min(subset=["RMSE", "MAE"], color="#43E97B"),
                     use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(results_df, x="Model", y="R²", color="Model",
                         color_discrete_sequence=COLORS)
            st.plotly_chart(plotly_dark_layout(fig, "R² Score Comparison"), use_container_width=True)
        with c2:
            fig = px.bar(results_df, x="Model", y="RMSE", color="Model",
                         color_discrete_sequence=COLORS)
            st.plotly_chart(plotly_dark_layout(fig, "RMSE Comparison"), use_container_width=True)
    else:
        st.warning("No model results found. Run the pipeline first.")

    # Show saved figures
    st.markdown("### Visualizations")
    fig_dir = os.path.join(ROOT, "outputs", "figures")
    if os.path.isdir(fig_dir):
        figs = sorted([f for f in os.listdir(fig_dir) if f.endswith(".png")])
        selected = st.selectbox("Select plot", figs)
        if selected:
            st.image(os.path.join(fig_dir, selected), use_container_width=True)
