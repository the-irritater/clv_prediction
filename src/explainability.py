"""
Model Explainability
=====================
SHAP-based global and local explanations for CLV predictions.

Authors: Sanman, Varsha
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

plt.rcParams.update({
    "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
    "text.color": "#1a1d23", "axes.labelcolor": "#1a1d23",
    "xtick.color": "#1a1d23", "ytick.color": "#1a1d23",
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
})


def _save_fig(fig, name, output_dir):
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   [SAVED] {name}.png")


def explain_model(model, X_train, X_test, feature_names,
                  output_dir="reports/figures", model_type="auto"):
    """Generate SHAP explanations for the best model."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n[INFO] Generating SHAP Explanations...")

    # Use a subsample for speed
    n_sample = min(2000, X_train.shape[0])
    X_bg = X_train[:n_sample]
    X_explain = X_test[:min(1000, X_test.shape[0])]

    # Auto-detect model type for correct SHAP explainer
    tree_types = ("XGBRegressor", "LGBMRegressor", "RandomForestRegressor",
                  "GradientBoostingRegressor", "XGBClassifier", "LGBMClassifier")
    is_tree = model_type == "tree" or type(model).__name__ in tree_types

    if is_tree:
        print(f"   Using TreeExplainer for {type(model).__name__}")
        explainer = shap.TreeExplainer(model, data=X_bg, feature_names=feature_names)
    else:
        print(f"   Using generic Explainer for {type(model).__name__}")
        explainer = shap.Explainer(model, X_bg, feature_names=feature_names)

    shap_values = explainer(X_explain)

    # 1. Global - Beeswarm plot
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP Beeswarm - Global Feature Impact", fontsize=14, fontweight="bold")
    _save_fig(fig, "14_shap_beeswarm", output_dir)

    # 2. Global - Bar plot (mean |SHAP|)
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(shap_values, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP Feature Importance (Mean |SHAP|)", fontsize=14, fontweight="bold")
    _save_fig(fig, "15_shap_importance", output_dir)

    # 3. Local - Waterfall for highest CLV prediction
    predictions = model.predict(X_explain)
    high_idx = int(np.argmax(predictions))
    fig, ax = plt.subplots(figsize=(12, 7))
    shap.plots.waterfall(shap_values[high_idx], max_display=12, show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP Waterfall - High CLV Customer (Pred: ${predictions[high_idx]:,.0f})",
                 fontsize=13, fontweight="bold")
    _save_fig(fig, "16_shap_waterfall_high", output_dir)

    # 4. Local - Waterfall for lowest CLV prediction
    low_idx = int(np.argmin(predictions))
    fig, ax = plt.subplots(figsize=(12, 7))
    shap.plots.waterfall(shap_values[low_idx], max_display=12, show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP Waterfall - Low CLV Customer (Pred: ${predictions[low_idx]:,.0f})",
                 fontsize=13, fontweight="bold")
    _save_fig(fig, "17_shap_waterfall_low", output_dir)

    # 5. Dependence plot for top feature
    top_feature_idx = int(np.argmax(np.abs(shap_values.values).mean(axis=0)))
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.scatter(shap_values[:, top_feature_idx], show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP Dependence - {feature_names[top_feature_idx]}",
                 fontsize=14, fontweight="bold")
    _save_fig(fig, "18_shap_dependence", output_dir)

    print("[OK] SHAP explanations complete")
    return shap_values
