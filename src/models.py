"""
Model Training & Evaluation
=============================
Trains 4 ML models (Ridge, RandomForest, XGBoost, LightGBM) plus
probabilistic BG/NBD + Gamma-Gamma. Tunes top models with Optuna
and compares using RMSE/MAE/MAPE/R-squared.

Authors: Sanman, Varsha
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import optuna
import lifetimes

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.rcParams.update({
    "figure.facecolor": "#0e1117", "axes.facecolor": "#1a1d23",
    "axes.edgecolor": "#2d3139", "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0", "xtick.color": "#a0a0a0",
    "ytick.color": "#a0a0a0", "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold",
})
PALETTE = ["#6C63FF", "#FF6584", "#43E97B", "#FFD93D"]


def _save_fig(fig, name, output_dir):
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def mape(y_true, y_pred):
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "Model": name,
        "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
        "MAE": round(mean_absolute_error(y_test, preds), 2),
        "MAPE (%)": round(mape(y_test, preds), 2),
        "R2": round(r2_score(y_test, preds), 4),
    }


def _tune_xgb(X_train, y_train, n_trials=30):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "random_state": 42,
        }
        model = xgb.XGBRegressor(**params, verbosity=0)
        scores = cross_val_score(model, X_train, y_train, cv=3,
                                  scoring="neg_root_mean_squared_error", n_jobs=-1)
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _tune_lgb(X_train, y_train, n_trials=30):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "random_state": 42, "verbosity": -1,
        }
        model = lgb.LGBMRegressor(**params)
        scores = cross_val_score(model, X_train, y_train, cv=3,
                                  scoring="neg_root_mean_squared_error", n_jobs=-1)
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_and_evaluate(features_df, feature_cols, target="clv",
                       models_dir="models", reports_dir="reports", n_tune_trials=30):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(reports_dir, "figures"), exist_ok=True)
    print("\n[INFO] Training Models...")

    X = features_df[feature_cols].values
    y = features_df[target].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(features_df)), test_size=0.2, random_state=42)
    print(f"   Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    results = []
    models = {}

    # 1. Ridge Regression (Baseline)
    print("   > Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    results.append(evaluate_model("Ridge (Baseline)", ridge, X_test, y_test))
    models["ridge"] = ridge

    # 2. Random Forest (Intermediate)
    print("   > Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(evaluate_model("Random Forest", rf, X_test, y_test))
    models["random_forest"] = rf

    # 3. XGBoost (Tuned)
    print(f"   > Tuning XGBoost ({n_tune_trials} trials)...")
    xgb_params = _tune_xgb(X_train, y_train, n_trials=n_tune_trials)
    xgb_model = xgb.XGBRegressor(**xgb_params, random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)
    results.append(evaluate_model("XGBoost (Tuned)", xgb_model, X_test, y_test))
    models["xgboost"] = xgb_model

    # 4. LightGBM (Tuned)
    print(f"   > Tuning LightGBM ({n_tune_trials} trials)...")
    lgb_params = _tune_lgb(X_train, y_train, n_trials=n_tune_trials)
    lgb_model = lgb.LGBMRegressor(**lgb_params, random_state=42, verbosity=-1)
    lgb_model.fit(X_train, y_train)
    results.append(evaluate_model("LightGBM (Tuned)", lgb_model, X_test, y_test))
    models["lightgbm"] = lgb_model

    # 5. Probabilistic (BG/NBD + Gamma-Gamma)
    print("   > Training Probabilistic (BG/NBD + Gamma-Gamma)...")
    try:
        df_train = features_df.iloc[idx_train].copy()
        df_test = features_df.iloc[idx_test].copy()
        
        for df in [df_train, df_test]:
            df['freq_life'] = (df['frequency'] - 1).clip(lower=0)
            df['rec_life'] = df['customer_lifespan']
            df['T_life'] = df['days_since_first_purchase']
            df['monetary_life'] = df['avg_order_value']
            
        bgf = lifetimes.BetaGeoFitter(penalizer_coef=0.01)
        bgf.fit(df_train['freq_life'], df_train['rec_life'], df_train['T_life'])
        
        ggf = lifetimes.GammaGammaFitter(penalizer_coef=0.01)
        repeat_train = df_train[df_train['freq_life'] > 0]
        ggf.fit(repeat_train['freq_life'], repeat_train['monetary_life'])
        
        preds_prob = ggf.customer_lifetime_value(
            bgf, df_test['freq_life'], df_test['rec_life'], df_test['T_life'], df_test['monetary_life'],
            time=6, discount_rate=0.01, freq="D"
        ).fillna(0).values
        
        results.append({
            "Model": "BG/NBD + Gamma-Gamma",
            "RMSE": round(np.sqrt(mean_squared_error(y_test, preds_prob)), 2),
            "MAE": round(mean_absolute_error(y_test, preds_prob), 2),
            "MAPE (%)": round(mape(y_test, preds_prob), 2),
            "R2": round(r2_score(y_test, preds_prob), 4),
        })
        models["bgnbd"] = bgf
        models["gamma_gamma"] = ggf
    except Exception as e:
        print(f"   [WARNING] Probabilistic model failed: {e}")

    # Results table
    results_df = pd.DataFrame(results)
    print(f"\n[RESULTS] Model Comparison:\n{results_df.to_string(index=False)}")

    # Best model (only considering ML models for SHAP/residual compatibility)
    ml_results = results_df[results_df["Model"] != "BG/NBD + Gamma-Gamma"]
    best_name = ml_results.loc[ml_results["R2"].idxmax(), "Model"]
    best_key = {r["Model"]: k for k, r in zip(models.keys(), results)}[best_name]
    best_model = models[best_key]
    print(f"\n[BEST] Best model: {best_name}")

    # Save models
    for name, model in models.items():
        joblib.dump(model, os.path.join(models_dir, f"{name}.joblib"))
    joblib.dump(feature_cols, os.path.join(models_dir, "feature_cols.joblib"))

    # Plot: Model comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["RMSE", "MAE", "R2"]
    for i, metric in enumerate(metrics):
        vals = results_df[metric].values
        bars = axes[i].bar(results_df["Model"], vals, color=PALETTE, edgecolor="none")
        axes[i].set_title(metric)
        axes[i].tick_params(axis="x", rotation=25)
        for bar, v in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                          f"{v}", ha="center", va="bottom", fontsize=9, color="#e0e0e0")
    _save_fig(fig, "12_model_comparison", os.path.join(reports_dir, "figures"))

    # Plot: Residuals for best model
    preds = best_model.predict(X_test)
    residuals = y_test - preds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(preds, residuals, alpha=0.1, s=5, color=PALETTE[0])
    axes[0].axhline(0, color=PALETTE[1], linewidth=2, linestyle="--")
    axes[0].set_title(f"Residuals - {best_name}")
    axes[0].set_xlabel("Predicted CLV")
    axes[0].set_ylabel("Residual")
    axes[1].hist(residuals, bins=60, color=PALETTE[0], edgecolor="none", alpha=0.8)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")
    _save_fig(fig, "13_residuals", os.path.join(reports_dir, "figures"))

    # Cross-validation on best model
    print(f"\n[CV] 5-Fold CV for {best_name}...")
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5,
                                 scoring="neg_root_mean_squared_error", n_jobs=-1)
    print(f"   CV RMSE: {-cv_scores.mean():.2f} +/- {cv_scores.std():.2f}")

    return models, results_df, best_model, best_key, (X_train, X_test, y_train, y_test)
