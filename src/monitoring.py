"""
Monitoring & Drift Detection
==============================
PSI-based feature drift detection and model performance tracking.
"""

import numpy as np
import pandas as pd


def calculate_psi(expected, actual, bins=10):
    """Population Stability Index between two distributions."""
    breakpoints = np.linspace(0, 100, bins + 1)
    expected_percents = np.percentile(expected, breakpoints)

    expected_counts = np.histogram(expected, bins=expected_percents)[0]
    actual_counts = np.histogram(actual, bins=expected_percents)[0]

    # Avoid zero
    expected_pct = (expected_counts + 1) / (len(expected) + bins)
    actual_pct = (actual_counts + 1) / (len(actual) + bins)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(psi, 6)


def detect_drift(train_df, new_df, feature_cols, threshold=0.2):
    """Check PSI for all features and flag drifted ones."""
    print("\n📡 Running Drift Detection...")
    results = []
    for col in feature_cols:
        if col not in train_df.columns or col not in new_df.columns:
            continue
        try:
            psi = calculate_psi(train_df[col].dropna().values, new_df[col].dropna().values)
            status = "🔴 DRIFT" if psi > threshold else ("🟡 WARNING" if psi > 0.1 else "🟢 OK")
            results.append({"Feature": col, "PSI": psi, "Status": status})
        except Exception:
            results.append({"Feature": col, "PSI": None, "Status": "⚠️ ERROR"})

    results_df = pd.DataFrame(results).sort_values("PSI", ascending=False)
    drifted = results_df[results_df["Status"] == "🔴 DRIFT"]

    print(f"   Features checked: {len(results_df)}")
    print(f"   Drifted features: {len(drifted)}")
    if len(drifted) > 0:
        print(f"\n{drifted.to_string(index=False)}")
    print("✅ Drift detection complete")
    return results_df


def monitoring_strategy():
    """Return a documented monitoring strategy."""
    return """
╔══════════════════════════════════════════════════════════════════╗
║                   MODEL MONITORING STRATEGY                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. DATA DRIFT (Weekly)                                         ║
║     • Compute PSI for all input features                        ║
║     • Alert if PSI > 0.2 for any feature                        ║
║     • Track categorical distribution shifts                      ║
║                                                                  ║
║  2. PREDICTION DRIFT (Weekly)                                   ║
║     • Monitor CLV prediction distribution                        ║
║     • Alert if mean prediction shifts > 15%                      ║
║     • Track segment migration rates                              ║
║                                                                  ║
║  3. PERFORMANCE DECAY (Monthly)                                 ║
║     • Recalculate RMSE/MAE on fresh labeled data                ║
║     • Alert if RMSE degrades > 10% from baseline                ║
║     • Compare actual vs predicted CLV for closed cohorts         ║
║                                                                  ║
║  4. RETRAINING TRIGGERS                                         ║
║     • PSI > 0.25 on 3+ features → immediate retrain             ║
║     • RMSE degradation > 15% → retrain + investigate             ║
║     • Quarterly scheduled retrain regardless                     ║
║     • Major business event (acquisition, pricing change)         ║
║                                                                  ║
║  5. INFRASTRUCTURE                                              ║
║     • Log all predictions with timestamps                        ║
║     • Version all models with metadata                           ║
║     • A/B test new models before full deployment                 ║
║     • Maintain shadow scoring for candidate models               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
