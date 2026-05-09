"""
run_prediction.py — FloodSense Prediction Leaderboard
Neural Nova · BTech Fest 2026 · BNU Lahore
------------------------------------------------------
DROP THIS FILE into your project folder (same level as your trained model).
Run: python run_prediction.py
Your submission file will be saved as: floodsense_submission.csv

Requirements:
- Your trained model must be loadable (joblib or pickle).
- Update the THREE configuration values in the USER CONFIG section below.
- Do not modify anything outside the USER CONFIG section.
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIG — edit these three values only
# ─────────────────────────────────────────────────────────────────────────────

TEAM_NAME = "strangers"  # e.g. "Team Nexus"
INSTITUTION = "umt"  # e.g. "FAST NUCES, Lahore"

# Path to your saved model file — supports joblib (.pkl, .joblib) or pickle
MODEL_PATH = "best_model.pkl"  # e.g. "model/random_forest.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# DO NOT EDIT BELOW THIS LINE
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "district": "Nowshera",
        "precipitation": 142.5,
        "precip_3day_avg": 98.2,
        "precip_7day_avg": 61.4,
        "soil_moisture": 0.74,
        "soil_3day_avg": 0.68,
        "water_area_km2": 890.3,
        "water_area_change": 210.5,
        "water_area_pct_change": 0.31,
        "temperature": 31.2,
        "humidity": 88.0,
        "pressure": 98800.0,
        "evaporation": -0.0018,
        "wind_speed": 4.1,
        "month": 8,
        "day_of_year": 227,
        "is_monsoon": 1,
        "ds_idx": 3.0,
    },
    {
        "district": "Jacobabad",
        "precipitation": 0.4,
        "precip_3day_avg": 1.2,
        "precip_7day_avg": 0.9,
        "soil_moisture": 0.09,
        "soil_3day_avg": 0.11,
        "water_area_km2": 42.1,
        "water_area_change": -5.3,
        "water_area_pct_change": -0.11,
        "temperature": 43.7,
        "humidity": 24.0,
        "pressure": 99800.0,
        "evaporation": 0.0061,
        "wind_speed": 6.8,
        "month": 6,
        "day_of_year": 168,
        "is_monsoon": 0,
        "ds_idx": 1.0,
    },
    {
        "district": "Sindh_District",
        "precipitation": 387.0,
        "precip_3day_avg": 241.3,
        "precip_7day_avg": 138.6,
        "soil_moisture": 0.91,
        "soil_3day_avg": 0.87,
        "water_area_km2": 1420.7,
        "water_area_change": 640.2,
        "water_area_pct_change": 0.82,
        "temperature": 28.5,
        "humidity": 96.0,
        "pressure": 98200.0,
        "evaporation": -0.0031,
        "wind_speed": 5.2,
        "month": 9,
        "day_of_year": 258,
        "is_monsoon": 1,
        "ds_idx": 2.0,
    },
    {
        "district": "KP_District",
        "precipitation": 55.8,
        "precip_3day_avg": 38.4,
        "precip_7day_avg": 22.1,
        "soil_moisture": 0.42,
        "soil_3day_avg": 0.38,
        "water_area_km2": 320.9,
        "water_area_change": 45.6,
        "water_area_pct_change": 0.17,
        "temperature": 24.1,
        "humidity": 71.0,
        "pressure": 99100.0,
        "evaporation": -0.0009,
        "wind_speed": 3.3,
        "month": 7,
        "day_of_year": 196,
        "is_monsoon": 1,
        "ds_idx": 1.0,
    },
    {
        "district": "Balochistan_District",
        "precipitation": 18.3,
        "precip_3day_avg": 9.7,
        "precip_7day_avg": 5.4,
        "soil_moisture": 0.21,
        "soil_3day_avg": 0.19,
        "water_area_km2": 88.4,
        "water_area_change": 12.1,
        "water_area_pct_change": 0.16,
        "temperature": 33.8,
        "humidity": 41.0,
        "pressure": 99400.0,
        "evaporation": 0.0022,
        "wind_speed": 5.9,
        "month": 8,
        "day_of_year": 215,
        "is_monsoon": 1,
        "ds_idx": 4.0,
    },
]

# District encoding: maps scenario district names → (encoded int, avg elevation m)
_DISTRICT_MAP = {
    "Nowshera": {"enc": 1, "avg_elevation_m": 285},
    "Jacobabad": {"enc": 2, "avg_elevation_m": 109},
    "Sindh_District": {"enc": 2, "avg_elevation_m": 109},
    "KP_District": {"enc": 1, "avg_elevation_m": 285},
    "Balochistan_District": {"enc": 0, "avg_elevation_m": 610},
}

# Skewed columns that go through PowerTransformer before StandardScaler
_SKEWED = [
    "precip_lag1",
    "precipitation",
    "precip_3day_avg",
    "wind_speed",
    "precip_7day_avg",
    "water_area_pct_change",
    "soil_moisture",
    "soil_3day_avg",
    "evaporation",
    "water_area_change",
    "water_area_accel",
    "is_monsoon",
]

# The 23 features the trained model expects (loaded from feature_cols.pkl)
FEATURE_COLS = [
    "water_area_km2",
    "water_area_change",
    "water_area_pct_change",
    "pressure",
    "temp_3day_avg",
    "wind_speed",
    "avg_elevation_m",
    "soil_3day_avg",
    "evaporation",
    "precipitation",
    "temperature",
    "humidity",
    "precip_3day_avg",
    "precip_7day_avg",
    "soil_moisture",
    "water_area_lag1",
    "water_area_lag2",
    "precip_lag1",
    "water_area_accel",
    "month",
    "is_monsoon",
    "day_of_year",
    "district_enc",
]


def prob_to_risk(prob):
    if prob < 0.25:
        return "Low"
    elif prob < 0.50:
        return "Medium"
    elif prob < 0.75:
        return "High"
    else:
        return "Critical"


def load_model(path):
    if not os.path.exists(path):
        print(f"\n  ERROR: Model file not found at '{path}'")
        print("  Update MODEL_PATH in the USER CONFIG section at the top of this file.")
        sys.exit(1)
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        pass
    try:
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"\n  ERROR: Could not load model from '{path}'")
        print(f"  Details: {e}")
        sys.exit(1)


def load_preprocessors():
    """Load PowerTransformer and StandardScaler saved alongside the model."""
    import joblib

    base = os.path.dirname(os.path.abspath(__file__))
    pt = joblib.load(os.path.join(base, "power_transformer.pkl"))
    sc = joblib.load(os.path.join(base, "standard_scaler.pkl"))
    return pt, sc


def build_feature_row(scenario):
    """Map a scenario dict (with ds_idx / district name) to the 23-column row."""
    d = _DISTRICT_MAP.get(scenario["district"], {"enc": 1, "avg_elevation_m": 285})
    return {
        "water_area_km2": scenario["water_area_km2"],
        "water_area_change": scenario["water_area_change"],
        "water_area_pct_change": scenario["water_area_pct_change"],
        "pressure": scenario["pressure"],
        "temp_3day_avg": scenario["temperature"],
        "wind_speed": scenario["wind_speed"],
        "avg_elevation_m": d["avg_elevation_m"],
        "soil_3day_avg": scenario["soil_3day_avg"],
        "evaporation": scenario["evaporation"],
        "precipitation": scenario["precipitation"],
        "temperature": scenario["temperature"],
        "humidity": scenario["humidity"],
        "precip_3day_avg": scenario["precip_3day_avg"],
        "precip_7day_avg": scenario["precip_7day_avg"],
        "soil_moisture": scenario["soil_moisture"],
        "water_area_lag1": scenario["water_area_km2"],
        "water_area_lag2": scenario["water_area_km2"],
        "precip_lag1": scenario["precipitation"],
        "water_area_accel": 0.0,
        "month": scenario["month"],
        "is_monsoon": scenario["is_monsoon"],
        "day_of_year": scenario["day_of_year"],
        "district_enc": d["enc"],
    }


def scale_features(df_row, pt, sc):
    """Apply Yeo-Johnson PowerTransformer then StandardScaler."""
    X = df_row[FEATURE_COLS].copy().astype(float)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(0.0)
    skewed_present = [c for c in _SKEWED if c in X.columns]
    if skewed_present:
        transformed = pt.transform(X[skewed_present])
        transformed = np.where(np.isfinite(transformed), transformed, 0.0)
        X[skewed_present] = transformed
    return sc.transform(X)


def get_top_feature(model, feature_names):
    """Extract the most important feature from the model if available."""
    try:
        importances = model.feature_importances_
        idx = int(np.argmax(importances))
        return feature_names[idx]
    except AttributeError:
        pass
    try:
        # Linear models
        coef = np.abs(model.coef_[0])
        idx = int(np.argmax(coef))
        return feature_names[idx]
    except AttributeError:
        return "Not available"


def validate_config():
    errors = []
    if TEAM_NAME == "YOUR_TEAM_NAME":
        errors.append("  - Set your TEAM_NAME in the USER CONFIG section.")
    if INSTITUTION == "YOUR_INSTITUTION":
        errors.append("  - Set your INSTITUTION in the USER CONFIG section.")
    if errors:
        print("\n  Please complete the USER CONFIG section before running:\n")
        for e in errors:
            print(e)
        sys.exit(1)


def main():
    print("\n" + "=" * 60)
    print("  FLOODSENSE — Prediction Leaderboard")
    print("  Neural Nova · BTech Fest 2026")
    print("=" * 60)

    validate_config()

    print(f"\n  Team       : {TEAM_NAME}")
    print(f"  Institution: {INSTITUTION}")
    print(f"  Model      : {MODEL_PATH}")
    print(f"  Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n  Loading model...")

    model = load_model(MODEL_PATH)
    print("  Model loaded successfully.")

    pt, sc = load_preprocessors()
    print("  Preprocessors loaded (PowerTransformer + StandardScaler).\n")

    top_feature = get_top_feature(model, FEATURE_COLS)

    results = []
    print(f"  {'District':<25} {'Risk':<10} {'Confidence':>10}   Top Feature")
    print("  " + "-" * 65)

    for scenario in SCENARIOS:
        row = build_feature_row(scenario)
        df_row = pd.DataFrame([row])
        try:
            X_scaled = scale_features(df_row, pt, sc)
            prob = float(model.predict_proba(X_scaled)[:, 1][0])
        except Exception as e:
            print(f"\n  ERROR: Model prediction failed.\n  Details: {e}")
            print("\n  Common cause: your model was trained on different features.")
            print(
                "  Check that your feature columns match exactly what you trained on."
            )
            sys.exit(1)

        risk = prob_to_risk(prob)
        confidence = round(prob * 100, 1)

        print(
            f"  {scenario['district']:<25} {risk:<10} {confidence:>9.1f}%   {top_feature}"
        )

        results.append(
            {
                "team_name": TEAM_NAME,
                "institution": INSTITUTION,
                "district": scenario["district"],
                "risk_level": risk,
                "confidence_pct": confidence,
                "top_feature": top_feature,
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    output_df = pd.DataFrame(results)
    output_file = "floodsense_submission.csv"
    output_df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print(f"  Submission saved: {output_file}")
    print("  Share this file via the Google Form before 12:10 PM.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
