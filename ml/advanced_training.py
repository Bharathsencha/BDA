"""
Advanced Machine Learning Training & Multi-Model Benchmark Suite.
Trains Random Forest, Gradient Boosting, Decision Tree, and Logistic Regression models.
Evaluates 20-minute forward lookahead with temporal train/test split (Jan-Sep vs Oct-Dec).
Computes ROC-AUC curves, Precision-Recall tradeoffs, and saves benchmark metrics.
"""
import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.config import PROCESSED_DATA_DIR, MODEL_DIR

def run_advanced_training(data_path: Path = PROCESSED_DATA_DIR / "features_labels.csv"):
    if not data_path.exists():
        print(f"[!] Error: Feature dataset not found at {data_path}")
        return

    print("=================================================================")
    print(" Advanced Multi-Model Benchmark & Training Suite")
    print(" (20-Minute Forward Lookahead | Zero Target Leakage)")
    print("=================================================================")

    df = pd.read_csv(data_path)
    feature_cols = [
        "zone_id", "hour", "day_of_week", "is_weekend", "is_rush_hour",
        "demand_trips", "dropoff_trips", "net_deficit", "inflow_outflow_ratio", "demand_momentum", "avg_speed_mph"
    ]
    target_col = "is_surge_in_20min"

    df_clean = df.dropna(subset=feature_cols + [target_col]).copy()
    df_clean["month"] = pd.to_datetime(df_clean["time_bucket"]).dt.month

    # Temporal split: Jan-Sep (train) vs Oct-Dec (test)
    train_mask = df_clean["month"] <= 9
    test_mask = df_clean["month"] > 9

    X_train = df_clean.loc[train_mask, feature_cols]
    y_train = df_clean.loc[train_mask, target_col].astype(int)
    X_test = df_clean.loc[test_mask, feature_cols]
    y_test = df_clean.loc[test_mask, target_col].astype(int)

    surge_count = y_train.sum() + y_test.sum()
    print(f"[*] Total Samples: {len(df_clean)} | Surges (>1.5x in 20 min): {surge_count} ({surge_count/len(df_clean)*100:.2f}%)")
    print(f"[*] Train Samples (Jan-Sep): {len(X_train)} | Test Samples (Oct-Dec): {len(X_test)}")

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, class_weight="balanced", min_samples_split=15, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", min_samples_split=20, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    }

    benchmark_results = {}
    curves_data = {}

    for name, model in models.items():
        print(f"\n[*] Training {name}...")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
        y_pred = (y_prob >= 0.5).astype(int)

        roc_auc = float(roc_auc_score(y_test, y_prob))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Tuned threshold metric for fleet dispatch
        tuned_thresh = 0.35
        y_pred_tuned = (y_prob >= tuned_thresh).astype(int)
        recall_tuned = float(recall_score(y_test, y_pred_tuned, zero_division=0))
        precision_tuned = float(precision_score(y_test, y_pred_tuned, zero_division=0))

        # Save model artifact
        slug = name.lower().replace(" ", "_")
        model_file = MODEL_DIR / f"{slug}_surge_model.pkl"
        joblib.dump(model, model_file)

        # ROC Curve Points (downsampled for compact storage)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob)
        
        step_idx = max(1, len(fpr) // 100)
        curves_data[name] = {
            "fpr": [round(float(x), 4) for x in fpr[::step_idx]],
            "tpr": [round(float(x), 4) for x in tpr[::step_idx]],
            "precision_curve": [round(float(x), 4) for x in p_curve[::max(1, len(p_curve)//100)]],
            "recall_curve": [round(float(x), 4) for x in r_curve[::max(1, len(r_curve)//100)]]
        }

        benchmark_results[name] = {
            "roc_auc": round(roc_auc, 4),
            "f1_score": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "recall_tuned_35pct": round(recall_tuned, 4),
            "precision_tuned_35pct": round(precision_tuned, 4),
            "confusion_matrix": cm,
            "artifact_path": str(model_file.relative_to(PROJECT_ROOT))
        }

        print(f"    - ROC-AUC: {roc_auc:.4f} | Recall (0.50): {recall:.4f} | Recall (Tuned 0.35): {recall_tuned*100:.1f}%")

    # Feature Importance for Random Forest
    rf_model = models["Random Forest"]
    rf_importances = {feature_cols[i]: round(float(rf_model.feature_importances_[i]), 4) for i in range(len(feature_cols))}
    sorted_importances = dict(sorted(rf_importances.items(), key=lambda item: item[1], reverse=True))

    summary = {
        "dataset_stats": {
            "total_records": len(df_clean),
            "surge_records": int(surge_count),
            "normal_records": int(len(df_clean) - surge_count),
            "features": feature_cols,
            "horizon": "20 minutes in advance",
            "leak_free": True
        },
        "model_benchmarks": benchmark_results,
        "feature_importance": sorted_importances,
        "curves": curves_data
    }

    summary_file = MODEL_DIR / "evaluation_summary.json"
    with open(summary_file, "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"\n[OK] Saved evaluation summary and curves to: {summary_file.relative_to(PROJECT_ROOT)}")
    print("=================================================================")

if __name__ == "__main__":
    run_advanced_training()
