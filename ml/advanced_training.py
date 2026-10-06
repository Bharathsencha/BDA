"""
Advanced Machine Learning Training & Multi-Model Benchmark.
Trains Random Forest, Gradient Boosting, Decision Tree, and Logistic Regression models.
Computes ROC-AUC curves, Precision-Recall tradeoffs, and saves benchmark metrics.
"""
import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
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
    print("=================================================================")

    df = pd.read_csv(data_path)
    feature_cols = [
        "zone_id", "hour", "day_of_week", "is_weekend", "is_rush_hour",
        "demand_trips", "avg_fare", "avg_miles", "avg_seconds", "avg_speed_mph"
    ]
    target_col = "is_surge_above_1_5x"

    df_clean = df.dropna(subset=feature_cols + [target_col])
    X = df_clean[feature_cols]
    y = df_clean[target_col].astype(int)

    surge_count = y.sum()
    print(f"[*] Total Samples: {len(X)} | Surge (>1.5x): {surge_count} ({surge_count/len(y)*100:.1f}%) | Normal: {len(y)-surge_count}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if surge_count > 5 else None
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=9, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    }

    benchmark_results = {}
    curves_data = {}

    for name, model in models.items():
        print(f"\n[*] Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else y_pred

        roc_auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.5
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Save model artifact
        slug = name.lower().replace(" ", "_")
        model_file = MODEL_DIR / f"{slug}_surge_model.pkl"
        joblib.dump(model, model_file)

        # ROC Curve Points
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob)

        curves_data[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "precision_curve": p_curve.tolist(),
            "recall_curve": r_curve.tolist()
        }

        benchmark_results[name] = {
            "roc_auc": round(roc_auc, 4),
            "f1_score": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "confusion_matrix": cm,
            "artifact_path": str(model_file)
        }

        print(f"    - ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f}")

    # Feature Importance for Tree-Based Models
    rf_model = models["Random Forest"]
    rf_importances = {feature_cols[i]: round(float(rf_model.feature_importances_[i]), 4) for i in range(len(feature_cols))}
    sorted_importances = dict(sorted(rf_importances.items(), key=lambda item: item[1], reverse=True))

    summary = {
        "dataset_stats": {
            "total_records": len(X),
            "surge_records": int(surge_count),
            "normal_records": int(len(X) - surge_count),
            "features": feature_cols
        },
        "model_benchmarks": benchmark_results,
        "feature_importance": sorted_importances,
        "curves": curves_data
    }

    summary_file = MODEL_DIR / "evaluation_summary.json"
    with open(summary_file, "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"\n[OK] Saved evaluation summary and curves to: {summary_file}")
    print("=================================================================")

if __name__ == "__main__":
    run_advanced_training()
