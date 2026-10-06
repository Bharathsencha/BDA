"""
Machine Learning Baseline: Decision Tree & Random Forest Classifiers
Predicts 20-minute advance surge multiplier (> 1.5x) based on MapReduce aggregated features.
"""
import sys
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import PROCESSED_DATA_DIR, MODEL_DIR

def train_models(data_path: Path = PROCESSED_DATA_DIR / "features_labels.csv"):
    if not data_path.exists():
        print(f"[!] Feature dataset not found at {data_path}. Run MapReduce first!")
        return

    print("=================================================================")
    print("  Training Decision Tree & Random Forest Surge Classifiers")
    print("=================================================================")

    df = pd.read_csv(data_path)
    print(f"[*] Loaded {len(df)} spatio-temporal feature instances.")
    
    # Feature columns and target label
    feature_cols = [
        "zone_id", "hour", "day_of_week", "is_weekend", "is_rush_hour",
        "demand_trips", "avg_fare", "avg_miles", "avg_seconds", "avg_speed_mph"
    ]
    target_col = "is_surge_above_1_5x"

    # Clean data
    df_clean = df.dropna(subset=feature_cols + [target_col])
    X = df_clean[feature_cols]
    y = df_clean[target_col].astype(int)

    surge_count = y.sum()
    print(f"[*] Target Distribution: Total={len(y)} | Surge (>1.5x)={surge_count} ({surge_count/len(y)*100:.1f}%) | Normal={len(y)-surge_count}")

    # Train / Test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if surge_count > 5 else None
    )

    # 1. Decision Tree Classifier (Interpretable baseline)
    print("\n-----------------------------------------------------------------")
    print(" 1. Decision Tree Classifier")
    print("-----------------------------------------------------------------")
    dt = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    y_prob_dt = dt.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else y_pred_dt

    print(classification_report(y_test, y_pred_dt, target_names=["Normal (<1.5x)", "Surge (>=1.5x)"]))
    if len(np.unique(y_test)) > 1:
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_dt):.4f}")

    # 2. Random Forest Classifier (Ensemble)
    print("\n-----------------------------------------------------------------")
    print(" 2. Random Forest Classifier (Ensemble)")
    print("-----------------------------------------------------------------")
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else y_pred_rf

    print(classification_report(y_test, y_pred_rf, target_names=["Normal (<1.5x)", "Surge (>=1.5x)"]))
    if len(np.unique(y_test)) > 1:
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_rf):.4f}")

    # Feature Importance Ranking
    print("\n-----------------------------------------------------------------")
    print("  Feature Importance Ranking (Random Forest):")
    print("-----------------------------------------------------------------")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for rank, idx in enumerate(indices):
        print(f"  {rank+1:02d}. {feature_cols[idx]:<18} : {importances[idx]*100:.2f}%")

    # Save trained models
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dt_path = MODEL_DIR / "decision_tree_surge_model.pkl"
    rf_path = MODEL_DIR / "random_forest_surge_model.pkl"
    joblib.dump(dt, dt_path)
    joblib.dump(rf, rf_path)
    print(f"\n[] Saved serialized models to:")
    print(f"    - {dt_path}")
    print(f"    - {rf_path}")
    print("=================================================================")

if __name__ == "__main__":
    train_models()
