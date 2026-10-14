"""
Machine Learning Baseline: Decision Tree & Random Forest Surge Classifiers.
Predicts 20-minute advance surge multiplier (> 1.5x) based on leak-free MapReduce features.
Implements temporal train/test split (Jan-Sep Train vs Oct-Dec Test) and threshold tuning.
"""
import sys
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from config.config import PROCESSED_DATA_DIR, MODEL_DIR

def train_baseline_models(data_path: Path = PROCESSED_DATA_DIR / "features_labels.csv"):
    if not data_path.exists():
        print(f"[!] Feature dataset not found at {data_path}. Run MapReduce first!")
        return

    print("=================================================================")
    print("  Training Decision Tree & Random Forest Surge Classifiers")
    print("  (20-Minute Forward Lookahead | Zero Target Leakage)")
    print("=================================================================")

    df = pd.read_csv(data_path)
    print(f"[*] Loaded {len(df)} spatio-temporal feature instances across all 12 months.")

    feature_cols = [
        "zone_id", "hour", "day_of_week", "is_weekend", "is_rush_hour",
        "demand_trips", "dropoff_trips", "inflow_outflow_ratio", "demand_momentum", "avg_speed_mph"
    ]
    target_col = "is_surge_in_20min"

    df_clean = df.dropna(subset=feature_cols + [target_col])

    # Temporal split: Months 1-9 for Training, Months 10-12 for Out-of-Time Testing
    df_clean["month"] = pd.to_datetime(df_clean["time_bucket"]).dt.month
    train_mask = df_clean["month"] <= 9
    test_mask = df_clean["month"] > 9

    X_train = df_clean.loc[train_mask, feature_cols]
    y_train = df_clean.loc[train_mask, target_col].astype(int)
    X_test = df_clean.loc[test_mask, feature_cols]
    y_test = df_clean.loc[test_mask, target_col].astype(int)

    print(f"[*] Temporal Split: Train (Jan-Sep)={len(X_train)} samples ({y_train.sum()} surges) | Test (Oct-Dec)={len(X_test)} samples ({y_test.sum()} surges)")

    # 1. Decision Tree Classifier (Interpretable baseline)
    print("\n-----------------------------------------------------------------")
    print(" 1. Decision Tree Classifier")
    print("-----------------------------------------------------------------")
    dt = DecisionTreeClassifier(max_depth=6, class_weight="balanced", min_samples_split=20, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    y_prob_dt = dt.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred_dt, target_names=["Normal (<1.5x)", "Surge (>=1.5x)"]))
    roc_dt = roc_auc_score(y_test, y_prob_dt)
    print(f"Decision Tree ROC-AUC Score: {roc_dt:.4f}")

    # 2. Random Forest Classifier (Ensemble)
    print("\n-----------------------------------------------------------------")
    print(" 2. Random Forest Classifier (Ensemble)")
    print("-----------------------------------------------------------------")
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, class_weight="balanced", min_samples_split=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    # Standard 0.5 threshold
    y_pred_rf = (y_prob_rf >= 0.5).astype(int)
    print("Standard Threshold (0.50):")
    print(classification_report(y_test, y_pred_rf, target_names=["Normal (<1.5x)", "Surge (>=1.5x)"]))
    roc_rf = roc_auc_score(y_test, y_prob_rf)
    print(f"Random Forest ROC-AUC Score: {roc_rf:.4f}")

    # Tuned Threshold for High-Recall Pre-Positioning (Threshold = 0.35)
    tuned_thresh = 0.35
    y_pred_tuned = (y_prob_rf >= tuned_thresh).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_tuned, average="binary")
    print(f"\nTuned Threshold for Fleet Pre-Positioning ({tuned_thresh:.2f}):")
    print(f"  - Recall (Surge Caught)    : {rec*100:.2f}% (Critical for pre-dispatching idle vehicles)")
    print(f"  - Precision                : {prec*100:.2f}%")
    print(f"  - F1-Score                 : {f1:.4f}")

    # Feature Importance Ranking
    print("\n-----------------------------------------------------------------")
    print("  Feature Importance Ranking (Random Forest):")
    print("-----------------------------------------------------------------")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for rank, idx in enumerate(indices):
        print(f"  {rank+1:02d}. {feature_cols[idx]:<22} : {importances[idx]*100:.2f}%")

    # Save models
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dt_path = MODEL_DIR / "decision_tree_surge_model.pkl"
    rf_path = MODEL_DIR / "random_forest_surge_model.pkl"
    joblib.dump(dt, dt_path)
    joblib.dump(rf, rf_path)
    print(f"\n[] Saved serialized models to:")
    print(f"    - {dt_path.relative_to(PROJECT_ROOT)}")
    print(f"    - {rf_path.relative_to(PROJECT_ROOT)}")
    print("=================================================================")

if __name__ == "__main__":
    train_baseline_models()
