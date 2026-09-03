import os
import sys
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import shap
import lime
import lime.lime_tabular

# UTF-8 stdout configuration
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def log(msg: str):
    print(msg, flush=True)

DATA_DIR = os.path.join(".", "data")
DATASET_JSON = os.path.join(DATA_DIR, "peer_market_dataset.json")

# Define feature columns (Step 3: Feature Fusion)
FEATURE_COLS = [
    "feat_momentum_5d",
    "feat_volume_surge",
    "sent_asset_quality_net",
    "sent_asset_quality_pos",
    "sent_asset_quality_neg",
    "sent_margins_profitability_net",
    "sent_margins_profitability_pos",
    "sent_margins_profitability_neg",
    "sent_balance_sheet_net",
    "sent_balance_sheet_pos",
    "sent_balance_sheet_neg",
    "sent_capital_liquidity_net",
    "sent_capital_liquidity_pos",
    "sent_capital_liquidity_neg",
    "sent_operating_efficiency_net",
    "sent_operating_efficiency_pos",
    "sent_operating_efficiency_neg"
]

TARGET_COL = "target_label"

def load_data():
    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # Sort strictly chronologically by call_date
    df["call_date_dt"] = pd.to_datetime(df["call_date"])
    df = df.sort_values("call_date_dt").reset_index(drop=True)
    return df

CLASSES = ["down", "flat", "up"]

def train_model(df, cutoff_date="2025-07-01"):
    train_df = df[df["call_date_dt"] <= pd.to_datetime(cutoff_date)].copy()
    
    le = LabelEncoder()
    le.fit(CLASSES)
    
    X_train = train_df[FEATURE_COLS].values
    y_train = le.transform(train_df[TARGET_COL].values)
    
    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)
    return model, le

def time_aware_split(df, cutoff_date="2025-07-01"):
    """
    Step 4: Time-Aware Chronological Train/Test Split.
    NEVER random shuffle across quarters to prevent lookahead data leakage!
    """
    df["call_date_dt"] = pd.to_datetime(df["call_date"])
    cutoff_dt = pd.to_datetime(cutoff_date)

    train_df = df[df["call_date_dt"] <= cutoff_dt].copy()
    test_df = df[df["call_date_dt"] > cutoff_dt].copy()

    return train_df, test_df

def run_ml_pipeline():
    log("=========================================================================")
    log("  STEPS 3-6: XGBOOST ML MODELING, CHRONOLOGICAL SPLIT & SHAP/LIME       ")
    log("=========================================================================")

    df = load_data()
    log(f"[+] Loaded Pooled Dataset: {len(df)} event rows across 10 peer banks.")
    log(f"[+] Date Range: {df['call_date'].min()} to {df['call_date'].max()}")

    # Step 4: Time-Aware Train/Test Split
    cutoff_date = "2025-07-01"
    train_df, test_df = time_aware_split(df, cutoff_date)

    log(f"\n[Step 4] Time-Aware Chronological Split (Cutoff: {cutoff_date}):")
    log(f"  • Train Set : {len(train_df)} rows ({train_df['call_date'].min()} to {train_df['call_date'].max()})")
    log(f"  • Test Set  : {len(test_df)} rows ({test_df['call_date'].min()} to {test_df['call_date'].max()})")

    # Encode categorical labels ('down' -> 0, 'flat' -> 1, 'up' -> 2)
    le = LabelEncoder()
    le.fit(["down", "flat", "up"])
    
    X_train = train_df[FEATURE_COLS].values
    y_train = le.transform(train_df[TARGET_COL].values)

    X_test = test_df[FEATURE_COLS].values
    y_test = le.transform(test_df[TARGET_COL].values)

    # Step 5: Fit XGBoost Classifier
    log(f"\n[Step 5] Fitting XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Naive Majority Class Baseline Comparison
    majority_class = train_df[TARGET_COL].mode()[0]
    y_naive = le.transform([majority_class] * len(test_df))
    naive_acc = accuracy_score(y_test, y_naive)
    xgb_acc = accuracy_score(y_test, y_pred)
    xgb_f1 = f1_score(y_test, y_pred, average="weighted")

    log("\n=========================================================================")
    log("  MODEL EVALUATION METRICS (XGBOOST VS NAIVE BASELINE)                   ")
    log("=========================================================================")
    log(f"  • Naive Majority Baseline Accuracy : {naive_acc * 100:.2f}% (Predicting '{majority_class}')")
    log(f"  • XGBoost Model Accuracy           : {xgb_acc * 100:.2f}%")
    log(f"  • XGBoost Model Weighted F1-Score  : {xgb_f1 * 100:.2f}%\n")

    log("CLASSIFICATION REPORT:")
    target_names = list(le.classes_)
    print(classification_report(y_test, y_pred, target_names=target_names))

    log("CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=[f"Actual_{c}" for c in target_names], columns=[f"Pred_{c}" for c in target_names])
    print(cm_df.to_string())

    # Step 6A: SHAP Global Feature Importance
    log("\n=========================================================================")
    log("  STEP 6A: SHAP GLOBAL FEATURE IMPORTANCE ANALYSIS                       ")
    log("=========================================================================")
    explainer = shap.TreeExplainer(model)
    shap_obj = explainer(X_test)

    if hasattr(shap_obj, "values"):
        sv = shap_obj.values
        if len(sv.shape) == 3:
            mean_abs_shap = np.abs(sv).mean(axis=(0, 2))
        else:
            mean_abs_shap = np.abs(sv).mean(axis=0)
    else:
        raw_sv = np.array(explainer.shap_values(X_test))
        mean_abs_shap = np.abs(raw_sv).mean(axis=tuple(range(len(raw_sv.shape) - 1)))

    mean_abs_shap = np.ravel(mean_abs_shap)

    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "shap_importance": mean_abs_shap
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)

    print(importance_df.head(10).to_string(index=False))

    # Step 6B: LIME Local Prediction Explanation
    log("\n=========================================================================")
    log("  STEP 6B: LIME LOCAL CONCALL PREDICTION EXPLANATION                     ")
    log("=========================================================================")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=FEATURE_COLS,
        class_names=target_names,
        mode="classification",
        random_state=42
    )

    # Explain first instance in test set (Axis Bank concall)
    sample_idx = 0
    sample_row = test_df.iloc[sample_idx]
    sample_company = sample_row["company"]
    sample_date = sample_row["call_date"]
    actual_label = sample_row[TARGET_COL]
    pred_label = le.inverse_transform([y_pred[sample_idx]])[0]

    log(f"[+] Explaining prediction for instance #{sample_idx} ({sample_company} on {sample_date}):")
    log(f"    • Actual Target Label   : {actual_label.upper()}")
    log(f"    • Predicted Target Label: {pred_label.upper()}")
    log(f"    • Prediction Probabilities: Down: {y_pred_proba[sample_idx][0]:.2f} | Flat: {y_pred_proba[sample_idx][1]:.2f} | Up: {y_pred_proba[sample_idx][2]:.2f}\n")

    exp = lime_explainer.explain_instance(
        data_row=X_test[sample_idx],
        predict_fn=model.predict_proba,
        num_features=6
    )

    log("LIME FEATURE CONTRIBUTION WEIGHTS:")
    for feat, weight in exp.as_list():
        log(f"  • {feat:40s} : {weight:+.4f}")

if __name__ == "__main__":
    run_ml_pipeline()
