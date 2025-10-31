import os, sys, json, joblib, numpy as np, pandas as pd
from datetime import datetime
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
)
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# ===== utils เดิมของโปรเจกต์ =====
# root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# if root_path not in sys.path:
#     sys.path.append(root_path)
from fileDir import getModelDir, getPredDir

def fix_dtypes(df: pd.DataFrame, numeric_like_threshold: float = 0.9) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        s = df[c]
        coerced = pd.to_numeric(s, errors="coerce")
        ratio_numeric = coerced.notna().mean()
        if ratio_numeric >= numeric_like_threshold:
            df[c] = coerced 
        else:
            df[c] = s.astype("category")
    try:
        str_cols = df.select_dtypes(include=["string"]).columns.tolist()
        for c in str_cols:
            df[c] = df[c].astype("category")
    except Exception:
        pass
    return df


# ---------- TRAIN ----------
def trainLGBM(version: int, train_df: pd.DataFrame, hypertune: bool=False) -> float:
    FEATURE_PATH = getModelDir("feature", version, "lightgbm")
    MODEL_PATH   = getModelDir("model",   version, "lightgbm")
    PARAM_PATH   = getModelDir("param",   version, "lightgbm")

    target = "default_12month"
    if target not in train_df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset!")

    X = train_df.drop(columns=[target]).copy()
    y = train_df[target].astype(int)

    X = fix_dtypes(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = fix_dtypes(X_train)
    X_test  = fix_dtypes(X_test)

    skf = StratifiedKFold(n_splits=(5 if hypertune else 3), shuffle=True, random_state=42)

    # ---- Optuna objective ----
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 400, 1800, step=200),
            "num_leaves":        trial.suggest_categorical("num_leaves", [63, 95,127,191, 255, 511]),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample":         trial.suggest_categorical("subsample", [0.8, 0.9, 1.0]),
            "colsample_bytree":  trial.suggest_categorical("colsample_bytree", [0.8, 0.9, 1.0]),
            "max_depth":         trial.suggest_categorical("max_depth", [-1, 8,10,12, 20]),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120, step=10),
            "min_gain_to_split": 0.0, 
            "max_bin":           trial.suggest_categorical("max_bin", [255,383, 511]),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 0.3, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 0.8, log=True),
            "class_weight":      trial.suggest_categorical("class_weight", [None, "balanced"]),
            "random_state":      42,
            "n_jobs":            -1,
        }
        aucs = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            X_tr = fix_dtypes(X_tr)
            X_va = fix_dtypes(X_va)

            model = LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                callbacks=[early_stopping(100), log_evaluation(0)],
            )
            preds = model.predict_proba(X_va)[:, 1]
            aucs.append(roc_auc_score(y_va, preds))
        return float(np.mean(aucs))

    # print("Running hyperparameter tuning (LightGBM, CPU, auto-categorical)…")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=(20 if hypertune else 10), show_progress_bar=True)

    # print("\nBest Parameters Found:", study.best_params)
    param_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_score": study.best_value,
        "best_params": study.best_params
    }
    with open(PARAM_PATH, "a") as f:
        json.dump(param_data, f)
        f.write("\n")
    # print("Logged best parameters to:", PARAM_PATH)

    # ---- final train ----
    final_params = {
        **study.best_params,
        "random_state": 42,
        "n_jobs": -1,
    }
    final_model = LGBMClassifier(**final_params)
    final_model.fit(
        X_train, y_train,
        eval_metric="auc",
        eval_set=[(X_test, y_test)],
        callbacks=[early_stopping(100), log_evaluation(50)],
    )

    # evaluate + pick threshold (F1)
    y_proba = final_model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    if len(thresholds) > 0:
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
    else:
        best_threshold = 0.5
    y_pred = (y_proba >= best_threshold).astype(int)
    threshold = float(np.round(best_threshold, 3))

    print("\nOptimal Threshold:", threshold)
    # print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    # print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

    # save artifacts
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, FEATURE_PATH)
    joblib.dump(final_model, MODEL_PATH)

    # print("\nLightGBM (CPU, auto-categorical) training complete and saved!")
    return threshold, y_proba

# ---------- TEST ----------
def testLGBM(version: int, test_df: pd.DataFrame, ids: pd.Series, threshold: float):
    FEATURE_PATH = getModelDir("feature", version, "lightgbm")
    MODEL_PATH   = getModelDir("model",   version, "lightgbm")
    PRED_PATH    = getPredDir(version, "prediction_lightgbm")

    feature_names = joblib.load(FEATURE_PATH)
    model: LGBMClassifier = joblib.load(MODEL_PATH)

    X_new = test_df[feature_names].copy()
    X_new = fix_dtypes(X_new)

    y_proba = model.predict_proba(X_new)[:, 1]
    pred = (y_proba >= threshold).astype(int)

    out = pd.DataFrame({
        "ID": ids,
        "default_12month": pred,
    })
    out = out[["ID", "default_12month"]].copy()
    out.to_csv(PRED_PATH, index=False)
    print(f"\nPredictions saved to: {PRED_PATH}")

    return y_proba
