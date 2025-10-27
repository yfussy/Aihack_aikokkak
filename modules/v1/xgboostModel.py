import os
import sys
import shutil
import joblib
import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from imblearn.over_sampling import SMOTE

# try to import xgboost and detect version
try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError(
        "xgboost is not installed or failed to import. "
        "To enable GPU training install a GPU-enabled xgboost build (or CPU-only if no GPU). "
        "For pip GPU-enabled builds: `pip install xgboost` (ensure it was built with CUDA), "
        "or use conda: `conda install -c rapidsai -c nvidia -c conda-forge xgboost`."
    ) from e

# project path setup (same as your original)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.append(root_path)
from fileDir import getModelDir, getPredDir, getDataDir


# ==================== SETTINGS ====================
N_SPLITS = 4
N_ITER = 30
# ==================== SETTINGS ====================

def _detect_gpu():
    """
    Return True if a CUDA-capable GPU is likely available.
    Checks for nvidia-smi in PATH or CUDA_VISIBLE_DEVICES env var.
    This is a best-effort heuristic — presence of nvidia-smi is a good signal.
    """
    # check CUDA_VISIBLE_DEVICES (common in containers)
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_env is not None and cuda_env.strip() not in ("", "-1"):
        return True

    # check if nvidia-smi executable is available
    if shutil.which("nvidia-smi") is not None:
        return True

    # fallback: check common CUDA env var
    if os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME"):
        return True

    return False


def trainTestXgboost(version: int, train_df, test_df, ids):
    FEATURE_PATH = getModelDir("train_features_model_xgboost", version, True)
    SCALER_PATH = getModelDir("scaler_model_xgboost", version, True)
    MODEL_PATH = getModelDir("model_xgboost", version, True)
    PRED_PATH = getPredDir(version, "prediction_xgboost")

    # ------------------- PREP -------------------
    target = "default_12month"
    X = train_df.drop(columns=[target])
    y = train_df[target]

    # encode categoricals (assume feature engineering done prior)
    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, FEATURE_PATH)

    # stratified split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    joblib.dump(scaler, SCALER_PATH)

    # SMOTE (apply if imbalance is significant)
    imbalance_ratio = (y_train.value_counts().max() / y_train.value_counts().min())
    if imbalance_ratio > 1.5:
        sm = SMOTE(random_state=42, sampling_strategy="auto")
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        print(f"SMOTE applied. Imbalance ratio before: {imbalance_ratio:.2f}")
    else:
        X_train_res, y_train_res = X_train, y_train
        print(f"SMOTE skipped. Imbalance ratio: {imbalance_ratio:.2f}")

    # ------------------- GPU DETECTION -------------------
    gpu_available = _detect_gpu()
    if gpu_available:
        device = "cuda"
        print("GPU detected -> using XGBoost GPU training (gpu_hist).")
    else:
        device = "cpu"
        print("No GPU detected -> using XGBoost CPU training (hist).")

    # ------------------- MODEL + SEARCH -------------------
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,
        random_state=42,
        early_stopping_rounds=50,
    )

    param_grid = {
        # "n_estimators": [300, 400, 600],
        # "max_depth": [6, 8, 10],
        # "learning_rate": [0.005, 0.01, 0.02],
        # "subsample": [0.6, 0.8, 1.0],
        # "colsample_bytree": [0.6, 0.8, 1.0],
        # "min_child_weight": [1, 3, 5],
        # "gamma": [0, 0.1, 0.2, 0.3],
        # "reg_alpha": [0, 0.01, 0.1],
        # "reg_lambda": [1, 1.5, 2],
        # # scale_pos_weight helps with imbalance if desired (tune or set based on class ratio)
        # "scale_pos_weight": [1, max(1, int(imbalance_ratio // 1)), max(1, int(imbalance_ratio // 1) * 2)],
        "n_estimators": [300],
        "max_depth": [10],
        "learning_rate": [0.02],
        "subsample": [0.8],
        "colsample_bytree": [1.0],
        "min_child_weight": [5],
        "gamma": [0.2],
        "reg_alpha": [0],
        "reg_lambda": [1],
        "scale_pos_weight": [5],
    }

    # use stratified k-fold for outer cross-validation in RandomizedSearch
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=N_ITER,
        scoring="roc_auc",
        cv=skf,
        verbose=1,
        n_jobs=-1,
        random_state=42,
        return_train_score=False,
    )

    # We pass early stopping via fit_params so each candidate uses early stopping on the validation set.
    fit_params = {
        "eval_set": [(X_valid, y_valid)],
        "verbose": False,
    }

    print("Tuning hyperparameters with early stopping (using validation set)...")
    search.fit(X_train_res, y_train_res, **fit_params)
    best_model = search.best_estimator_
    print("Best Parameters Found:", search.best_params_)

    best_model.set_params(tree_method="hist", device="cuda")

    best_model.fit(
        X_train_res,
        y_train_res,
        eval_set=[(X_valid, y_valid)],
        verbose=True,
    )

    # ------------------- VALIDATION METRICS -------------------
    y_proba_valid = best_model.predict_proba(X_valid)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_valid, y_proba_valid)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    # thresholds has length len(precision)-1; guard against empty
    if thresholds.size == 0:
        best_threshold = 0.5
    else:
        best_threshold = thresholds[np.argmax(f1_scores)]
    threshold = float(np.round(best_threshold, 3))

    y_pred_valid = (y_proba_valid >= best_threshold).astype(int)
    auc = roc_auc_score(y_valid, y_proba_valid)
    avg_precision = average_precision_score(y_valid, y_proba_valid)

    print(f"\nOptimal Threshold (F1-based): {threshold}")
    print("\nConfusion Matrix (validation):")
    print(confusion_matrix(y_valid, y_pred_valid))
    print("\nClassification Report (validation):")
    print(classification_report(y_valid, y_pred_valid))
    print(f"ROC-AUC (validation): {auc:.4f}")
    print(f"PR-AUC / Average Precision (validation): {avg_precision:.4f}")

    # persist artifacts
    joblib.dump(best_model, MODEL_PATH)

    # ------------------- TEST PREDICTION -------------------
    # ensure test columns match training features
    test_encoded = pd.get_dummies(test_df, drop_first=True)
    test_encoded = test_encoded.reindex(columns=feature_names, fill_value=0)
    test_encoded = scaler.transform(test_encoded)

    y_proba_test = best_model.predict_proba(test_encoded)[:, 1]
    pred = (y_proba_test >= threshold).astype(int)

    output_df = pd.DataFrame({
        "ID": ids,
        "default_12month": pred,
    })

    output_df.to_csv(PRED_PATH, index=False)
    print(f"\nPredictions saved to: {PRED_PATH}")
    print("Training + prediction complete.")