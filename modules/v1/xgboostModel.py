import os
import sys
import shutil
import joblib
import numpy as np
import pandas as pd
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
import optuna
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
    from xgboost import XGBClassifier, DMatrix, train
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
from fileDir import getModelDir, getPredDir

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

# ==================== SETTINGS ====================
if (_detect_gpu()):
    N_SPLITS = 4
    N_ITER = 30
else:
    N_SPLITS = 3
    N_ITER = 10
# ==================== SETTINGS ====================

def trainXgboost(version: int, train_df) -> float:
    """
    Returns threshold value -> sends this value to testXgboost()
    """
    FEATURE_PATH = getModelDir("feature", version, "xgboost")
    MODEL_PATH = getModelDir("model", version, "xgboost")
    PARAM_PATH = getModelDir("param", version, "xgboost")

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

    # ------------------- GPU DETECTION -------------------
    gpu_available = _detect_gpu()
    if gpu_available:
        device = "cuda"
        print("GPU detected -> using XGBoost GPU training (gpu_hist).")
    else:
        device = "cpu"
        print("No GPU detected -> using XGBoost CPU training (hist).")

    # ------------------- MODEL + SEARCH -------------------

    imbalance_ratio = (y_train.value_counts().max() / y_train.value_counts().min())

    def objective(trial):
        # parameter search space
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "scale_pos_weight": imbalance_ratio,
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True)
        }

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_idx, valid_idx in kf.split(X, y):
            dtrain = DMatrix(X.iloc[train_idx], label=y.iloc[train_idx])
            dvalid = DMatrix(X.iloc[valid_idx], label=y.iloc[valid_idx])

            model = train(
                params,
                dtrain,
                num_boost_round=3000,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=100,
                verbose_eval=False
            )

            preds = model.predict(dvalid)
            score = roc_auc_score(y[valid_idx], preds)
            scores.append(score)

            # Optuna prune bad trials early
            trial.report(np.mean(scores), len(scores))
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)


    # Run optimization
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("Best AUC:", study.best_value)
    print("Best Params:", study.best_params)


    param_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_score": study.best_value,
        "best_params": study.best_params
    }

    with open(PARAM_PATH, "a") as f:
        json.dump(param_data, f, indent=4)
        f.write("\n")  # one run per line

    print("Logged best parameters to " + PARAM_PATH + ".json")

    best_params = {
        **study.best_params,
        "loss_function": "Logloss",
        "random_seed": 42
    }

    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)

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

    return threshold, y_proba_valid

def testXgboost(version, test_df, ids, threshold):
    FEATURE_PATH = getModelDir("feature", version, "xgboost")
    MODEL_PATH = getModelDir("model", version, "xgboost")
    PRED_PATH = getPredDir(version, "prediction_xgboost")

    model: XGBClassifier = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)
    test_encoded = pd.get_dummies(test_df, drop_first=True)
    test_encoded = test_encoded.reindex(columns=feature_names, fill_value=0)

    y_proba_test = model.predict_proba(test_encoded)[:, 1]
    pred = (y_proba_test >= threshold).astype(int)

    output_df = pd.DataFrame({
        "ID": ids,
        "default_12month": pred,
    })

    output_df.to_csv(PRED_PATH, index=False)
    print(f"\nPredictions saved to: {PRED_PATH}")
    print("Training + prediction complete.")

    return y_proba_test