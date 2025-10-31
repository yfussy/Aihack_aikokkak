import os
import sys
import shutil
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from catboost import CatBoostClassifier, Pool, cv
import optuna
import json
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve
)
from imblearn.combine import SMOTETomek

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

def trainCatboost(version: int, train_df, categorical_columns: list, hypertune=False) -> float:
    FEATURE_PATH = getModelDir("feature", version, "catboost")
    MODEL_PATH = getModelDir("model", version, "catboost")
    PARAM_PATH = getModelDir("param", version, "catboost")

    # ------------------- PREP -------------------
    target = "default_12month"
    if target not in train_df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset!")

    X = train_df.drop(columns=[target])
    y = train_df[target]

    # Ensure categorical columns exist
    categorical_features = [col for col in categorical_columns if col in X.columns]

    # Convert categorical columns to string (CatBoost requires this when numbers are used)
    X[categorical_features] = X[categorical_features].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    eval_pool = Pool(X_test, y_test, cat_features=categorical_features)

    print("GPU detected -> using CatBoost GPU training." if _detect_gpu() else "No GPU detected -> using CPU training.")

    skf = StratifiedKFold(n_splits=5 if hypertune else 3, shuffle=True, random_state=42)

    # ----------- OPTUNA TUNING -----------
    def objective(trial):
        # GPU detection
        if _detect_gpu():
            task_type = "GPU"
            bootstrap_type = "Bayesian" 
            subsample = None
            bagging_temperature = trial.suggest_float("bagging_temperature", 0, 1)
            iterations = trial.suggest_int("iterations", 500, 1500) if hypertune else trial.suggest_int("iterations", 400, 1000)
            rsm = trial.suggest_float("rsm", 0.7, 1.0) if not _detect_gpu() else None
            simple_ctr = trial.suggest_categorical("simple_ctr", ["Borders", "Buckets"])
        else:
            task_type = "CPU"
            bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bernoulli"])
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            bagging_temperature = None
            iterations = trial.suggest_int("iterations", 400, 1000)
            rsm = None
            simple_ctr = trial.suggest_categorical("simple_ctr", ["Borders", "Counter", "Buckets", "BinarizedTargetMeanValue"])

        params = {
            "task_type": task_type,
            "bootstrap_type": bootstrap_type,
            "bagging_temperature": bagging_temperature,
            "subsample": subsample,
            "iterations": iterations,
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 5.0, log=True),
            "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
            "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 5),
            "leaf_estimation_method": trial.suggest_categorical("leaf_estimation_method", ["Newton", "Gradient"]),
            "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 10),
            "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 1, 3),
            "auto_class_weights": trial.suggest_categorical("auto_class_weights", ["Balanced", None]),
            "od_type": trial.suggest_categorical("od_type", ["IncToDec"]),
            "od_wait": trial.suggest_int("od_wait", 30, 100),
            "rsm": rsm,
            "simple_ctr": simple_ctr,
        }

        model = CatBoostClassifier(**params)
        aucs = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            pool_train = Pool(X_train, y_train, cat_features=categorical_features)
            pool_val = Pool(X_val, y_val, cat_features=categorical_features)
            model.fit(pool_train, eval_set=pool_val, verbose=False, early_stopping_rounds=100)
            preds = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, preds))
        return np.mean(aucs)

    print("Running hyperparameter tuning with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20 if hypertune else 10, timeout=None, show_progress_bar=True)

    print("\nBest Parameters Found:", study.best_params)

    param_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_score": study.best_value,
        "best_params": study.best_params
    }

    with open(PARAM_PATH, "a") as f:
        json.dump(param_data, f, indent=4)
        f.write("\n")  # one run per line

    print("Logged best parameters to " + PARAM_PATH + ".json")

    # ----------- FINAL MODEL TRAINING -----------

    final_params = {
        **study.best_params,
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "random_seed": 42,
        "task_type": "GPU" if _detect_gpu() else "CPU",
        "verbose": 200,
    }

    final_model = CatBoostClassifier(**final_params)
    final_model.fit(train_pool, eval_set=eval_pool, use_best_model=True, plot=True)

    y_proba = final_model.predict_proba(X_test)[:, 1]

    # Optimal threshold by F1
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pred = (y_proba >= best_threshold).astype(int)
    threshold = round(best_threshold, 3)

    print("\nOptimal Threshold:", threshold)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

    feature_names = X.columns.tolist()
    joblib.dump(feature_names, FEATURE_PATH)
    final_model.save_model(MODEL_PATH)

    print("\nCatBoost model training complete and saved!")

    return threshold

def testCatboost(version: int, test_df, ids, threshold, categorical_columns):
    FEATURE_PATH = getModelDir("feature", version, "catboost")
    MODEL_PATH = getModelDir("model", version, "catboost")
    PRED_PATH = getPredDir(version, "prediction_catboost")

    feature_names = joblib.load(FEATURE_PATH)
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    test_df = test_df[feature_names]

    cat_features = [col for col in categorical_columns if col in test_df.columns]
    test_df[cat_features] = test_df[cat_features].astype(str)

    y_proba = model.predict_proba(test_df)[:, 1]
    pred = (y_proba >= threshold).astype(int)

    output_df = pd.DataFrame({
        "ID": ids,
        "default_12month": pred,
    })

    output_df.to_csv(PRED_PATH, index=False)
    print(f"\nPredictions saved to: {PRED_PATH}")
    print("Training + prediction complete.")