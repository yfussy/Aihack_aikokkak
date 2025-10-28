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
from sklearn.model_selection import train_test_split
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

def trainCatboost(version: int, train_df) -> float:
    FEATURE_PATH = getModelDir("feature", version, "catboost")
    SCALER_PATH = getModelDir("scaler", version, "catboost")
    MODEL_PATH = getModelDir("model", version, "catboost")
    PARAM_PATH = getModelDir("param", version, "catboost")

    # ------------------- PREP -------------------
    target = "default_12month"
    X = train_df.drop(columns=[target])
    y = train_df[target]

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    joblib.dump(scaler, SCALER_PATH)

    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    X_train[categorical_features] = X_train[categorical_features].astype(str)
    X_test[categorical_features] = X_test[categorical_features].astype(str)
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    eval_pool = Pool(X_test, y_test, cat_features=categorical_features)

    print("GPU detected -> using CatBoost GPU training." if _detect_gpu() else "No GPU detected -> using CPU training.")

    # ----------- OPTUNA TUNING -----------
    def objective(trial):
    # GPU-safe bootstrap
        bootstrap_type = "Bayesian" if _detect_gpu() else trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "Poisson"])

        params = {
            "iterations": trial.suggest_int("iterations", 300, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "bootstrap_type": bootstrap_type,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "AUC",
            "loss_function": "Logloss",
            "verbose": False,
            "task_type": "GPU" if _detect_gpu() else "CPU",
            "devices": "0" if _detect_gpu() else None,
        }

        # bagging_temperature only allowed for Bayesian
        if bootstrap_type == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 1)
        
        # subsample only for Bernoulli or Poisson on CPU
        if bootstrap_type in ["Bernoulli", "Poisson"] and not _detect_gpu():
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=50, verbose=False)

        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)

        return auc

    print("Running hyperparameter tuning with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25, show_progress_bar=True)

    print("\nBest Parameters Found:")
    print(study.best_params)
    best_params = study.best_params

    param_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_score": study.best_value,
        "best_params": best_params
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
        "scale_pos_weight": scale_pos_weight,
        "random_seed": 42,
        "task_type": "GPU" if _detect_gpu() else "CPU",
        "verbose": 200,
        "early_stopping_rounds": 100,
    }

    final_model = CatBoostClassifier(**final_params)
    final_model.fit(train_pool, eval_set=eval_pool, use_best_model=True)

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

def testCatboost(version: int, test_df, ids, threshold):
    FEATURE_PATH = getModelDir("feature", version, "catboost")
    SCALER_PATH = getModelDir("scaler", version, "catboost")
    MODEL_PATH = getModelDir("model", version, "catboost")
    PRED_PATH = getPredDir(3, "prediction_catboost")

    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_PATH)
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    test_df = test_df[feature_names]

    num_features = test_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    test_df[num_features] = scaler.transform(test_df[num_features])

    cat_features = [col for col in feature_names if col not in num_features]
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