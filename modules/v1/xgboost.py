import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

import sys
import os
root_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if root_path not in sys.path:
    sys.path.append(root_path)
from fileDir import getModelDir, getPredDir

def trainTestXgboost(version: int, train_df, test_df, ids):
    FEATURE_PATH = getModelDir("train_features_model_xgboost", version, True)
    SCALER_PATH = getModelDir("scaler_model_xgboost", version, True)
    MODEL_PATH = getModelDir("model_xgboost", version, True)
    PRED_PATH = getPredDir(version, "prediction_xgboost")

    # TRAIN ---------------------------------------------------------------
    target = "default_12month"
    X = train_df.drop(columns=[target])
    y = train_df[target]

    X = pd.get_dummies(X, drop_first=True)

    feature_names = X.columns.tolist()
    joblib.dump(feature_names, FEATURE_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [10],
        "learning_rate": [0.001, 0.015, 0.02],
        "subsample": [0.8, 0.9, 1.0], # bound [0,1]
        "colsample_bytree": [0.8, 0.9, 1.0], # bound [0,1]
        "min_child_weight": [0.4, 0.5, 0.6],
        "gamma": [0.2, 0.25, 0.3],
    }

    base_model = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=25,
        scoring="roc_auc",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )

    print("Tuning hyperparameters...")
    search.fit(X_train_res, y_train_res)
    best_model = search.best_estimator_

    print("Best Parameters Found:", search.best_params_)

    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Automatically find the best threshold based on F1
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    threshold = round(best_threshold, 3)

    y_pred = (y_proba >= best_threshold).astype(int)

    print("\nOptimal Threshold:", threshold)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

    joblib.dump(best_model, MODEL_PATH)

    # TEST----------------------------------------------------------------
    X_test_xgb = scaler.transform(train_df)

    y_proba = best_model.predict_proba(X_test_xgb)[:, 1]

    pred = (y_proba >= threshold).astype(int)

    output_df = pd.DataFrame({
        "ID": ids,
        "default_12month": pred
    })

    output_df.to_csv(PRED_PATH, index=False)