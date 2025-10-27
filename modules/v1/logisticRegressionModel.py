import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib
import sys
import os
root_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if root_path not in sys.path:
    sys.path.append(root_path)
from fileDir import getModelDir, getPredDir

def trainLogisticRegression(version: int, train_df, test_df, ids):
    FEATURE_PATH = getModelDir("train_features_model_logistic", version, True)
    SCALER_PATH = getModelDir("scaler_model_logistic", version, True)
    MODEL_PATH = getModelDir("model_logistic", version, True)
    PRED_PATH = getPredDir(version, "prediction_logistic")

    # TRAIN ---------------------------------------------------------------
    target = "default_12month"
    X = train_df.drop(columns=[target])
    y = train_df[target]

    X = pd.get_dummies(X, drop_first=True)

    feature_names = X.columns.tolist()
    joblib.dump(feature_names, FEATURE_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = LogisticRegression(
        solver="lbfgs",  # works well for smaller datasets and L1/L2 regularization
        class_weight="balanced",  # helps with imbalance
        max_iter=1000,
    )
    model.fit(X_train_res, y_train_res)

    y_proba = model.predict_proba(X_test)[:, 1]

    # Find best threshold using F1
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

    joblib.dump(model, MODEL_PATH)

    # TEST----------------------------------------------------------------
    train_encoded = pd.get_dummies(train_df.drop(columns=["default_12month"], errors="ignore"), drop_first=True)
    test_encoded  = pd.get_dummies(test_df, drop_first=True)

    # Align test with training features
    _, test_encoded = train_encoded.align(test_encoded, join="left", axis=1, fill_value=0)

    X_test = scaler.transform(test_encoded)

    y_proba = model.predict_proba(X_test)[:, 1]
    pred = (y_proba >= threshold).astype(int)

    output_df = pd.DataFrame({
        "ID": ids,
        "default_12month": pred
    })

    output_df.to_csv(PRED_PATH, index=False)