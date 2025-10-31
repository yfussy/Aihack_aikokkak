from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import joblib
import optuna
import os, sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.append(root_path)
from fileDir import getModelDir, getPredDir

import numpy as np
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score,
    confusion_matrix
)

def find_best_thresholds(y_true, y_proba, beta=1.0, cost_fp=1.0, cost_fn=1.0,
                         require_recall=None, return_all=False):
    """
    Returns best thresholds according to several criteria.
    
    Parameters
    ----------
    y_true : array-like (n,)
        True binary labels (0/1).
    y_proba : array-like (n,)
        Predicted probabilities for positive class.
    beta : float
        Beta for F-beta (beta=1 -> F1).
    cost_fp, cost_fn : float
        Unit costs for false positive and false negative if using cost-based criterion.
    require_recall : float or None
        If set, find the threshold that gives the highest precision subject to recall >= require_recall.
    return_all : bool
        If True, also return arrays of thresholds/metrics for debugging or plotting.
        
    Returns
    -------
    best : dict
        Keys include:
        - 'youden_threshold', 'youden_sens', 'youden_spec'
        - 'f1_threshold', 'f1', 'precision_at_f1', 'recall_at_f1'
        - 'pr_threshold' (threshold that maximizes precision at required recall if given)
        - 'cost_threshold', 'cost' (threshold minimizing expected cost)
        - 'roc_threshold_closest' (closest-to-(0,1) point on ROC)
    If return_all True, also returns:
        roc_tpr, roc_fpr, roc_thresholds,
        pr_precision, pr_recall, pr_thresholds
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    assert y_true.shape[0] == y_proba.shape[0]

    # ---------- ROC / Youden ----------
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    # Note roc_thresholds are decreasing when using sklearn
    # Sensitivity = tpr, Specificity = 1 - fpr
    youden = tpr - fpr
    idx_you = np.argmax(youden)
    youden_threshold = roc_thresholds[idx_you]
    youden_sens = tpr[idx_you]
    youden_spec = 1 - fpr[idx_you]

    # ---------- Closest to (0,1) on ROC ----------
    # distance sqrt((1-tpr)^2 + fpr^2)
    dist = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
    idx_closest = np.argmin(dist)
    roc_threshold_closest = roc_thresholds[idx_closest]

    # ---------- Precision-Recall / F1 ----------
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns len = n_thresholds+1; thresholds aligned with precision[1:]
    # compute F1 for thresholds (align properly)
    # We'll compute F1 by scanning thresholds using confusion matrix instead (safer).
    unique_thresholds = np.unique(np.concatenate([roc_thresholds, pr_thresholds, [0.5]]))
    best_f1 = -1
    best_f1_t = 0.5
    best_prec = best_rec = None

    for t in unique_thresholds:
        preds = (y_proba >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        if (p + r) == 0:
            f1 = 0.0
        else:
            f1 = (1 + beta ** 2) * (p * r) / ((beta ** 2) * p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_t = t
            best_prec = p
            best_rec = r

    # ---------- Cost-based threshold ----------
    # For each threshold compute expected cost = FP * cost_fp + FN * cost_fn
    # Get counts from confusion matrix
    best_cost = np.inf
    best_cost_t = 0.5
    for t in unique_thresholds:
        preds = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = fp * cost_fp + fn * cost_fn
        if cost < best_cost:
            best_cost = cost
            best_cost_t = t

    # ---------- Precision at required recall ----------
    pr_result_threshold = None
    if require_recall is not None:
        # find thresholds from precision_recall_curve where recall >= require_recall
        # recall array from pr_curve is decreasing; iterate and find max precision with recall >= required
        pr_th = None
        pr_prec = -1
        for prec, rec, th in zip(precision[1:], recall[1:], pr_thresholds):
            if rec >= require_recall and prec > pr_prec:
                pr_prec = prec
                pr_th = th
        pr_result_threshold = pr_th

    result = {
        "youden_threshold": float(youden_threshold),
        "youden_sensitivity": float(youden_sens),
        "youden_specificity": float(youden_spec),

        "roc_threshold_closest": float(roc_threshold_closest),

        "f1_threshold": float(best_f1_t),
        "f1_score": float(best_f1),
        "precision_at_f1": float(best_prec),
        "recall_at_f1": float(best_rec),

        "cost_threshold": float(best_cost_t),
        "cost": float(best_cost),

        "precision_recall_threshold_given_recall": (float(pr_result_threshold) if pr_result_threshold is not None else None)
    }

    if return_all:
        return result, (tpr, fpr, roc_thresholds, precision, recall, pr_thresholds)
    return result


def trainTestStack(version: int, ids, train_df, p_train, p_test):
    PRED_PATH = getPredDir(version, "prediction_stack")

    y = train_df["default_12month"]

    def objective(trial):
        w1 = trial.suggest_float("w_xgb", 0, 1)
        w2 = trial.suggest_float("w_cat", 0, 1)
        w3 = trial.suggest_float("w_lgbm", 0, 1)
        weights = np.array([w1, w2, w3])
        weights /= weights.sum()
        blended = sum(w * p for w, p in zip(weights, p_train))
        return roc_auc_score(y, blended)
    
    # Run optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)

    print("✅ Best Score:", study.best_value)
    print("✅ Best Weights:", study.best_params)


    w = study.best_params
    weights = np.array([w["w_xgb"], w["w_cat"], w["w_lgbm"]])
    weights = weights / np.sum(weights)
    train_proba = sum(w * p for w, p in zip(weights, p_train))

    # Find best threshold on train
    res = find_best_thresholds(y, train_proba)
    threshold = res.get("f1_threshold", 0.5)

    # Blend test probabilities and apply threshold
    final_proba = sum(w * p for w, p in zip(weights, p_test))
    pred = (final_proba >= threshold).astype(int)

    # Save predictions
    output_df = pd.DataFrame({"ID": ids, "default_12month": pred})
    output_df.to_csv(PRED_PATH, index=False)

    print(f"\nPredictions saved to: {PRED_PATH}")
    print("✅ Training + prediction complete.")



