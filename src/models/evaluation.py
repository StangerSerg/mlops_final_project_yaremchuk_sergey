import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model: CatBoostClassifier, X: pd.DataFrame, y: pd.Series
) -> dict[str, float]:
    """
    Вычисляет RMSE для оценки качества модели.

    Args:
        model: Обученная модель.
        X: Матрица признаков.
        y: Целевая переменная.

    Returns:
        Словарь с метриками. Ключи названия метрик:
        accuracy, precision, recall, f1, roc_auc, pr_auc, logloss, brier,
        значения - float.
    """
    threshold = 0.5
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics = {
        "accuracy":  accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall":    recall_score(y, preds, zero_division=0),
        "f1":        f1_score(y, preds, zero_division=0),
        "roc_auc":   roc_auc_score(y, proba),
        "pr_auc":    average_precision_score(y, proba),
        "logloss":   log_loss(y, proba),
        "brier":     brier_score_loss(y, proba),
    }

    return metrics