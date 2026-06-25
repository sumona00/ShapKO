import numpy as np
import math

def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = float(x.mean())
    s = float(x.std())
    return (x - m) / (s + eps)

def safe_auc(y_true, y_score) -> float:
    y_true = list(map(int, y_true))
    y_score = list(map(float, y_score))
    if len(y_true) == 0 or len(set(y_true)) < 2:
        return math.nan
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(np.asarray(y_true), np.asarray(y_score)))
    except Exception:
        return math.nan

def safe_ap(y_true, y_score) -> float:
    y_true = list(map(int, y_true))
    y_score = list(map(float, y_score))
    if len(y_true) == 0 or len(set(y_true)) < 2:
        return math.nan
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(np.asarray(y_true), np.asarray(y_score)))
    except Exception:
        return math.nan
