# metrics.py
import numpy as np

def per_group_accuracy(y_true, y_pred, groups):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); groups = np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        m = (groups == g)
        out[int(g)] = float((y_true[m] == y_pred[m]).mean()) if m.any() else float("nan")
    return out

def classwise_accuracy(y_true, y_pred):
    """
    Assumes binary labels {0,1}. Returns {0:acc0, 1:acc1}.
    Uses NaN if a class is missing from y_true.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    out = {}
    for c in [0, 1]:
        m = (y_true == c)
        out[c] = float((y_pred[m] == c).mean()) if m.any() else float("nan")
    return out

def balanced_class_accuracy(y_true, y_pred):
    accs = classwise_accuracy(y_true, y_pred)
    vals = [v for v in accs.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")

def overall_and_gap(y_true, y_pred, groups):
    overall = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    accs = per_group_accuracy(y_true, y_pred, groups)
    vals = [v for v in accs.values() if not np.isnan(v)]
    gap = float(max(vals) - min(vals)) if vals else 0.0
    return overall, accs, gap
