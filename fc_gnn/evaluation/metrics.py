"""Standard ML evaluation metrics for cybersecurity anomaly detection."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, average_precision_score,
    matthews_corrcoef, confusion_matrix
)
from typing import Dict


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                    y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Compute accuracy, macro-F1, AUPRC, FAR, MCC.

    Args:
        y_true: [N] ground truth labels
        y_pred: [N] predicted labels
        y_prob: [N, C] predicted probabilities (optional, for AUPRC)

    Returns:
        dict of metric name → value
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # False Alarm Rate: FP / (FP + TN) using one-vs-rest for benign class (class 0)
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    # Binary FAR: count correctly predicted attacks vs benign misclassifications
    benign_mask = (y_true == 0)
    if benign_mask.sum() > 0:
        far = (y_pred[benign_mask] != 0).mean()
    else:
        far = 0.0

    # AUPRC (macro-averaged)
    if y_prob is not None:
        from sklearn.preprocessing import label_binarize
        classes = np.unique(y_true)
        if len(classes) > 2:
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            try:
                auprc = average_precision_score(y_bin, y_prob, average="macro")
            except Exception:
                auprc = 0.0
        else:
            try:
                auprc = average_precision_score(y_true, y_prob[:, 1])
            except Exception:
                auprc = 0.0
    else:
        auprc = 0.0

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "auprc": float(auprc),
        "far": float(far),
        "mcc": float(mcc),
    }
