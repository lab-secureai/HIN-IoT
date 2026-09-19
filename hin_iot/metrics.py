"""Classification metrics and result aggregation (no PyTorch dependency)."""

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

METRIC_COLUMNS: List[str] = [
    "accuracy", "balanced_accuracy", "precision", "recall", "specificity",
    "binary_f1", "macro_f1", "mcc", "pr_auc", "roc_auc",
]
COST_COLUMNS: List[str] = ["train_ms_per_epoch", "infer_ms_per_sample", "peak_gpu_mb"]


def _safe(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except ValueError:
        return float("nan")


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Metrics for binary malware detection (malware = positive class 1).

    ``all_malware_f1`` is the binary F1 of a detector that labels every file as
    malware. It is the reference level for binary F1 on imbalanced test sets
    (for example the held-out architecture in LOAO).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n_pos = int((y_true == 1).sum())
    n = int(len(y_true))
    prevalence = n_pos / n if n else float("nan")
    out = {
        "n_samples": n,
        "n_malware": n_pos,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": _safe(balanced_accuracy_score, y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "binary_f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
        "all_malware_f1": (2 * prevalence / (1 + prevalence)) if n else float("nan"),
    }
    if y_prob is not None:
        # Average precision is defined for any test set with at least one malware file;
        # ROC-AUC needs both classes and is NaN otherwise.
        if n_pos > 0:
            out["pr_auc"] = _safe(average_precision_score, y_true, y_prob)
        if 0 < n_pos < n:
            out["roc_auc"] = _safe(roc_auc_score, y_true, y_prob)
    return out


def summarize(df_raw: pd.DataFrame, group_cols: Iterable[str],
              value_cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Mean, sample standard deviation (ddof=1) and a formatted "mean ± std" column per metric."""
    group_cols = list(group_cols)
    if value_cols is None:
        value_cols = [c for c in METRIC_COLUMNS + ["all_malware_f1"] + COST_COLUMNS if c in df_raw.columns]
    rows = []
    for keys, grp in df_raw.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row["n_runs"] = len(grp)
        for col in value_cols:
            vals = pd.to_numeric(grp[col], errors="coerce")
            mean, std = vals.mean(), vals.std(ddof=1)
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[col] = f"{mean:.4f} ± {0.0 if np.isnan(std) else std:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)
