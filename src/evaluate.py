"""
Evaluation for ICS anomaly detection.

Central design decision of this project: every metric is reported under TWO
protocols, side by side.

  pointwise    - each timestep judged independently. Strict and honest.
  point_adjust - if ANY timestep inside a contiguous ground-truth anomaly
                 segment is flagged, the ENTIRE segment counts as detected.

Point-adjust is the protocol used by most published ICS / time-series anomaly
detection work, which is why headline F1 scores above 0.9 are common in the
literature. Kim et al. (AAAI 2022) showed that under point-adjust a RANDOM
anomaly score can beat published state-of-the-art methods. The gap between the
two protocols is therefore a result in its own right, not a footnote.

Usage:
    from evaluate import evaluate_all
    res = evaluate_all(y_true, scores, threshold=t)
    print(res.to_frame())
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


# --------------------------------------------------------------------------
# segment utilities
# --------------------------------------------------------------------------

def find_segments(y: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index pairs for each contiguous run of 1s in y."""
    y = np.asarray(y).astype(int).ravel()
    if y.size == 0:
        return []
    # pad with zeros so runs at the boundaries are detected
    padded = np.concatenate(([0], y, [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Apply the point-adjust protocol to a binary prediction vector.

    For every contiguous ground-truth anomaly segment, if the model flagged at
    least one timestep inside it, all timesteps in that segment become 1.
    Predictions outside anomaly segments (false positives) are untouched.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    adjusted = y_pred.copy()
    for start, end in find_segments(y_true):
        if y_pred[start:end].any():
            adjusted[start:end] = 1
    return adjusted


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    mcc: float
    n_pred_pos: int

    def as_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mcc": self.mcc,
            "n_pred_pos": self.n_pred_pos,
        }


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """Threshold-dependent metrics. Accuracy is deliberately not reported:
    with anomaly rates around 1 percent it is meaningless."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    # MCC is undefined (0) when a whole row or column of the confusion matrix
    # is empty; sklearn already returns 0.0 there.
    mcc = matthews_corrcoef(y_true, y_pred) if y_pred.any() or y_true.any() else 0.0
    return Metrics(float(p), float(r), float(f), float(mcc), int(y_pred.sum()))


@dataclass
class Result:
    """Full evaluation of one model on one dataset."""

    pointwise: Metrics
    point_adjusted: Metrics
    roc_auc: float
    pr_auc: float
    threshold: float
    anomaly_rate: float
    meta: dict = field(default_factory=dict)

    @property
    def f1_inflation(self) -> float:
        """How much point-adjust flatters this model. The headline number."""
        return self.point_adjusted.f1 - self.pointwise.f1

    def to_frame(self):
        import pandas as pd

        rows = {
            "pointwise": self.pointwise.as_dict(),
            "point_adjusted": self.point_adjusted.as_dict(),
        }
        df = pd.DataFrame(rows).T
        df["roc_auc"] = self.roc_auc
        df["pr_auc"] = self.pr_auc
        return df

    def summary(self) -> str:
        return (
            f"{self.meta.get('model', 'model')} on {self.meta.get('dataset', 'data')}\n"
            f"  anomaly rate     {self.anomaly_rate:.4f}   threshold {self.threshold:.4g}\n"
            f"  ROC-AUC          {self.roc_auc:.4f}    PR-AUC {self.pr_auc:.4f}\n"
            f"  pointwise      P {self.pointwise.precision:.3f}  "
            f"R {self.pointwise.recall:.3f}  F1 {self.pointwise.f1:.3f}  "
            f"MCC {self.pointwise.mcc:.3f}\n"
            f"  point-adjusted P {self.point_adjusted.precision:.3f}  "
            f"R {self.point_adjusted.recall:.3f}  F1 {self.point_adjusted.f1:.3f}  "
            f"MCC {self.point_adjusted.mcc:.3f}\n"
            f"  F1 inflation     {self.f1_inflation:+.3f}"
        )


def evaluate_all(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
    percentile: float = 99.0,
    **meta,
) -> Result:
    """
    Evaluate continuous anomaly `scores` against binary `y_true`.

    Higher score must mean more anomalous. If `threshold` is None it is set at
    the given percentile of the scores. Threshold-free metrics (ROC-AUC,
    PR-AUC) are reported alongside so results do not hinge on that choice.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if y_true.shape != scores.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {scores.shape}")

    if threshold is None:
        threshold = float(np.percentile(scores, percentile))

    y_pred = (scores >= threshold).astype(int)
    y_pred_adj = point_adjust(y_true, y_pred)

    both_classes = 0 < y_true.sum() < y_true.size
    roc = float(roc_auc_score(y_true, scores)) if both_classes else float("nan")
    prauc = float(average_precision_score(y_true, scores)) if both_classes else float("nan")

    return Result(
        pointwise=binary_metrics(y_true, y_pred),
        point_adjusted=binary_metrics(y_true, y_pred_adj),
        roc_auc=roc,
        pr_auc=prauc,
        threshold=threshold,
        anomaly_rate=float(y_true.mean()),
        meta=meta,
    )


# --------------------------------------------------------------------------
# sanity check: reproduce the Kim et al. random-score result
# --------------------------------------------------------------------------

def random_baseline_check(
    n: int = 100_000,
    n_segments: int = 40,
    seg_len: int = 250,
    percentile: float = 99.0,
    seed: int = 0,
) -> Result:
    """
    Score pure noise against synthetic anomaly segments.

    If point-adjusted F1 comes back high while pointwise F1 is near zero, the
    point_adjust implementation is CORRECT and you have reproduced the core
    finding of Kim et al. (AAAI 2022) yourself. A low point-adjusted F1 here
    means the implementation is broken.
    """
    rng = np.random.default_rng(seed)

    y = np.zeros(n, dtype=int)
    for start in rng.choice(n - seg_len, size=n_segments, replace=False):
        y[start : start + seg_len] = 1

    scores = rng.random(n)  # pure noise, contains no information about y
    return evaluate_all(y, scores, percentile=percentile,
                        model="random noise", dataset="synthetic")


if __name__ == "__main__":
    res = random_baseline_check()
    print(res.summary())
    print()
    if res.point_adjusted.f1 > 0.5 and res.pointwise.f1 < 0.1:
        print("PASS: point_adjust implemented correctly.")
        print("Random noise scores near-perfectly under the protocol used by")
        print("most of the published ICS anomaly detection literature.")
    else:
        print("CHECK: unexpected result, inspect point_adjust().")
