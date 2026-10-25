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

References:
  Kim et al. (AAAI 2022): S. Kim, K. Choi, H.-S. Choi, B. Lee, and S. Yoon,
                          "Towards a rigorous evaluation of time-series anomaly
                          detection," in Proc. AAAI Conf. Artif. Intell., vol.
                          36, no. 7, 2022, pp. 7194-7201, doi:
                          10.1609/aaai.v36i7.20680.
  Heydari & Nyarko (2026): V. Heydari and K. Nyarko, "Alarm-budgeted event-
                           level evaluation of ICS anomaly detection: Lessons
                           from SWaT and WADI," IEEE Access, vol. 14, pp.
                           17585-17609, 2026, doi: 10.1109/ACCESS.2026.3659034.

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


# --------------------------------------------------------------------------
# A1 -> A2 primary protocol: alarm events, budgets, event-level metrics
# --------------------------------------------------------------------------
# Operators act on alarm EVENTS, not on single flagged seconds. Every model's
# per-second alarms go through the same `eventize` with the same parameters
# at calibration (A1) and at test (A2); thresholds are picked on A1 only.
# Time is measured in rows: both streams are 1 Hz (A2 has a single 82 s
# timestamp gap, recorded in the data report).

def eventize(
    alarm: np.ndarray,
    dilate: int = 0,
    merge_gap: int = 0,
    min_duration: int = 1,
    cooldown: int = 0,
) -> list[tuple[int, int]]:
    """
    Per-second boolean alarms -> [start, end) alarm events:
      1. dilate every alarmed second by `dilate` seconds on both sides,
      2. merge runs separated by <= `merge_gap` seconds,
      3. drop runs shorter than `min_duration` seconds,
      4. cooldown: an event starting <= `cooldown` s after the previous
         event's end is folded into it rather than raised as a new alarm.
    "operator" profile: 2 / 5 / 2 / 10. "min30" profile: 0 / 0 / 30 / 0.
    """
    a = np.asarray(alarm, dtype=bool).ravel()
    n = a.size
    if dilate > 0 and a.any():
        idx = np.flatnonzero(a)
        diff = np.zeros(n + 1, dtype=int)
        np.add.at(diff, np.clip(idx - dilate, 0, n), 1)
        np.add.at(diff, np.clip(idx + dilate + 1, 0, n), -1)
        a = np.cumsum(diff[:n]) > 0
    runs = find_segments(a.astype(int))

    def _merge(runs, gap):
        out = []
        for s, e in runs:
            if out and s - out[-1][1] <= gap:
                out[-1] = (out[-1][0], e)
            else:
                out.append((s, e))
        return out

    runs = _merge(runs, merge_gap)
    runs = [(s, e) for s, e in runs if e - s >= min_duration]
    return _merge(runs, cooldown) if cooldown > 0 else runs


def events_to_mask(events: list[tuple[int, int]], n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in events:
        m[s:e] = True
    return m


def alarm_load(scores: np.ndarray, threshold: float, event_params: dict) -> dict:
    """Achieved alarm workload of a nominal stream at `threshold`."""
    ev = eventize(scores >= threshold, **event_params)
    hours = len(scores) / 3600.0
    duty = sum(e - s for s, e in ev) / max(len(scores), 1)
    return {"events": len(ev), "events_per_hour": len(ev) / hours, "duty": duty}


def calibrate_budget(
    cal_scores: np.ndarray,
    events_per_hour: float,
    duty_cap: float,
    event_params: dict,
) -> tuple[float, dict]:
    """
    Alarm-budget calibration (after Heydari & Nyarko 2026) on the nominal
    calibration stream: the
    lowest threshold whose eventized alarms stay within `events_per_hour`
    and `duty_cap`, searched over a quantile grid of the calibration scores.

    Alarm counts are not perfectly monotone in the threshold, so the search
    walks DOWN from above the maximum and stops at the first violation: the
    chosen threshold and every grid threshold above it meet the budget,
    which rules out a lucky isolated low threshold.
    """
    s = np.asarray(cal_scores, dtype=float)
    levels = np.unique(np.concatenate([np.linspace(0.5, 0.99, 50),
                                       1 - np.logspace(-2, -6.5, 160)]))
    grid = np.unique(np.quantile(s, levels))[::-1]
    top = float(s.max()) + 1e-9 * max(1.0, abs(float(s.max())))
    chosen, load = top, alarm_load(s, top, event_params)
    for t in grid:
        l = alarm_load(s, float(t), event_params)
        if l["events_per_hour"] > events_per_hour or l["duty"] > duty_cap:
            break
        chosen, load = float(t), l
    return chosen, load


def attack_event_metrics(
    y_true: np.ndarray, events: list[tuple[int, int]], n_boot: int = 2000, seed: int = 0
) -> dict:
    """
    Event-level outcome on the attacked stream, one row per annotated attack
    interval (contiguous run of label 1):
      detected       any alarm event overlaps the interval
      delay_s        first alarmed second inside the interval minus its start
      coverage       fraction of the interval's seconds inside alarm events
    plus alarm events that overlap no attack interval (false alarm events)
    and their rate per hour of attack-free A2 time. The miss-rate confidence
    interval resamples attack INTERVALS (not rows) with replacement.
    """
    y = np.asarray(y_true).astype(int)
    intervals = find_segments(y)
    mask = events_to_mask(events, len(y))
    per = []
    for k, (s, e) in enumerate(intervals):
        hit = np.flatnonzero(mask[s:e])
        per.append({"interval": k, "start": s, "length_s": e - s,
                    "detected": bool(hit.size), "delay_s": int(hit[0]) if hit.size else None,
                    "coverage": float(mask[s:e].mean())})
    false_ev = [ev for ev in events if not y[ev[0]:ev[1]].any()]
    normal_hours = (y == 0).sum() / 3600.0
    det = np.array([p["detected"] for p in per], dtype=float)
    rng = np.random.default_rng(seed)
    boot = det[rng.integers(0, len(det), size=(n_boot, len(det)))].mean(axis=1) if len(det) else np.array([np.nan])
    delays = [p["delay_s"] for p in per if p["detected"]]
    return {
        "n_intervals": len(intervals),
        "detected": int(det.sum()),
        "miss_rate": float(1 - det.mean()) if len(det) else float("nan"),
        "miss_rate_ci95": [float(1 - np.quantile(boot, 0.975)), float(1 - np.quantile(boot, 0.025))],
        "median_delay_s": float(np.median(delays)) if delays else None,
        "mean_coverage": float(np.mean([p["coverage"] for p in per])) if per else float("nan"),
        "alarm_events": len(events),
        "false_alarm_events": len(false_ev),
        "false_alarm_events_per_normal_hour": len(false_ev) / normal_hours if normal_hours else float("nan"),
        "normal_duty": float(mask[y == 0].mean()) if (y == 0).any() else float("nan"),
        "per_interval": per,
    }


def udr_far_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Undetected rate (FN / (FN + TP)) and false alarm rate
    (FP / (FP + TN)), pointwise."""
    y, p = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tp, fn = int(((p == 1) & (y == 1)).sum()), int(((p == 0) & (y == 1)).sum())
    fp, tn = int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 0)).sum())
    return {"udr": fn / max(fn + tp, 1), "far": fp / max(fp + tn, 1)}
