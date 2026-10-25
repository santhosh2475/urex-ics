"""
Feature selection and data-quality checks, computed on TRAINING rows only.

    from features import pearson_select, drift_report, subset

    keep, table = pearson_select(split)          # indices + the full ranking
    split_fs = subset(split, keep)               # same split, fewer columns

Pearson correlation between a feature and the 0/1 label is the
point-biserial correlation: how far the feature's mean moves between normal
and attack rows, in units of its spread. `pearson_select` uses it twice:

  relevance   |r(feature, label)| -- rank features by how much they move
              when an attack is on.
  redundancy  |r(feature, feature)| -- of any pair above `max_redundancy`,
              keep only the more relevant one. SWaT has 15 such pairs in
              training (P101/FIT201/MV201/P203 are all one pump-and-valve
              state, UV401/P501/FIT504 one more).

Redundancy matters most for Naive Bayes: it multiplies one likelihood per
feature as if they were independent, so four copies of the same signal are
counted as four independent pieces of evidence.

Everything is computed on split.X_train / split.y_train. Ranking features
on the full dataset would leak: on SWaT the full-data |r| of FIT504 is 0.77
but only 0.33 on the chronological training window, because the full-data
number is dominated by one 10-hour attack that sits in validation.

Pearson only sees linear, mean-shift relationships. An attack that changes
a sensor's variance or its timing but not its mean scores |r| ~ 0 and will
be dropped. That is the right trade for Naive Bayes and logistic
regression (both are mean-shift models themselves); it is the wrong filter
for the sequence models, so apply it per model family, not globally.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from data import Split


def _corr_with_label(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|Pearson r| of each column of X with binary y; 0 for constant columns."""
    y = y.astype(float)
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    denom = np.sqrt((Xc**2).sum(axis=0) * (yc**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Xc * yc[:, None]).sum(axis=0) / denom
    return np.nan_to_num(np.abs(r), nan=0.0)


def label_relevance(
    X: np.ndarray, y: np.ndarray, how: str = "pooled", context: int = 1800
) -> np.ndarray:
    """
    |Pearson r(feature, label)| per column.

    how="pooled"   one correlation over all rows. Dominated by the longest
                   attacks: on the stratified SWaT train set the single
                   10-hour stage-5 attack is 73% of attack rows, so every
                   stage-4/5 flow sensor scores ~0.83 and everything else
                   looks irrelevant by comparison.
    how="episode"  one correlation per attack episode, computed over that
                   episode plus `context` normal rows either side, then
                   averaged across episodes. Every attack gets one vote
                   regardless of its length, so a sensor that only matters
                   for short stage-1 attacks still ranks.
    """
    if how == "pooled":
        return _corr_with_label(X, y)
    if how != "episode":
        raise ValueError(f"unknown relevance {how!r}")
    from data import contiguous_runs

    per = [
        _corr_with_label(X[max(0, s - context) : e + context], y[max(0, s - context) : e + context])
        for s, e in contiguous_runs(y.astype(bool))
    ]
    return np.mean(per, axis=0)


def pearson_select(
    split: Split,
    max_redundancy: float = 0.95,
    min_relevance: float = 0.0,
    top_k: int | None = None,
    exclude: tuple[str, ...] = (),
    relevance: str = "pooled",
) -> tuple[list[int], pd.DataFrame]:
    """
    Greedy relevance-then-redundancy filter (a Pearson-only mRMR).

    1. Drop features constant on the training rows (no information, and
       their test-time variation is pure distribution shift).
    2. Drop anything named in `exclude` (e.g. sensors `drift_report` flags).
    3. Rank the rest by |r(feature, label)|, highest first, with
       `relevance` = "pooled" or "episode" (see `label_relevance`).
    4. Walk the ranking; keep a feature only if its |r| with every feature
       already kept is <= max_redundancy, and its relevance >= min_relevance.
    5. Stop after `top_k` kept features, if given.

    Returns (kept column indices in ranking order, a table explaining every
    feature's fate) so the decision is inspectable, not just applied.
    """
    X, y = split.X_train, split.y_train
    names = np.array(split.feature_names)
    std = X.std(axis=0)
    relevance = label_relevance(X, y, how=relevance)

    order = np.argsort(-relevance)
    live = std > 0
    C = np.abs(np.corrcoef(X[:, live], rowvar=False))
    live_pos = {j: k for k, j in enumerate(np.flatnonzero(live))}

    kept: list[int] = []
    rows = []
    for j in order:
        status, partner = "kept", ""
        if not live[j]:
            status = "constant in train"
        elif names[j] in exclude:
            status = "excluded"
        elif relevance[j] < min_relevance:
            status = "below min_relevance"
        elif top_k is not None and len(kept) >= top_k:
            status = "beyond top_k"
        else:
            for k in kept:
                if C[live_pos[j], live_pos[k]] > max_redundancy:
                    status = "redundant"
                    partner = f"{names[k]} (|r|={C[live_pos[j], live_pos[k]]:.3f})"
                    break
        if status == "kept":
            kept.append(int(j))
        rows.append({"feature": names[j], "abs_r_label": relevance[j],
                     "status": status, "redundant_with": partner})
    return kept, pd.DataFrame(rows)


def drift_report(split: Split, z_threshold: float = 1.0) -> pd.DataFrame:
    """
    How far each sensor's NORMAL-row mean moves from training to validation,
    in training standard deviations. Uses validation, never test.

    A sensor whose normal behaviour itself drifts by several sigma between
    periods (SWaT's chemical analysers: AIT201 shifts -7 sigma by the
    validation window and -18 sigma by the test window) makes every
    unsupervised detector flag the whole later period as anomalous. That
    is what produced lstm_ae / lstm_forecast recall 1.000 at precision equal
    to the base rate in the chronological run: a cleaning problem, not a
    modelling one.
    """
    tr = split.X_train[split.y_train == 0]
    va = split.X_val[split.y_val == 0]
    sd = tr.std(axis=0)
    constant = sd < 1e-9
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(constant, np.nan, (va.mean(axis=0) - tr.mean(axis=0)) / sd)
        ratio = np.where(constant, np.nan, va.std(axis=0) / sd)
    df = pd.DataFrame({
        "feature": split.feature_names,
        "val_normal_shift_z": z,
        "val_std_ratio": ratio,
        # constant on normal training rows: any movement later is new
        # behaviour the training data never showed, not measurable drift
        "constant_in_train_normal": constant,
    })
    df["drifting"] = df["val_normal_shift_z"].abs() > z_threshold
    return df.sort_values("val_normal_shift_z", key=np.abs, ascending=False,
                          na_position="last").reset_index(drop=True)


def subset(split: Split, keep: list[int]) -> Split:
    """The same split with only the `keep` columns, everything else intact."""
    keep = list(keep)
    out = replace(
        split,
        X_train=split.X_train[:, keep],
        X_val=split.X_val[:, keep],
        X_test=split.X_test[:, keep],
        feature_names=[split.feature_names[j] for j in keep],
        name=f"{split.name}-fs{len(keep)}",
    )
    # carry over reporting attributes set outside the dataclass fields
    # (incidents, row_index, fold) -- dataclasses.replace drops them
    for attr in ("incidents", "row_index", "fold"):
        if hasattr(split, attr):
            setattr(out, attr, getattr(split, attr))
    return out
