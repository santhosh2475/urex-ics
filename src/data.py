"""
Data loading and splitting.

Everything in this module is dataset-independent and already correct. The
per-dataset loaders at the bottom are stubs: fill each one in once you have
the actual files, and the rest of the pipeline works unchanged.

Two rules this module enforces, because breaking either invalidates every
downstream result:

  1. Splits are CHRONOLOGICAL. No shuffling, no random_state. A random split
     on time-series data leaks future information into training and produces
     results that cannot be reproduced in deployment.

  2. Scalers are fit on TRAINING DATA ONLY. Fitting on the full series leaks
     the test distribution's mean and variance into training.

Both are asserted, not merely documented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import DATA_RAW, TRAIN_FRAC, VAL_FRAC


# --------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------

@dataclass
class Split:
    """A chronologically split, scaled dataset ready for modelling."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    name: str
    # [start, end) bounds of each contiguous stretch of real time inside
    # each part. A chronological split has one block per part; the
    # stratified split concatenates several non-adjacent blocks, and every
    # windowed model must window within a block, never across the join.
    # None means one block spanning the whole part.
    blocks: dict[str, list[tuple[int, int]]] | None = field(default=None)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def bounds(self, part: str) -> list[tuple[int, int]]:
        """Block bounds within `part` ('train', 'val' or 'test')."""
        if self.blocks and part in self.blocks:
            return self.blocks[part]
        return [(0, len(getattr(self, f"y_{part}")))]

    def continuity_mask(self, part: str) -> np.ndarray:
        """True everywhere except the first row of every block after the
        first, so `contiguous_runs` of (mask & this) never spans a join."""
        mask = np.ones(len(getattr(self, f"y_{part}")), dtype=bool)
        for start, _ in self.bounds(part)[1:]:
            mask[start] = False
        return mask

    def normal_only(self) -> np.ndarray:
        """Training rows labelled normal, as one array.

        Unsupervised detectors use the normal training rows. Supervised
        detectors (boosting) use X_train with y_train. Both draw from the
        SAME training window, so the comparison stays fair.

        WARNING for sequence models: this concatenates non-adjacent rows.
        Every removed attack segment becomes a splice where the timeline
        jumps. Windows built across a splice contain a discontinuity that
        never occurs in deployment, and a model trained to reconstruct them
        is being taught an artefact. Sequence models should use
        `normal_mask()` with `make_windows_segmented` instead.
        """
        return self.X_train[self.y_train == 0]

    def normal_mask(self) -> np.ndarray:
        """Boolean mask of normal training rows, positions preserved."""
        return self.y_train == 0

    def summary(self) -> str:
        def rate(y):
            return y.mean() if y.size else float("nan")

        return (
            f"{self.name}: {self.n_features} features\n"
            f"  train {self.X_train.shape[0]:>8,} rows  anomaly rate {rate(self.y_train):.4f}\n"
            f"  val   {self.X_val.shape[0]:>8,} rows  anomaly rate {rate(self.y_val):.4f}\n"
            f"  test  {self.X_test.shape[0]:>8,} rows  anomaly rate {rate(self.y_test):.4f}"
        )


# --------------------------------------------------------------------------
# splitting and scaling
# --------------------------------------------------------------------------

def _episode_safe_cut(y: np.ndarray, target: int, after: int = 0) -> int:
    """
    Nudge a target split index forward so it never lands inside a
    contiguous attack episode.

    A plain `int(n * frac)` cut doesn't know where the attacks are, so it
    can bisect one -- half its rows end up in the earlier split, half in
    the later one, as an accident of arithmetic rather than a decision.
    This snaps such a target FORWARD to that episode's end, never
    backward: the earlier split (chronologically first) gets the whole
    episode, rather than leaving a truncated, label-inconsistent fragment
    dangling at the start of the later one. Consistently snapping the same
    direction is also what makes the result deterministic and easy to
    reason about, rather than picking whichever side is numerically
    closer.

    One concrete consequence worth knowing about: an unusually long
    episode sitting near a target cut gets pulled wholly into the split
    that already contains it rather than split, which can push that split
    noticeably larger than its nominal fraction. That's intentional --
    it's strictly better than the alternative of bisecting a real attack
    and reporting an inconsistent label for it on both sides.

    `after` stops the result from landing before an earlier cut (needed
    when computing the val/test boundary once train/val is already fixed).
    """
    target = max(target, after)
    for start, end in contiguous_runs(y.astype(bool)):
        if start <= target < end:
            return max(end, after)
    return target


def chronological_split(
    df: pd.DataFrame,
    label_col: str = "label",
    name: str = "unnamed",
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    scale: bool = True,
) -> Split:
    """
    Split a time-ordered dataframe into train/val/test by position --
    nudged, when necessary, so neither boundary bisects an attack episode
    (see `_episode_safe_cut`). train_frac/val_frac are therefore TARGETS,
    not guarantees: an episode that straddles a target cut shifts that
    boundary forward to the episode's end, so the realised split sizes can
    differ from the nominal fractions. The alternative -- cutting through
    an episode and splitting its rows across two sets with no regard for
    where the attack actually was -- is worse, and was previously silent:
    nothing recorded whether a boundary happened to land mid-attack.

    `df` MUST already be sorted by time. Feature columns are everything except
    `label_col`; drop timestamps and other non-features before calling.
    """
    if label_col not in df.columns:
        raise KeyError(f"label column {label_col!r} not found in {list(df.columns)[:10]}")

    y = df[label_col].to_numpy().astype(int)
    features = df.drop(columns=[label_col])
    feature_names = list(features.columns)
    X = features.to_numpy(dtype=float)

    n = len(df)
    i_train = _episode_safe_cut(y, int(n * train_frac))
    i_val = _episode_safe_cut(y, int(n * (train_frac + val_frac)), after=i_train)

    X_train, X_val, X_test = X[:i_train], X[i_train:i_val], X[i_val:]
    y_train, y_val, y_test = y[:i_train], y[i_train:i_val], y[i_val:]

    if scale:
        scaler = StandardScaler().fit(X_train)  # train only, deliberately
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    split = Split(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names, name
    )
    _assert_split_valid(split, n)
    return split


# --------------------------------------------------------------------------
# episode-stratified split
# --------------------------------------------------------------------------
# The chronological split above is the deployment-realistic protocol, but on
# SWaT it hands each part a very different set of attacks: one 35,900-second
# episode (about 10 hours, stage 5) lands in validation alone and makes up
# 78% of all attack rows, train sees 9 short attacks and no stage-4 attack at
# all, and the three parts end up at 0.93% / 22.95% / 3.76% attack rate.
# Supervised models are then evaluated on attack types they never saw, and
# no threshold chosen on validation transfers to test.
#
# `stratified_episode_split` is the attack-coverage protocol. Ordinary
# stratified k-fold is NOT an option on time series -- shuffling single rows
# puts second t in train and second t+1 in test, which is a leak. Instead:
#
#   1. Attack episodes closer together than `merge_gap` seconds are grouped
#      into one incident, so related attacks never straddle two parts.
#   2. The timeline is cut at the midpoint of every gap between incidents,
#      giving blocks of [normal lead-in, incident, normal tail-off].
#   3. The attack-free prefix (SWaT's whole normal week) always goes to
#      train, so the unsupervised models keep their full normal history.
#   4. Incident blocks are assigned to train/val/test stratified by the
#      stage they target and balanced on attack-row counts.
#   5. `purge` rows either side of every boundary between two different
#      parts are dropped, so no window, lag or rolling feature built inside
#      one part can reach into a neighbouring block of another part.
#
# Rows keep their time order within each part, the scaler is still fit on
# train only, and blocks are recorded on the Split so windowed models never
# window across a join. Report this protocol ALONGSIDE the chronological
# one, never instead of it: chronological answers "how would this behave if
# deployed on day 1", stratified answers "how well does it detect each kind
# of attack when it has seen examples of that kind".

def attack_incidents(
    y: np.ndarray,
    X: pd.DataFrame | None = None,
    merge_gap: int = 1200,
    context: int = 1800,
) -> pd.DataFrame:
    """
    One row per attack incident (episodes closer than `merge_gap` merged).

    If `X` (unscaled feature frame aligned with `y`) is given, each incident
    is also tagged with the process stage it most likely targets: the stage
    digit of the sensor whose mean moved the most, in standard deviations of
    the `context` rows of normal operation just before it. iTrust's own
    attack list would be the authoritative source; this is the data-driven
    stand-in, and for SWaT it recovers the stage of the documented attack
    points in most cases.
    """
    episodes = contiguous_runs(np.asarray(y).astype(bool))
    if not episodes:
        raise ValueError("no attack rows in y")

    groups = [[episodes[0]]]
    for s, e in episodes[1:]:
        if s - groups[-1][-1][1] < merge_gap:
            groups[-1].append((s, e))
        else:
            groups.append([(s, e)])

    sd = None
    if X is not None:
        first = episodes[0][0]
        sd = X.iloc[:first].std().replace(0, np.nan)

    rows = []
    for k, g in enumerate(groups):
        start, end = g[0][0], g[-1][1]
        row = {
            "incident": k,
            "start": start,
            "end": end,
            "n_episodes": len(g),
            "attack_rows": int(sum(e - s for s, e in g)),
        }
        if X is not None:
            # stage of the longest episode in the incident
            s, e = max(g, key=lambda se: se[1] - se[0])
            pre = X.iloc[max(0, s - context) : s]
            shift = ((X.iloc[s:e].mean() - pre.mean()).abs() / sd).fillna(0)
            top = shift.idxmax()
            digit = next((ch for ch in top if ch.isdigit()), "?")
            row["stage"] = f"P{digit}"
            row["top_sensor"] = top
        rows.append(row)
    return pd.DataFrame(rows)


def _assign_incidents(
    incidents: pd.DataFrame, fracs: tuple[float, float, float]
) -> np.ndarray:
    """
    Greedy stratified assignment of incidents to parts 0/1/2.

    Within each stage (largest stage first), incidents are placed largest
    first into whichever part is furthest below its target share of that
    stage's incidents, ties broken by the part furthest below its target
    share of all attack rows, then by train > test > val. A stage with only
    two incidents therefore lands in train (so supervised models see it)
    and test (so it is evaluated) rather than wasting one on validation.
    Deterministic; no random state.
    """
    fracs = np.asarray(fracs, dtype=float)
    total_rows = incidents["attack_rows"].sum()
    rows_assigned = np.zeros(3)
    part = np.full(len(incidents), -1)

    stages = incidents.groupby("stage")["attack_rows"].sum().sort_values(ascending=False)
    for stage in stages.index:
        idx = incidents.index[incidents["stage"] == stage]
        idx = incidents.loc[idx].sort_values("attack_rows", ascending=False).index
        count = np.zeros(3)
        for i in idx:
            count_deficit = fracs * len(idx) - count
            row_deficit = (fracs * total_rows - rows_assigned) / max(total_rows, 1)
            priority = np.array([0, 2, 1])  # train, then test, then val
            p = int(np.lexsort((priority, -np.round(row_deficit, 6), -np.round(count_deficit, 6)))[0])
            part[i] = p
            count[p] += 1
            rows_assigned[p] += incidents.loc[i, "attack_rows"]
    return part


def stratified_episode_split(
    df: pd.DataFrame,
    label_col: str = "label",
    name: str = "unnamed",
    fracs: tuple[float, float, float] = (0.5, 0.25, 0.25),
    merge_gap: int = 1200,
    purge: int = 300,
    scale: bool = True,
    verbose: bool = True,
) -> Split:
    """
    Attack-coverage split: whole incidents stratified by targeted stage,
    with purged boundaries. See the section comment above for why and how.

    `fracs` are target shares of incidents (per stage) and of attack rows,
    not of total rows -- train also receives the whole attack-free prefix,
    so it is still by far the largest part in rows. `purge` must exceed WINDOW_SIZE and the longest lag,
    and must be less than half of `merge_gap` so it only ever drops normal
    rows; both are checked.
    """
    if not 0 < 2 * purge < merge_gap:
        raise ValueError(f"need 0 < 2*purge < merge_gap, got purge={purge}, merge_gap={merge_gap}")

    y = df[label_col].to_numpy().astype(int)
    inc = attack_incidents(y, df.drop(columns=[label_col]), merge_gap=merge_gap)
    inc["part"] = _assign_incidents(inc, fracs)
    return _split_from_assignment(df, inc, label_col, name, merge_gap, purge, scale, verbose)


def _split_from_assignment(
    df: pd.DataFrame,
    inc: pd.DataFrame,
    label_col: str,
    name: str,
    merge_gap: int,
    purge: int,
    scale: bool,
    verbose: bool,
) -> Split:
    """Build a purged, block-aware Split from incidents already tagged with
    `part` (0 train, 1 val, 2 test). Shared by the single stratified split
    and every fold of `episode_folds`."""
    y = df[label_col].to_numpy().astype(int)
    features = df.drop(columns=[label_col])
    feature_names = list(features.columns)
    X = features.to_numpy(dtype=float)
    n = len(df)

    # block boundaries at the midpoint of each gap between incidents
    cuts = [0]
    for k in range(1, len(inc)):
        cuts.append((inc.loc[k - 1, "end"] + inc.loc[k, "start"]) // 2)
    cuts.append(n)
    row_part = np.empty(n, dtype=int)
    for k in range(len(inc)):
        row_part[cuts[k] : cuts[k + 1]] = inc.loc[k, "part"]

    # the attack-free prefix belongs to train, whatever incident 0 got
    row_part[: max(0, inc.loc[0, "start"] - merge_gap // 2)] = 0

    # purge both sides of every boundary between two different parts
    keep = np.ones(n, dtype=bool)
    for b in np.flatnonzero(np.diff(row_part)) + 1:
        lo, hi = max(0, b - purge), min(n, b + purge)
        assert not y[lo:hi].any(), "purge would drop attack rows; raise merge_gap"
        keep[lo:hi] = False

    parts, blocks = {}, {}
    for p, part_name in enumerate(("train", "val", "test")):
        mask = keep & (row_part == p)
        idx = np.flatnonzero(mask)
        parts[part_name] = idx
        # contiguous stretches of real time inside this part, re-based
        local = []
        for s, e in contiguous_runs(mask):
            offset = np.searchsorted(idx, s)
            local.append((int(offset), int(offset + (e - s))))
        blocks[part_name] = local

    X_train, X_val, X_test = (X[parts[k]] for k in ("train", "val", "test"))
    y_train, y_val, y_test = (y[parts[k]] for k in ("train", "val", "test"))

    if scale:
        scaler = StandardScaler().fit(X_train)  # train only, deliberately
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    split = Split(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names, name,
        blocks=blocks,
    )
    _assert_split_valid(split, int(keep.sum()))
    split.incidents = inc.copy()  # kept for reporting; not part of the dataclass contract
    split.row_index = parts       # original row of every row in each part, likewise

    if verbose:
        names = np.array(["train", "val", "test"])
        tab = pd.crosstab(names[inc["part"]], inc["stage"])
        print(f"{name}: {len(inc)} incidents, {n - keep.sum():,} rows purged at "
              f"{(np.diff(row_part) != 0).sum()} boundaries")
        print(tab.to_string())
        print()
    return split


# --------------------------------------------------------------------------
# grouped cross-validation over incidents
# --------------------------------------------------------------------------
# One stratified split leaves ~6 incidents in test, and the model ranking
# flips with which 6 they are (naive_bayes: val PR-AUC 0.46, test 0.04).
# `episode_folds` rotates instead: incidents are dealt into `n_folds` groups
# stratified by stage, and fold k uses group k as test, group k+1 as
# validation, and the rest (plus the attack-free prefix) as train. Every
# incident is tested exactly once, so every model is scored on every attack
# and the spread across folds says how much a single-split number can be
# trusted. Same purging, block bounds and train-only scaling as above.

def _assign_folds(incidents: pd.DataFrame, n_folds: int) -> np.ndarray:
    """Deal incidents into folds: within each stage (largest stage first),
    largest incident first, into the fold with the fewest incidents of that
    stage, ties broken by fewest attack rows overall, then lowest index."""
    fold = np.full(len(incidents), -1)
    rows = np.zeros(n_folds)
    stages = incidents.groupby("stage")["attack_rows"].sum().sort_values(ascending=False)
    for stage in stages.index:
        idx = incidents[incidents["stage"] == stage].sort_values("attack_rows", ascending=False).index
        count = np.zeros(n_folds)
        for i in idx:
            f = int(np.lexsort((np.arange(n_folds), rows, count))[0])
            fold[i] = f
            count[f] += 1
            rows[f] += incidents.loc[i, "attack_rows"]
    return fold


def episode_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    label_col: str = "label",
    name: str = "unnamed",
    merge_gap: int = 1200,
    purge: int = 300,
    scale: bool = True,
    verbose: bool = False,
) -> list[Split]:
    """Grouped, stage-stratified, purged K-fold over attack incidents.
    Returns one Split per fold; `split.fold` and `split.incidents` record
    which incidents went where."""
    if not 0 < 2 * purge < merge_gap:
        raise ValueError(f"need 0 < 2*purge < merge_gap, got purge={purge}, merge_gap={merge_gap}")
    if n_folds < 3:
        raise ValueError("need n_folds >= 3: one test fold, one validation fold, the rest train")

    y = df[label_col].to_numpy().astype(int)
    inc = attack_incidents(y, df.drop(columns=[label_col]), merge_gap=merge_gap)
    inc["fold"] = _assign_folds(inc, n_folds)

    splits = []
    for k in range(n_folds):
        inc_k = inc.copy()
        inc_k["part"] = np.select(
            [inc_k["fold"] == k, inc_k["fold"] == (k + 1) % n_folds], [2, 1], 0
        )
        sp = _split_from_assignment(
            df, inc_k, label_col, f"{name}-fold{k}", merge_gap, purge, scale, verbose
        )
        sp.fold = k
        splits.append(sp)
    return splits


def _assert_split_valid(split: Split, n_original: int) -> None:
    """Guardrails. If any of these fire, the results would be meaningless."""
    total = len(split.y_train) + len(split.y_val) + len(split.y_test)
    assert total == n_original, f"rows lost in split: {total} != {n_original}"

    for part, X, y in (
        ("train", split.X_train, split.y_train),
        ("val", split.X_val, split.y_val),
        ("test", split.X_test, split.y_test),
    ):
        assert len(X) == len(y), f"{part}: X/y length mismatch"
        assert len(X) > 0, f"{part} split is empty, check the fractions"

    if split.y_test.sum() == 0:
        raise ValueError(
            f"{split.name}: test split contains no anomalies. The chronological "
            "split has put all attacks in train or val. Check whether the "
            "dataset ships separate normal and attack files that need "
            "interleaving, rather than one continuous run."
        )
    if split.y_train.sum() == 0:
        print(
            f"  NOTE {split.name}: training split has no anomalies. Fine for "
            "unsupervised models, but supervised boosting cannot be trained "
            "on it. See the label-availability experiment."
        )


def make_windows(
    X: np.ndarray, y: np.ndarray | None = None, window: int = 60, stride: int = 1
):
    """
    Convert (n_samples, n_features) into overlapping sequences for LSTM models.

    Returns X_win of shape (n_windows, window, n_features). If `y` is given,
    each window is labelled by its LAST timestep, so window i predicts the
    label at the same index as the pointwise models. Keeping the alignment
    identical is what makes the two model families comparable.
    """
    if len(X) < window:
        raise ValueError(f"need at least {window} rows, got {len(X)}")

    idx = np.arange(0, len(X) - window + 1, stride)
    X_win = np.stack([X[i : i + window] for i in idx])

    if y is None:
        return X_win
    y_win = np.array([y[i + window - 1] for i in idx])
    return X_win, y_win


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index pairs for each contiguous run of True."""
    mask = np.asarray(mask).astype(bool).ravel()
    if mask.size == 0:
        return []
    padded = np.concatenate(([0], mask.astype(int), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def make_windows_segmented(
    X: np.ndarray,
    mask: np.ndarray,
    window: int = 60,
    stride: int = 1,
) -> np.ndarray:
    """
    Window only the rows where `mask` is True, never spanning a gap.

    Use this instead of `make_windows(X[mask])` whenever the mask removes
    rows from the middle of a series. Masking first and windowing second
    splices non-adjacent timesteps together and manufactures transitions
    that do not exist in the data; windowing each contiguous run separately
    keeps every window a genuine stretch of real time.

    Runs shorter than `window` contribute nothing and are dropped.
    """
    parts = [
        make_windows(X[start:end], window=window, stride=stride)
        for start, end in contiguous_runs(mask)
        if end - start >= window
    ]
    if not parts:
        raise ValueError(
            f"no contiguous run of masked rows reaches {window} timesteps; "
            "either the window is too long for this dataset or the mask is "
            "too fragmented to build sequences from"
        )
    return np.concatenate(parts)


def window_ends(mask: np.ndarray, window: int = 60, stride: int = 1) -> np.ndarray:
    """
    Index of the LAST timestep of every window lying wholly inside one
    contiguous run of True in `mask` -- the same windows
    `make_windows_segmented` builds, as indices instead of copies.

    Stacking every window up front costs n * window * n_features floats:
    about 13 GB in float64 for SWaT's ~550k training rows, more than most
    laptops have. Models gather each batch from these indices instead
    (`X[ends[:, None] + arange(-window + 1, 1)]`), so memory stays at one
    copy of X plus one batch.
    """
    parts = [
        np.arange(start + window - 1, end, stride)
        for start, end in contiguous_runs(mask)
        if end - start >= window
    ]
    if not parts:
        raise ValueError(
            f"no contiguous run of masked rows reaches {window} timesteps"
        )
    return np.concatenate(parts)


def unwindow_scores(
    scores: np.ndarray, n_original: int, window: int = 60, stride: int = 1
) -> np.ndarray:
    """
    Map per-window scores back onto the original timeline.

    The first `window - 1` timesteps have no complete window, so they inherit
    the first available score. Necessary so windowed and pointwise models
    produce score vectors of identical length and can be evaluated against
    the same y_true.
    """
    out = np.empty(n_original, dtype=float)
    out[: window - 1] = scores[0]
    positions = np.arange(window - 1, window - 1 + len(scores) * stride, stride)
    out[positions[positions < n_original]] = scores[: (positions < n_original).sum()]
    if positions.max() < n_original - 1:
        out[positions.max() + 1 :] = scores[-1]
    return out


# --------------------------------------------------------------------------
# per-dataset loaders  -- STUBS, fill in as each dataset arrives
# --------------------------------------------------------------------------
# Each loader returns a time-sorted DataFrame with one `label` column
# (1 = anomaly) and all other columns numeric features. Do the messy,
# dataset-specific cleaning HERE so nothing downstream needs special cases.

def load_swat(
    normal_path: Path | None = None,
    attack_path: Path | None = None,
    drop_stabilization_rows: int = 21_600,
) -> pd.DataFrame:
    """SWaT A1/A2 Dec 2015. iTrust ships this as two files: one week of
    normal operation and two days of attacks, each with its own header row.

    Known quirks, handled here so nothing downstream needs a special case:
      - column names carry leading/trailing spaces (e.g. ` Timestamp`).
      - the xlsx files have a blank title row; the real header is row 2.
      - the label column `Normal/Attack` is "Normal" or "Attack", but the
        shipped Attack file has rows with the typo "A ttack" -- compared
        case-insensitively with spaces stripped, so anything not "normal"
        counts as an attack.
      - `drop_stabilization_rows` removes the start of the Normal run, which
        is the plant stabilising rather than steady-state operation.
        21,600 rows = 6 hours at 1 Hz, the convention used by most published
        SWaT work; verify against the codebook once the real files are in
        hand and pass a different value here if it disagrees.
    """
    normal_path = Path(normal_path or DATA_RAW / "swat" / "SWaT_Dataset_Normal_v0.xlsx")
    attack_path = Path(attack_path or DATA_RAW / "swat" / "SWaT_Dataset_Attack_v0.xlsx")

    def _read(path: Path) -> pd.DataFrame:
        is_excel = path.suffix.lower() in (".xlsx", ".xls")
        df = pd.read_excel(path, header=1) if is_excel else pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    normal = _read(normal_path).iloc[drop_stabilization_rows:]
    attack = _read(attack_path)
    df = pd.concat([normal, attack], ignore_index=True)

    timestamp_col = next(c for c in df.columns if c.lower() == "timestamp")
    df[timestamp_col] = df[timestamp_col].astype(str).str.strip()
    try:
        # iTrust's usual format, e.g. "22/12/2015 4:30:00 PM". Explicit
        # format is much faster than inference over ~450k rows.
        df[timestamp_col] = pd.to_datetime(
            df[timestamp_col], format="%d/%m/%Y %I:%M:%S %p"
        )
    except ValueError:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], dayfirst=True)
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    label_col = next(c for c in df.columns if c.lower() == "normal/attack")
    label = (
        df[label_col].astype(str).str.strip().str.lower().str.replace(" ", "", regex=False)
    )
    df["label"] = (label != "normal").astype(int)
    df = df.drop(columns=[timestamp_col, label_col])

    feature_cols = [c for c in df.columns if c != "label"]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = [c for c in feature_cols if df[c].isna().any()]
    if bad:
        raise ValueError(
            f"non-numeric values found in SWaT feature columns after cleaning: {bad}. "
            "iTrust's SWaT physical data should be fully numeric -- inspect the raw file."
        )

    return df


# --------------------------------------------------------------------------
# A1 -> A2 primary protocol: separate streams, nominal-only partitions
# --------------------------------------------------------------------------
# The chronological and stratified splits above concatenate the normal and
# attack files and cut the result, which can put attack-labelled rows in
# training and validation. That changes the task, and hands attack-label
# models information a normal-only detector never gets. The primary protocol
# keeps the two files apart: A1 (nominal) is cut into fit / validation /
# calibration, in time order, and A2 (attacked) is touched once, for the
# final test. No A2 row or label is used for scaling, feature choice, early
# stopping, model selection or thresholds -- asserted, not just documented.

SENSOR_PREFIXES = ("FIT", "LIT", "AIT", "DPIT", "PIT")
ACTUATOR_PREFIXES = ("MV", "P", "UV")


def is_sensor(col: str) -> bool:
    return col.startswith(SENSOR_PREFIXES)


def is_actuator(col: str) -> bool:
    return not is_sensor(col) and col.startswith(ACTUATOR_PREFIXES)


def stage_of(col: str) -> int:
    """Process stage (1-6) from a SWaT tag, e.g. LIT301 -> 3."""
    digits = "".join(ch for ch in col if ch.isdigit())
    return int(digits[0]) if digits else 0


def _read_swat_file(path: Path) -> pd.DataFrame:
    """Read one iTrust SWaT file (xlsx with a title row, or csv export) into
    a frame with a parsed `timestamp`, numeric features and a 0/1 `label`."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, header=1)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    ts_col = next(c for c in df.columns if c.lower() == "timestamp")
    label_col = next(c for c in df.columns if c.lower() == "normal/attack")

    ts = df[ts_col].astype(str).str.strip()
    try:
        ts = pd.to_datetime(ts, format="%d/%m/%Y %I:%M:%S %p")
    except ValueError:
        ts = pd.to_datetime(ts, dayfirst=True)
    # anything that is not "normal" counts as attack: the attack file has
    # rows labelled "A ttack" (sic)
    label = df[label_col].astype(str).str.strip().str.lower().str.replace(" ", "", regex=False)

    out = df.drop(columns=[ts_col, label_col]).apply(pd.to_numeric, errors="coerce")
    out.insert(0, "timestamp", ts.to_numpy())
    out["label"] = (label != "normal").astype(int).to_numpy()
    return out


def stream_report(df: pd.DataFrame, name: str) -> dict:
    """Cadence, duplicates, missingness and label counts of one stream --
    stored with every run so the data version is checkable."""
    dt = df["timestamp"].diff().dt.total_seconds().iloc[1:]
    feats = df.drop(columns=["timestamp", "label"])
    return {
        "stream": name,
        "rows": int(len(df)),
        "start": str(df["timestamp"].iloc[0]),
        "end": str(df["timestamp"].iloc[-1]),
        "cadence_s_mode": float(dt.mode().iloc[0]) if len(dt) else None,
        "gaps_not_1s": int((dt != 1).sum()),
        "duplicate_timestamps": int(df["timestamp"].duplicated().sum()),
        "missing_values": int(feats.isna().sum().sum()),
        "attack_rows": int(df["label"].sum()),
        "n_features": int(feats.shape[1]),
    }


def load_swat_streams(profile: str = "swat2015", cache: bool = True):
    """
    The nominal A1 and attacked A2 streams of SWaT 2015, kept separate,
    timestamps preserved. Returns (a1, a2, report).

    Parsed frames are cached as pickles under data/processed/ (gitignored,
    same iTrust terms as data/raw/); delete them to force a re-parse.
    """
    from config import DATA_PROCESSED, DATA_PROFILES

    prof = DATA_PROFILES[profile]
    streams, sources = {}, {}
    for name, rel, alt in (("A1", prof["normal_file"], None),
                           ("A2", prof["attack_file"], prof.get("attack_csv"))):
        # the A2 csv is an export of the same iTrust xlsx, read when present
        use = alt if alt is not None and (DATA_RAW / alt).exists() else rel
        sources[name] = use                       # relative to data/raw, recorded below
        cache_path = DATA_PROCESSED / f"swat2015_{name}_raw.pkl"   # same files for every profile
        if cache and cache_path.exists():
            streams[name] = pd.read_pickle(cache_path)
            continue
        df = _read_swat_file(DATA_RAW / use)
        if cache:
            df.to_pickle(cache_path)
        streams[name] = df

    a1, a2 = streams["A1"], streams["A2"]
    raw_a1_rows = len(a1)
    if prof.get("a1_start"):
        a1 = a1[a1["timestamp"] >= pd.Timestamp(prof["a1_start"])]
    a1 = a1.iloc[prof.get("startup_rows", 0):].reset_index(drop=True)

    if a1["label"].any():
        raise ValueError(f"A1 contains {int(a1['label'].sum())} attack-labelled rows")
    if a1["timestamp"].iloc[-1] >= a2["timestamp"].iloc[0]:
        raise ValueError("A1 overlaps A2 in time")
    for name, df in (("A1", a1), ("A2", a2)):
        if df.drop(columns=["timestamp", "label"]).isna().any().any():
            raise ValueError(f"{name} has missing values; decide a policy before running")

    report = {
        "profile": profile,
        "A1_raw_rows": int(raw_a1_rows),
        "A1": {**stream_report(a1, "A1"), "source": sources["A1"]},
        "A2": {**stream_report(a2, "A2"), "source": sources["A2"]},
    }
    return a1, a2, report


@dataclass
class NominalSplit:
    """A1 fit / validation / calibration plus the held-out A2 test stream.

    Features are already preprocessed by a Preprocessor fit on X_fit only.
    `y_test` is the A2 row label; nothing upstream of evaluation may read it.
    """

    X_fit: np.ndarray
    X_val: np.ndarray
    X_cal: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    t_test: np.ndarray
    boundaries: dict
    preprocessor: "Preprocessor"
    name: str = "swat-a1a2"

    @property
    def n_features(self) -> int:
        return self.X_fit.shape[1]

    def columns(self, kind: str) -> list[int]:
        """Indices of 'sensor' or 'actuator' columns."""
        test = is_sensor if kind == "sensor" else is_actuator
        return [i for i, c in enumerate(self.feature_names) if test(c)]

    def as_split(self) -> Split:
        """View as a Split with all-normal labels, so the existing
        unsupervised windowed detectors (lstm_ae, lstm_forecast) train on
        A1-fit and early-stop on A1-validation unchanged."""
        return Split(
            self.X_fit, np.zeros(len(self.X_fit), dtype=int),
            self.X_val, np.zeros(len(self.X_val), dtype=int),
            self.X_test, self.y_test, self.feature_names, self.name,
        )

    def summary(self) -> str:
        b = self.boundaries
        lines = [f"{self.name}: {self.n_features} features after preprocessing"]
        for part in ("fit", "val", "cal", "test"):
            lines.append(f"  {part:<4} {b[part]['rows']:>8,} rows  {b[part]['start']} -> {b[part]['end']}")
        lines.append(f"  A2 attack rows {int(self.y_test.sum()):,} ({self.y_test.mean():.4f})")
        return "\n".join(lines)


class Preprocessor:
    """
    Every parameter is fit on A1-fit rows and applied unchanged to A1-val,
    A1-cal and A2: constant-column removal, scaling (standard / min-max /
    robust), optional winsorisation at fit-row quantiles.
    """

    def __init__(self, scaler: str = "standard", winsorize=None, **_ignored):
        self.scaler_name = scaler
        self.winsorize = winsorize

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        from sklearn.preprocessing import MinMaxScaler, RobustScaler

        self.dropped_ = [c for c in X.columns if X[c].nunique() <= 1]
        self.columns_ = [c for c in X.columns if c not in self.dropped_]
        Z = X[self.columns_].to_numpy(dtype=float)
        if self.winsorize:
            lo, hi = self.winsorize
            self.clip_ = (np.quantile(Z, lo, axis=0), np.quantile(Z, hi, axis=0))
            Z = np.clip(Z, *self.clip_)
        scaler = {"standard": StandardScaler, "minmax": MinMaxScaler,
                  "robust": RobustScaler}[self.scaler_name]()
        self.scaler_ = scaler.fit(Z)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = X[self.columns_].to_numpy(dtype=float)
        if self.winsorize:
            Z = np.clip(Z, *self.clip_)
        return self.scaler_.transform(Z)

    def describe(self) -> dict:
        return {"scaler": self.scaler_name, "winsorize": self.winsorize,
                "dropped_constant_in_fit": self.dropped_, "n_kept": len(self.columns_)}


def nominal_split(
    data_profile: str = "swat2015",
    preprocess: str = "shared",
    streams=None,
) -> tuple[NominalSplit, dict]:
    """Build the primary-protocol split. Returns (split, data report)."""
    from config import DATA_PROFILES, PREPROCESS_PROFILES

    a1, a2, report = streams if streams is not None else load_swat_streams(data_profile)
    prof = DATA_PROFILES[data_profile]
    n = len(a1)
    i_fit = int(n * prof["fit_frac"])
    i_val = i_fit + int(n * prof["val_frac"])

    feats = [c for c in a1.columns if c not in ("timestamp", "label")]
    if list(a2.columns) != list(a1.columns):
        raise ValueError("A1 and A2 columns differ")
    parts = {"fit": a1.iloc[:i_fit], "val": a1.iloc[i_fit:i_val],
             "cal": a1.iloc[i_val:], "test": a2}

    pre = Preprocessor(**PREPROCESS_PROFILES[preprocess]).fit(parts["fit"][feats])
    X = {k: pre.transform(v[feats]) for k, v in parts.items()}
    boundaries = {
        k: {"rows": int(len(v)), "start": str(v["timestamp"].iloc[0]),
            "end": str(v["timestamp"].iloc[-1])}
        for k, v in parts.items()
    }
    split = NominalSplit(
        X["fit"], X["val"], X["cal"], X["test"],
        a2["label"].to_numpy().astype(int), pre.columns_,
        a2["timestamp"].to_numpy(), boundaries, pre,
        name=f"{data_profile}-a1a2-{preprocess}",
    )
    report = {**report, "preprocess": preprocess, "preprocessor": pre.describe(),
              "boundaries": boundaries}
    return split, report


def load_wadi() -> pd.DataFrame:
    """WADI A1 Oct 2017. Large, and several columns are entirely NaN.
    Decide the imputation strategy once and record it in notes/datasets.md."""
    raise NotImplementedError("Fill in once the iTrust download arrives.")


def load_morris() -> pd.DataFrame:
    """Mississippi State gas pipeline SCADA. The contrast case.
    Verify empirically whether records are genuinely near-independent."""
    raise NotImplementedError("Fill in once located.")


def load_batadal() -> pd.DataFrame:
    """BATADAL. Small and clean, useful as a smoke test."""
    raise NotImplementedError("Fill in once the iTrust download arrives.")


LOADERS = {
    "swat": load_swat,
    "wadi": load_wadi,
    "morris": load_morris,
    "batadal": load_batadal,
}


def load(name: str, protocol: str = "chronological", **kwargs) -> Split:
    """Load a registered dataset and split it.

    protocol="chronological" (default, deployment-realistic) or
    "stratified" (episode-stratified, purged; see stratified_episode_split).
    """
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}. Available: {list(LOADERS)}")
    df = LOADERS[name](**kwargs)
    if protocol == "chronological":
        return chronological_split(df, name=name)
    if protocol == "stratified":
        return stratified_episode_split(df, name=f"{name}-stratified")
    raise ValueError(f"unknown protocol {protocol!r}")


# --------------------------------------------------------------------------
# synthetic data, for testing the pipeline before any real data arrives
# --------------------------------------------------------------------------

def make_synthetic(
    n: int = 20_000,
    n_features: int = 20,
    n_attacks: int = 12,
    attack_len: int = 50,
    temporal: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Generate a fake ICS-shaped dataset.

    `temporal=True` produces anomalies that are only visible as a change in
    the sequence dynamics (an autocorrelated drift), so a pointwise model
    should struggle. `temporal=False` produces anomalies visible in a single
    row (a level shift), so boosting should handle them easily.

    This is how you sanity-check that the whole pipeline behaves as the
    project's central hypothesis predicts, before real data arrives.
    """
    rng = np.random.default_rng(seed)

    # autocorrelated normal process
    X = np.zeros((n, n_features))
    X[0] = rng.normal(size=n_features)
    for t in range(1, n):
        X[t] = 0.95 * X[t - 1] + rng.normal(scale=0.3, size=n_features)

    y = np.zeros(n, dtype=int)
    starts = rng.choice(np.arange(n // 4, n - attack_len), size=n_attacks, replace=False)
    affected = rng.choice(n_features, size=max(1, n_features // 4), replace=False)

    for s in starts:
        y[s : s + attack_len] = 1
        if temporal:
            # subtle: break the autocorrelation without changing the marginal
            X[s : s + attack_len, affected] = rng.normal(
                scale=X[:, affected].std(), size=(attack_len, len(affected))
            )
        else:
            # obvious: shift the level, visible in any single row
            X[s : s + attack_len, affected] += 4.0

    df = pd.DataFrame(X, columns=[f"sensor_{i:02d}" for i in range(n_features)])
    df["label"] = y
    return df


if __name__ == "__main__":
    for temporal in (False, True):
        kind = "temporal" if temporal else "pointwise"
        df = make_synthetic(temporal=temporal)
        split = chronological_split(df, name=f"synthetic-{kind}")
        print(split.summary())
        print()
