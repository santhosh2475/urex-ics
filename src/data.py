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

import re
from dataclasses import dataclass
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

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

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

def chronological_split(
    df: pd.DataFrame,
    label_col: str = "label",
    name: str = "unnamed",
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    scale: bool = True,
) -> Split:
    """
    Split a time-ordered dataframe into train/val/test by position.

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
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))

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

_HAI_WINDOW_RE = re.compile(
    r"\[\s*\d+\]\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
)


def _parse_hai_attack_windows(label_path: Path) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Parse the '[ N]  start  end  duration' rows out of the HAI label
    summary file. This particular HAI split ships attack windows as a
    human-readable report, not a per-row label column."""
    text = Path(label_path).read_text()
    windows = [
        (pd.Timestamp(m.group(1)), pd.Timestamp(m.group(2)))
        for m in _HAI_WINDOW_RE.finditer(text)
    ]
    if not windows:
        raise ValueError(f"no attack windows parsed from {label_path}")
    return windows


def load_hai(
    data_path: Path | None = None,
    label_path: Path | None = None,
) -> pd.DataFrame:
    """HAI, hai3 test split (2022-08-12 16:00 to 2022-08-13 07:00, 1 Hz).

    `data_path` is a plain timestamp + sensor/actuator CSV with no label
    column. `label_path` is a report listing attack windows as
    [index] start_time end_time duration; a row is labelled 1 if its
    timestamp falls inside any window (inclusive of both ends)."""
    data_path = data_path or DATA_RAW / "hai" / "hai3_test_220812.csv"
    label_path = label_path or DATA_RAW / "hai" / "hai3_test_220812_attack_windows.txt"

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    label = np.zeros(len(df), dtype=int)
    for start, end in _parse_hai_attack_windows(label_path):
        label[(df["timestamp"] >= start) & (df["timestamp"] <= end)] = 1
    df["label"] = label

    return df.drop(columns=["timestamp"])


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
    normal_path = Path(normal_path or DATA_RAW / "swat" / "SWaT_Dataset_Normal_v1.xlsx")
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
    "hai": load_hai,
    "swat": load_swat,
    "wadi": load_wadi,
    "morris": load_morris,
    "batadal": load_batadal,
}


def load(name: str, **kwargs) -> Split:
    """Load a registered dataset and return a chronological split."""
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}. Available: {list(LOADERS)}")
    df = LOADERS[name](**kwargs)
    return chronological_split(df, name=name)


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
