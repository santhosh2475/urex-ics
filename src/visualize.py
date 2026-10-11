"""
Visual diagnostics for the pipeline, in seaborn.

Every function here takes plain objects already used elsewhere in the
project (a Split, a fitted Detector, a scores/y_true pair) and saves one
PNG -- nothing here recomputes a model or a split, it only draws what
already exists. Meant to be called from a short script or notebook cell,
one function per question you want answered:

    scaling_summary(df_raw, split)         -> did StandardScaler actually
                                               normalise all 51 sensors?
    plot_split_timeline(df_raw, split, ..) -> where do train/val/test
                                               actually fall in time?
    plot_feature_importance(detector, ...) -> which sensors does a
                                               tree-based model rely on?
    plot_loss_curve(detector, ...)         -> did a deep model's training
                                               loss actually go down?
    plot_score_timeline(y_true, scores,..) -> where does a model's score
                                               spike relative to real
                                               attacks?
    plot_score_distribution(y_true, ...)   -> does a model separate the
                                               two classes' scores at all?
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data import Split, contiguous_runs

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------
# 1. what StandardScaler actually did to the 51 sensors
# --------------------------------------------------------------------------

def scaling_summary(df_raw: pd.DataFrame, split: Split) -> pd.DataFrame:
    """
    One row per sensor: its raw (pre-scaling) mean/std/min/max, and its
    scaled (post-scaling, what the models actually train on) mean/std.

    df_raw is the full, UNSCALED DataFrame straight out of load_swat()
    (with the label column still attached -- dropped here). Because
    chronological_split() slices positionally and never reorders rows,
    df_raw.iloc[:len(split.y_train)] is exactly the same rows as
    split.X_train, just before scaling instead of after.
    """
    raw_train = df_raw.drop(columns=["label"]).iloc[: len(split.y_train)]
    scaled_train = split.X_train  # already a numpy array, same row order

    rows = []
    for i, name in enumerate(split.feature_names):
        raw_col = raw_train[name].to_numpy()
        scaled_col = scaled_train[:, i]
        rows.append(
            {
                "sensor": name,
                "raw_mean": raw_col.mean(),
                "raw_std": raw_col.std(),
                "raw_min": raw_col.min(),
                "raw_max": raw_col.max(),
                "scaled_mean": scaled_col.mean(),
                "scaled_std": scaled_col.std(),
            }
        )
    return pd.DataFrame(rows)


def plot_scaling_effect(summary: pd.DataFrame, out_path: Path) -> None:
    """
    Two horizontal panels, same sensor order on the y-axis: raw mean +/-1
    std (left, log-scaled x -- raw sensors span ~0.001 to ~600, a linear
    axis would flatten everything to one line) and scaled mean +/-1 std
    (right, linear x, everything now near mean=0 std=1). The point isn't
    that the two panels are on the same scale -- it's that the RIGHT panel
    is uniform across 51 wildly different sensors and the LEFT panel is not.
    """
    order = summary.sort_values("raw_std")["sensor"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 13), sharey=True)

    for ax, mean_col, std_col, title, log in [
        (axes[0], "raw_mean", "raw_std", "Before scaling (raw units)", True),
        (axes[1], "scaled_mean", "scaled_std", "After StandardScaler", False),
    ]:
        s = summary.set_index("sensor").loc[order]
        # a symmetric-log-safe floor: raw means can be 0 or negative for a
        # few actuator state columns, so clip only the x-axis, not the data
        ax.errorbar(
            s[mean_col], np.arange(len(s)), xerr=s[std_col],
            fmt="o", color="#1F6F9C", ecolor="#1F6F9C55", markersize=3,
        )
        ax.set_yticks(np.arange(len(s)))
        ax.set_yticklabels(s.index, fontsize=6)
        ax.set_title(title)
        ax.set_xlabel("mean ± 1 std")
        if log:
            ax.set_xscale("symlog")

    fig.suptitle("StandardScaler's effect on all 51 SWaT sensors (training rows only)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {out_path}")


# --------------------------------------------------------------------------
# 2. where train / val / test actually fall in time
# --------------------------------------------------------------------------

def plot_split_timeline(
    df_raw: pd.DataFrame, split: Split, sensor: str, out_path: Path, stride: int = 20
) -> None:
    """
    One real sensor's raw value across the whole dataset, coloured by which
    split it ended up in -- the same visual idea as colouring a time series
    red/blue at a train/test cutoff, except here there are two cutoffs and
    they're not evenly spaced (Section 2.3 of the theory doc: the later
    cutoff got pushed forward to avoid bisecting an attack episode).

    stride subsamples for a readable, fast-to-render line (925,119 raw
    points would be solid ink at this figure width).
    """
    values = df_raw[sensor].to_numpy()
    n_train, n_val = len(split.y_train), len(split.y_val)
    labels = np.array(
        ["train"] * n_train + ["val"] * n_val + ["test"] * (len(values) - n_train - n_val)
    )
    df = pd.DataFrame({"t": np.arange(len(values)), sensor: values, "split": labels})
    df = df.iloc[::stride]

    fig, ax = plt.subplots(figsize=(13, 4))
    sns.lineplot(
        data=df, x="t", y=sensor, hue="split", hue_order=["train", "val", "test"],
        palette={"train": "#1F6F9C", "val": "#B8721F", "test": "#2F7A4F"},
        linewidth=0.6, ax=ax,
    )
    ax.axvline(n_train, color="black", linestyle="--", linewidth=1)
    ax.axvline(n_train + n_val, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("seconds since 2015-12-28 10:00:00")
    ax.set_title(f"{sensor} across the real episode-aware train/val/test split")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {out_path}")


# --------------------------------------------------------------------------
# 3. which sensors a tree-based model actually relies on
# --------------------------------------------------------------------------

def plot_feature_importance(detector, feature_names: list[str], out_path: Path, top_n: int = 15) -> None:
    """
    Works for anything exposing scikit-learn's `.feature_importances_`
    (random_forest, j48, and xgboost's underlying model) -- higher means
    the model's splits relied on that sensor more, averaged across every
    split in every tree.
    """
    model = getattr(detector, "model", detector)
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"{detector.name} has no feature_importances_")

    imp = pd.DataFrame(
        {"sensor": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=imp, x="importance", y="sensor", color="#1F6F9C", ax=ax)
    ax.set_title(f"{detector.name}: top {top_n} sensors by split importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {out_path}")


# --------------------------------------------------------------------------
# 4. did a deep model's training loss actually go down
# --------------------------------------------------------------------------

def plot_loss_curve(detector, out_path: Path) -> None:
    """lstm_ae, lstm_forecast, and cnn_classifier all record one average
    loss per epoch in self.loss_history during fit() -- this just plots it."""
    history = getattr(detector, "loss_history", [])
    if not history:
        raise ValueError(f"{detector.name} has no recorded loss_history -- call fit() first")

    df = pd.DataFrame({"epoch": range(len(history)), "loss": history})
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=df, x="epoch", y="loss", marker="o", color="#B8721F", ax=ax)
    ax.set_title(f"{detector.name}: training loss per epoch")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {out_path}")


# --------------------------------------------------------------------------
# 5. where a model's score spikes relative to the real attacks
# --------------------------------------------------------------------------

def plot_score_timeline(
    y_true: np.ndarray, scores: np.ndarray, threshold: float, out_path: Path, title: str, stride: int = 5
) -> None:
    """
    The anomaly-score-over-time view: the model's continuous score across
    the test window, the calibrated threshold as a flat line, and the real
    attack windows shaded behind it -- so you can see directly whether a
    score spike actually lines up with a real attack, or not.
    """
    df = pd.DataFrame({"t": np.arange(len(scores)), "score": scores}).iloc[::stride]

    fig, ax = plt.subplots(figsize=(13, 4))
    for start, end in contiguous_runs(y_true.astype(bool)):
        ax.axvspan(start, end, color="#2F7A4F", alpha=0.15, linewidth=0)
    sns.lineplot(data=df, x="t", y="score", color="#1F6F9C", linewidth=0.7, ax=ax)
    ax.axhline(threshold, color="#A23B3B", linestyle="--", linewidth=1, label="threshold")
    ax.set_xlabel("seconds into the test window")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {out_path}")


# --------------------------------------------------------------------------
# 6. does a model's score actually separate the two classes
# --------------------------------------------------------------------------

def plot_score_distribution(y_true: np.ndarray, scores: np.ndarray, out_path: Path, title: str) -> None:
    """Two overlaid histograms of the same score, split by true label. A
    model that separates the classes shows two distinct humps; a model
    that isn't working shows one blob with no real separation."""
    df = pd.DataFrame({
        "score": scores,
        "label": np.where(y_true == 1, "attack", "normal"),
    })
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(
        data=df, x="score", hue="label", hue_order=["normal", "attack"],
        palette={"normal": "#1F6F9C", "attack": "#A23B3B"},
        stat="density", common_norm=False, element="step", bins=50, ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {out_path}")
