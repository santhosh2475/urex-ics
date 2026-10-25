"""Regression tests for the episode-stratified split, block-aware windowing,
and Pearson feature selection. Run with: python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from data import (
    chronological_split,
    contiguous_runs,
    make_synthetic,
    make_windows_segmented,
    stratified_episode_split,
    window_ends,
)
from experiment import score_part
from features import pearson_select, subset
from models import REGISTRY


def _staged_synthetic(n=30_000, seed=0):
    """Synthetic data whose attacks each hit one named stage, spaced far
    enough apart (> merge_gap) to form separate incidents."""
    rng = np.random.default_rng(seed)
    cols = ["FIT101", "LIT101", "AIT201", "P201", "LIT301", "FIT401", "FIT501", "FIT601"]
    X = rng.normal(size=(n, len(cols)))
    X[:, 3] = 1.0  # constant, like SWaT's never-switched pumps
    y = np.zeros(n, dtype=int)
    starts = np.arange(8_000, n - 600, 1_500)
    for k, s in enumerate(starts):
        y[s : s + 200] = 1
        X[s : s + 200, [0, 2, 4, 5, 6][k % 5]] += 5.0
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y
    return df


def test_stratified_split_loses_no_attack_rows_and_purges_only_normals():
    df = _staged_synthetic()
    sp = stratified_episode_split(df, name="t", merge_gap=1200, purge=300, verbose=False)
    n_attack = sp.y_train.sum() + sp.y_val.sum() + sp.y_test.sum()
    assert n_attack == df["label"].sum()
    for part in ("train", "val", "test"):
        assert getattr(sp, f"y_{part}").sum() > 0, part


def test_stratified_split_covers_every_stage_with_enough_incidents():
    df = _staged_synthetic()
    sp = stratified_episode_split(df, name="t", merge_gap=1200, purge=300, verbose=False)
    inc = sp.incidents
    for stage, g in inc.groupby("stage"):
        if len(g) >= 3:
            assert set(g["part"]) == {0, 1, 2}, stage


def test_stratified_blocks_tile_each_part_exactly():
    df = _staged_synthetic()
    sp = stratified_episode_split(df, name="t", merge_gap=1200, purge=300, verbose=False)
    for part in ("train", "val", "test"):
        bounds = sp.bounds(part)
        assert bounds[0][0] == 0
        assert bounds[-1][1] == len(getattr(sp, f"y_{part}"))
        assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))


def test_window_ends_matches_segmented_windows():
    """Index-based windowing must build exactly the windows the old
    materialised version did, just without the copies."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    mask = np.ones(500, dtype=bool)
    mask[100:130] = False
    mask[300] = False
    ends = window_ends(mask, window=20)
    gathered = X[ends[:, None] + np.arange(-19, 1)]
    assert np.array_equal(gathered, make_windows_segmented(X, mask, window=20))


def test_continuity_mask_stops_windows_crossing_a_block_join():
    df = _staged_synthetic()
    sp = stratified_episode_split(df, name="t", merge_gap=1200, purge=300, verbose=False)
    w = 60
    ends = window_ends(sp.continuity_mask("val"), window=w)
    starts = {s for s, _ in sp.bounds("val")[1:]}
    for e in ends:
        assert not any(e - w + 1 < s <= e for s in starts)


def test_block_scoring_keeps_one_score_per_row():
    df = _staged_synthetic()
    sp = stratified_episode_split(df, name="t", merge_gap=1200, purge=300, verbose=False)
    det = REGISTRY["lstm_forecast"](window=20, epochs=1, train_stride=10)
    det.fit(sp)
    assert len(score_part(det, sp, "test")) == len(sp.y_test)


def test_pearson_select_drops_constant_and_redundant_features():
    df = make_synthetic(n=4000, n_features=5, n_attacks=4, attack_len=50,
                        temporal=False, seed=2)
    df["dup"] = df["sensor_00"] * 2.0 + 1.0      # perfectly redundant
    df["flat"] = 3.0                             # constant
    sp = chronological_split(df, name="fs")
    keep, table = pearson_select(sp, max_redundancy=0.95)
    names = [sp.feature_names[j] for j in keep]
    assert "flat" not in names
    assert not {"sensor_00", "dup"} <= set(names)
    assert table.loc[table.feature == "flat", "status"].item() == "constant in train"
    fs = subset(sp, keep)
    assert fs.X_test.shape[1] == len(keep)


def test_bce_models_return_one_score_per_row():
    df = make_synthetic(n=3000, n_features=4, n_attacks=6, attack_len=40,
                        temporal=False, seed=3)
    sp = chronological_split(df, name="bce")
    for name, kw in [("logistic_regression", {}), ("adaboost", {"n_estimators": 5}),
                     ("logreg_adam", {"epochs": 2}), ("mlp_bce", {"epochs": 2}),
                     ("cnn_classifier", {"epochs": 1, "window": 20})]:
        det = REGISTRY[name](**kw)
        det.fit(sp)
        assert len(det.score(sp.X_test)) == len(sp.y_test), name


def test_episode_folds_test_every_incident_exactly_once():
    from data import episode_folds

    df = _staged_synthetic()
    folds = episode_folds(df, n_folds=4, name="t", merge_gap=1200, purge=300)
    tested = np.concatenate([
        sp.incidents.loc[sp.incidents["part"] == 2, "incident"].to_numpy() for sp in folds
    ])
    assert sorted(tested) == sorted(folds[0].incidents["incident"])
    for sp in folds:
        # train, val and test never share an original row
        idx = sp.row_index
        assert not set(idx["train"]) & set(idx["test"])
        assert not set(idx["val"]) & set(idx["test"])
        assert sp.y_test.sum() > 0 and sp.y_val.sum() > 0
