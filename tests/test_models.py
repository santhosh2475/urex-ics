"""Regression tests for the detectors. Run with: python -m pytest tests/ -v

Each test here corresponds to a bug that silently corrupted results rather
than raising, which is the only kind worth a permanent test.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from data import (
    chronological_split,
    contiguous_runs,
    make_windows_segmented,
    make_synthetic,
)
from models import GradientBoostingDetector, RandomDetector, REGISTRY


# --------------------------------------------------------------------------
# windowing must never span a masked-out gap
# --------------------------------------------------------------------------

def test_contiguous_runs():
    assert contiguous_runs(np.array([1, 1, 0, 0, 1])) == [(0, 2), (4, 5)]
    assert contiguous_runs(np.array([1, 1, 1])) == [(0, 3)]
    assert contiguous_runs(np.array([0, 0])) == []


def test_windows_never_span_a_removed_segment():
    """The splice bug: masking rows out and windowing the result joins
    non-adjacent timesteps and invents transitions that never occurred."""
    X = np.arange(200, dtype=float).reshape(200, 1)
    mask = np.ones(200, bool)
    mask[100:110] = False           # an attack segment removed from the middle

    win = make_windows_segmented(X, mask, window=10)

    # X is a ramp, so within any genuine window every step is exactly +1.
    steps = np.diff(win[:, :, 0], axis=1)
    assert (steps == 1).all(), "a window spans the gap at rows 100-109"
    assert len(win) == 91 + 81     # both runs windowed, neither bridged


def test_segmented_windowing_rejects_over_fragmented_masks():
    X = np.arange(100, dtype=float).reshape(100, 1)
    mask = np.zeros(100, bool)
    mask[::2] = True               # no run longer than 1
    with pytest.raises(ValueError, match="contiguous run"):
        make_windows_segmented(X, mask, window=10)


# --------------------------------------------------------------------------
# engineered features must not reach forward in time
# --------------------------------------------------------------------------

def test_lag_features_do_not_leak_the_future():
    """Perturbing row k must never change the features of any row before k.
    The old `.bfill()` filled the NaN head of shift/diff from later rows."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    det = GradientBoostingDetector(lags=[1, 5, 30])

    before = det._featurise(X)
    X_perturbed = X.copy()
    X_perturbed[120:] += 99.0                      # change only the future
    after = det._featurise(X_perturbed)

    assert np.allclose(before[:120], after[:120]), (
        "features before row 120 changed when only later rows were edited"
    )


def test_lag_features_have_no_nans():
    X = np.random.default_rng(0).normal(size=(100, 3))
    out = GradientBoostingDetector(lags=[1, 5, 30])._featurise(X)
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------
# the random baseline is the headline finding, so it must be reproducible
# --------------------------------------------------------------------------

def _random_cycle(split):
    """One full fit -> score(val) -> score(test) cycle, as experiment.py does."""
    det = RandomDetector().fit(split)
    return det.score(split.X_val), det.score(split.X_test)


def test_random_detector_cycle_is_reproducible():
    """The Kim et al. number must not move between runs, or from reusing the
    detector instance. fit() resets the stream, so every cycle is identical."""
    df = make_synthetic(n=1200, n_features=4, n_attacks=3, attack_len=30, seed=1)
    split = chronological_split(df, name="rand")

    val_a, test_a = _random_cycle(split)
    val_b, test_b = _random_cycle(split)
    assert (val_a == val_b).all() and (test_a == test_b).all()

    det = RandomDetector()
    det.score(np.zeros((77, 3)))                   # stale prior use
    val_c, test_c = _random_cycle(split)
    assert (val_c == val_a).all() and (test_c == test_a).all()


def test_random_val_and_test_scores_are_independent_draws():
    """Equal-length val and test splits must NOT receive the same scores. If
    they did, the threshold taken on validation would be a threshold taken on
    test, and commitment 4 would be broken without anything raising."""
    split = chronological_split(
        make_synthetic(n=1200, n_features=4, n_attacks=3, attack_len=30, seed=1),
        name="rand",
    )
    val, test = _random_cycle(split)
    n = min(len(val), len(test))
    assert not np.allclose(val[:n], test[:n])


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def test_registry_entries_are_all_constructible():
    for name, factory in REGISTRY.items():
        det = factory()
        assert isinstance(det.name, str) and det.name
        assert hasattr(det, "fit") and hasattr(det, "score")


def test_lag_variant_is_reachable_by_name():
    """The lags-vs-no-lags contrast is the project's most informative
    experiment; it has to be runnable without editing code."""
    assert REGISTRY["xgboost_lags"]().lags == [1, 5, 30]
    assert REGISTRY["xgboost"]().lags is None


# --------------------------------------------------------------------------
# scores from every model must align to the same timeline
# --------------------------------------------------------------------------

def test_all_detectors_return_one_score_per_row():
    """Windowed and pointwise models must produce identical-length score
    vectors, or they are not being evaluated against the same y_true."""
    df = make_synthetic(n=1200, n_features=4, n_attacks=3, attack_len=30, seed=1)
    split = chronological_split(df, name="align")

    for name in ("random", "lstm_ae", "lstm_forecast"):
        det = REGISTRY[name]()
        for attr in ("window", "epochs"):
            if hasattr(det, attr):
                setattr(det, attr, 20 if attr == "window" else 1)
        det.fit(split)
        assert len(det.score(split.X_test)) == len(split.y_test), name


def test_windowed_detectors_refuse_input_shorter_than_the_window():
    df = make_synthetic(n=1200, n_features=4, n_attacks=3, attack_len=30, seed=1)
    split = chronological_split(df, name="short")
    det = REGISTRY["lstm_forecast"]()
    det.window, det.epochs = 20, 1
    det.fit(split)
    with pytest.raises(ValueError, match="at least"):
        det.score(split.X_test[:5])
