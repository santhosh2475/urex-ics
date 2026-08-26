"""
Detectors.

Every model exposes the SAME interface:

    detector.fit(split)          # trains on split's training window only
    scores = detector.score(X)   # higher score = more anomalous

That uniformity is the point. If the two families needed different calling
conventions, the experiment grid would fill with special cases and the
comparison would drift out of alignment without anyone noticing.

Note the asymmetry the interface has to absorb: gradient boosting is
SUPERVISED (it needs labels) while the deep models are UNSUPERVISED (they
train on normal data only). Both draw from the same training window, so
neither sees data the other could not have seen.

The deep family is represented by TWO detectors and both are always reported.
They differ only in what the score measures -- reconstruction for the
autoencoder, one-step prediction for the forecaster -- and that difference
decides which anomalies each can see at all. See LSTMAutoencoderDetector's
docstring; reporting only one would attribute a scoring artefact to the whole
deep-learning family.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial

import numpy as np

from config import SEED, WINDOW_SIZE
from data import Split, make_windows, make_windows_segmented, unwindow_scores


def _resolve_device(requested: str | None) -> str:
    """Pick a torch device, tolerating builds with no MPS backend at all."""
    import torch

    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    # torch.backends.mps does not exist on every build, so getattr, not a
    # bare attribute access that would raise on a Linux lab machine.
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class Detector(ABC):
    """Common interface for every model in the benchmark."""

    name: str = "detector"
    supervised: bool = False

    @abstractmethod
    def fit(self, split: Split) -> "Detector":
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return one score per row of X. Higher means more anomalous."""
        ...

    def fit_score(self, split: Split) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: fit, then score validation and test."""
        self.fit(split)
        return self.score(split.X_val), self.score(split.X_test)


# --------------------------------------------------------------------------
# traditional ML
# --------------------------------------------------------------------------

class GradientBoostingDetector(Detector):
    """
    XGBoost, supervised. The traditional-ML anchor.

    `lags` adds engineered temporal features (lagged values, rolling means,
    first differences). Running with lags=None and lags=[1,5,30] is the single
    most informative experiment in the project: it measures how much of deep
    learning's advantage is genuinely architectural, and how much a
    practitioner can recover with an afternoon of feature engineering.
    """

    supervised = True

    def __init__(self, lags: list[int] | None = None, **params):
        self.lags = lags
        self.name = "xgboost" + (f"+lags{lags}" if lags else "")
        self.params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": SEED,
            "n_jobs": -1,
            "eval_metric": "aucpr",  # not accuracy: anomalies are rare
            # only does anything because fit() now passes eval_set=val below
            "early_stopping_rounds": 20,
            **params,
        }
        self.model = None
        # tail of the previously-scored array, carried forward so lag/roll/
        # diff features at the START of val and test are not cold-started at
        # zero. Chronological order matters: seeded from X_train's tail after
        # fit(), then advanced to each split's own tail after every score()
        # call. Correct as long as score() is called in time order once per
        # split -- exactly what fit_score() and experiment.py do.
        self._tail: np.ndarray | None = None

    def _featurise(self, X: np.ndarray) -> np.ndarray:
        if not self.lags:
            return X
        import pandas as pd

        df = pd.DataFrame(X)
        parts = [df]
        for lag in self.lags:
            parts.append(df.shift(lag).add_suffix(f"_lag{lag}"))
            parts.append(df.rolling(lag, min_periods=1).mean().add_suffix(f"_roll{lag}"))
            parts.append(df.diff(lag).add_suffix(f"_diff{lag}"))
        # Fill the NaN head with 0, NOT bfill. `shift` and `diff` leave NaN in
        # the first `lag` rows, and bfill would fill them by reaching FORWARD
        # in time -- row 0's diff feature would be built from the value at row
        # `lag`. That is a future leak, small but the exact kind commitment 2
        # rules out. Features are standardised upstream, so 0 is the training
        # mean and the most neutral value available.
        return pd.concat(parts, axis=1).fillna(0.0).to_numpy(dtype=float)

    def fit(self, split: Split) -> "GradientBoostingDetector":
        from xgboost import XGBClassifier

        if split.y_train.sum() == 0:
            raise ValueError(
                "Supervised model cannot train on a split with no anomalies. "
                "This is a genuine finding for the label-availability "
                "experiment, not a bug."
            )

        # handle imbalance rather than letting the model predict all-normal
        n_neg, n_pos = (split.y_train == 0).sum(), (split.y_train == 1).sum()

        # Early stopping needs a val set that can actually discriminate.
        # data.py permits y_val to contain zero anomalies (only y_test is
        # guarded); aucpr on an all-negative eval set is degenerate and
        # stops training after round 0 -- silently worse than not stopping
        # at all. Fall back to a plain, full-length fit in that case.
        use_early_stop = 0 < split.y_val.sum() < len(split.y_val)
        params = dict(self.params)
        if not use_early_stop:
            params.pop("early_stopping_rounds", None)

        self.model = XGBClassifier(
            scale_pos_weight=n_neg / max(n_pos, 1), **params
        )
        fit_kwargs = {"verbose": False}
        if use_early_stop:
            fit_kwargs["eval_set"] = [(self._featurise(split.X_val), split.y_val)]
        self.model.fit(self._featurise(split.X_train), split.y_train, **fit_kwargs)

        if self.lags:
            self._tail = split.X_train[-max(self.lags) :]
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() first")
        if self.lags and self._tail is not None:
            n_ctx = len(self._tail)
            feats = self._featurise(np.concatenate([self._tail, X]))[n_ctx:]
            self._tail = np.concatenate([self._tail, X])[-max(self.lags) :]
        else:
            feats = self._featurise(X)
        return self.model.predict_proba(feats)[:, 1]


# --------------------------------------------------------------------------
# deep learning
# --------------------------------------------------------------------------

class LSTMAutoencoderDetector(Detector):
    """
    LSTM autoencoder, unsupervised. The deep baseline FROM THE LITERATURE.

    Trained to reconstruct windows of NORMAL data only. At inference,
    reconstruction error becomes the anomaly score: normal patterns
    reconstruct well, anomalous ones do not.

    KNOWN LIMITATION -- read before quoting this model's numbers.

    The scored timestep is part of the encoder's input. The model is asked
    how well it can reproduce something it was already shown, so the score
    measures COMPRESSIBILITY, not predictability. That distinction decides
    whether this detector can see an anomaly at all:

      - level shifts (an attack that moves a sensor off its normal range)
        are less compressible, so this model detects them.
      - anomalies that preserve the marginal distribution and only break the
        temporal correlation are invisible to it. For an AR(0.95) process,
        the deviation of the last timestep from its window mean has variance
        0.904 under normal dynamics and 0.905 under an i.i.d. replacement.
        There is no gap for a reconstruction score to exploit, and measured
        test ROC-AUC on `make_synthetic(temporal=True)` lands at 0.38-0.46 --
        below chance -- at every bottleneck size and training budget tried.

    So this detector failing on temporally-defined attacks is a property of
    the reconstruction objective, not a bug and not evidence about deep
    learning as a family. Do NOT let it stand as the sole deep arm of the
    benchmark: pair it with LSTMForecastDetector, which scores the same
    windows on predictability instead, and report both.
    """

    supervised = False

    def __init__(
        self,
        window: int = WINDOW_SIZE,
        hidden: int = 64,
        epochs: int = 8,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str | None = None,
    ):
        self.window = window
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.name = f"lstm_ae(w={window},h={hidden})"
        self.model = None

    def _build(self, n_features: int):
        import torch
        import torch.nn as nn

        class LSTMAE(nn.Module):
            def __init__(self, n_features, hidden, window):
                super().__init__()
                self.window = window
                self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
                self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
                self.out = nn.Linear(hidden, n_features)

            def forward(self, x):
                _, (h, _) = self.encoder(x)
                # repeat the final hidden state across the window
                z = h[-1].unsqueeze(1).repeat(1, self.window, 1)
                d, _ = self.decoder(z)
                return self.out(d)

        torch.manual_seed(SEED)
        return LSTMAE(n_features, self.hidden, self.window)

    def fit(self, split: Split) -> "LSTMAutoencoderDetector":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self.device = _resolve_device(self.device)

        # Unsupervised: normal training rows only. Windowed per contiguous
        # normal run, NOT by masking first -- masking first splices the rows
        # either side of every removed attack together and teaches the model
        # to reconstruct jumps that never happen in deployment.
        X_win = make_windows_segmented(
            split.X_train, split.normal_mask(), window=self.window
        )

        self.model = self._build(split.n_features).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        loader = DataLoader(
            TensorDataset(torch.tensor(X_win, dtype=torch.float32)),
            batch_size=self.batch_size,
            shuffle=True,  # safe: windows already carry their own time order
        )

        self.model.train()
        for epoch in range(self.epochs):
            total = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(batch), batch)
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            print(f"    epoch {epoch:>3}  loss {total / len(X_win):.5f}")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("call fit() first")
        if len(X) < self.window:
            raise ValueError(
                f"{self.name}: got {len(X)} rows, need at least {self.window} "
                "to form one window. Windowed and pointwise models must be "
                "scored on the same rows, so silently returning a shorter "
                "vector here would break the comparison."
            )

        X_win = make_windows(X, window=self.window)
        self.model.eval()
        errors = []
        with torch.no_grad():
            for i in range(0, len(X_win), self.batch_size):
                batch = torch.tensor(
                    X_win[i : i + self.batch_size], dtype=torch.float32
                ).to(self.device)
                recon = self.model(batch)
                # mean squared error over the final timestep of each window,
                # so the score aligns with the label at that same index
                err = ((recon[:, -1, :] - batch[:, -1, :]) ** 2).mean(dim=1)
                errors.append(err.cpu().numpy())

        return unwindow_scores(np.concatenate(errors), len(X), window=self.window)


class LSTMForecastDetector(Detector):
    """
    LSTM one-step forecaster, unsupervised. The deep arm that can actually
    represent temporal dependency.

    Same family, same training data, same window as the autoencoder. One
    difference, and it is the whole point: the scored timestep is NOT in the
    model's input. Given x[t-window+1 .. t-1] the model predicts x[t], and
    the squared prediction error is the anomaly score.

    That makes the score a measure of PREDICTABILITY rather than
    compressibility, which is what an attack on process dynamics actually
    disturbs. Normal ICS signals are strongly autocorrelated and cheap to
    predict one step ahead; an attack that breaks the correlation structure
    raises the error even when it leaves the marginal distribution untouched
    -- exactly the case the autoencoder is blind to.

    This is the Malhotra et al. (2016) / Hundman et al. (KDD 2018)
    formulation, and it is the standard deep baseline in the telemetry and
    ICS anomaly detection literature.

    Fair-comparison note: still unsupervised (normal training rows only),
    still WINDOW_SIZE context, still scored at the window's last timestep, so
    its score vector aligns with every other model's against the same y_true.
    It sees one fewer timestep of context than the autoencoder because the
    last one is the target -- that is a handicap, not an advantage.
    """

    supervised = False

    def __init__(
        self,
        window: int = WINDOW_SIZE,
        hidden: int = 64,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str | None = None,
    ):
        self.window = window
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.name = f"lstm_forecast(w={window},h={hidden})"
        self.model = None

    def _build(self, n_features: int):
        import torch
        import torch.nn as nn

        class LSTMForecaster(nn.Module):
            def __init__(self, n_features, hidden):
                super().__init__()
                self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
                self.out = nn.Linear(hidden, n_features)

            def forward(self, x):
                out, _ = self.lstm(x)
                # predict from the final state only: one step ahead
                return self.out(out[:, -1, :])

        torch.manual_seed(SEED)
        return LSTMForecaster(n_features, self.hidden)

    def fit(self, split: Split) -> "LSTMForecastDetector":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self.device = _resolve_device(self.device)

        # Same contiguity rule as the autoencoder: a window that straddles a
        # removed attack would ask the model to predict across a splice.
        X_win = make_windows_segmented(
            split.X_train, split.normal_mask(), window=self.window
        )

        self.model = self._build(split.n_features).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        loader = DataLoader(
            TensorDataset(torch.tensor(X_win, dtype=torch.float32)),
            batch_size=self.batch_size,
            shuffle=True,  # safe: windows already carry their own time order
        )

        self.model.train()
        for epoch in range(self.epochs):
            total = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                opt.zero_grad()
                # context is everything but the last step; target is the last
                pred = self.model(batch[:, :-1, :])
                loss = loss_fn(pred, batch[:, -1, :])
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            print(f"    epoch {epoch:>3}  loss {total / len(X_win):.5f}")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("call fit() first")
        if len(X) < self.window:
            raise ValueError(
                f"{self.name}: got {len(X)} rows, need at least {self.window}."
            )

        X_win = make_windows(X, window=self.window)
        self.model.eval()
        errors = []
        with torch.no_grad():
            for i in range(0, len(X_win), self.batch_size):
                batch = torch.tensor(
                    X_win[i : i + self.batch_size], dtype=torch.float32
                ).to(self.device)
                pred = self.model(batch[:, :-1, :])
                # error on the window's final timestep, so the score lands on
                # the same index make_windows assigns the window's label
                err = ((pred - batch[:, -1, :]) ** 2).mean(dim=1)
                errors.append(err.cpu().numpy())

        return unwindow_scores(np.concatenate(errors), len(X), window=self.window)


# --------------------------------------------------------------------------
# trivial baselines -- always run these
# --------------------------------------------------------------------------

class RandomDetector(Detector):
    """Pure noise. If this scores well under point-adjust, that is the
    project's headline finding, not a bug."""

    name = "random"

    def __init__(self, seed: int = SEED):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def fit(self, split: Split) -> "RandomDetector":
        # Reset the stream, so a full fit -> score(val) -> score(test) cycle
        # gives the same numbers every run no matter how the instance was
        # used before. This baseline IS the headline finding; it has to be
        # exactly reproducible.
        self.rng = np.random.default_rng(self.seed)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # Draw from the ONGOING stream rather than a freshly seeded one. A
        # fresh generator per call would hand identical scores to validation
        # and test whenever the two splits happen to be the same length, and
        # the threshold picked on validation would then be a threshold picked
        # on test -- commitment 4, broken silently and invisibly.
        return self.rng.random(len(X))


# Every entry is a zero-argument factory, so experiment.py can build any
# detector by name without knowing its constructor.
REGISTRY = {
    "random": RandomDetector,
    "xgboost": GradientBoostingDetector,
    # The lags-vs-no-lags contrast is the single most informative experiment
    # in the project, so it needs to be runnable from the command line rather
    # than only by editing code.
    "xgboost_lags": partial(GradientBoostingDetector, lags=[1, 5, 30]),
    # Both deep detectors, deliberately. The autoencoder is what the
    # literature reports; the forecaster is what can see temporal structure.
    # Running only the first would misattribute a scoring artefact to the
    # deep-learning family as a whole.
    "lstm_ae": LSTMAutoencoderDetector,
    "lstm_forecast": LSTMForecastDetector,
}

DEEP_MODELS = ("lstm_ae", "lstm_forecast")
BOOSTING_MODELS = ("xgboost", "xgboost_lags")
