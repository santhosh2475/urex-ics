"""
Detectors.

Every model exposes the SAME interface:

    detector.fit(split)          # trains on split's training window only
    scores = detector.score(X)   # higher score = more anomalous

That uniformity is the point. If the two families needed different calling
conventions, the experiment grid would fill with special cases and the
comparison would drift out of alignment without anyone noticing.

Note the asymmetry the interface has to absorb: gradient boosting is
SUPERVISED (it needs labels) while the autoencoder is UNSUPERVISED (it trains
on normal data only). Both draw from the same training window, so neither
sees data the other could not have seen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from config import SEED, WINDOW_SIZE
from data import Split, make_windows, unwindow_scores


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
            **params,
        }
        self.model = None

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
        return pd.concat(parts, axis=1).bfill().to_numpy(dtype=float)

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
        self.model = XGBClassifier(
            scale_pos_weight=n_neg / max(n_pos, 1), **self.params
        )
        self.model.fit(self._featurise(split.X_train), split.y_train, verbose=False)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() first")
        return self.model.predict_proba(self._featurise(X))[:, 1]


# --------------------------------------------------------------------------
# deep learning
# --------------------------------------------------------------------------

class LSTMAutoencoderDetector(Detector):
    """
    LSTM autoencoder, unsupervised. The deep baseline.

    Trained to reconstruct windows of NORMAL data only. At inference,
    reconstruction error becomes the anomaly score: normal patterns
    reconstruct well, anomalous ones do not.
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

        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        X_normal = split.normal_only()  # unsupervised: normal data only
        X_win = make_windows(X_normal, window=self.window)

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
            if epoch % 1 == 0 or epoch == self.epochs - 1:
                print(f"    epoch {epoch:>3}  loss {total / len(X_win):.5f}")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("call fit() first")

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


# --------------------------------------------------------------------------
# trivial baselines -- always run these
# --------------------------------------------------------------------------

class RandomDetector(Detector):
    """Pure noise. If this scores well under point-adjust, that is the
    project's headline finding, not a bug."""

    name = "random"

    def __init__(self, seed: int = SEED):
        self.rng = np.random.default_rng(seed)

    def fit(self, split: Split) -> "RandomDetector":
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.rng.random(len(X))


REGISTRY = {
    "random": RandomDetector,
    "xgboost": GradientBoostingDetector,
    "lstm_ae": LSTMAutoencoderDetector,
}
