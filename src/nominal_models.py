"""
Detectors for the A1 -> A2 primary protocol. None of them sees an attack label.

    det = NOMINAL_REGISTRY["xgb_cusum"](seed=42).fit(split)   # split: data.NominalSplit
    scores = det.score(X)                                     # one score per row, higher = worse

Two axes, reported for every detector (`objective`, `family`):

  objective   predictive    supervised PROCESS PREDICTION:
                            a regressor learns sensor readings from actuator
                            states on nominal A1 data; the anomaly score is
                            CUSUM over standardised prediction residuals.
                            "Supervised" refers to input->target pairs, NOT
                            to attack labels.
              unsupervised  a score learnt from nominal data alone.
  family      classical / deep.

This separation follows the study protocol: "supervised" and "deep" are
different properties of a model, and the attack-label classifiers in
models.py (xgboost, cnn_classifier, ...) are an attack-label-trained
EXTENSION, never pooled with these.

Everything is fit on split.X_fit; anything chosen from data (early stopping,
the actuator->sensor mapping, residual statistics) uses split.X_val. Neither
split.X_cal (thresholds only) nor A2 is read here. `score()` is stateless
across calls: CUSUM and windowed models restart for every block they are
given, so no history crosses a block boundary.

Model families follow published SWaT comparisons: linear, random-forest,
k-NN and XGBoost regression (+ a matched LSTM predictor); PCA-SPE, Isolation
Forest, K-means, Elliptic Envelope, LOF, DBSCAN, One-Class SVM, a dense AE and the
LSTM autoencoder. Not implemented: DAGMM, USAD, OmniAnomaly, TranAD, aCVAE --
no reproducible settings are in the material this project holds.

`papers` tags, in full:
  Yu et al. 2026: R. Yu, M. Wang, J. Yun, J. Du, L. Wang, and Y. Suo, "Unified
                  reproduction and event-level evaluation of industrial
                  multivariate time-series anomaly detection methods," in Proc.
                  7th Int. Conf. Comput., Netw. Internet Things (CNIoT), 2026,
                  pp. 1-5, doi: 10.1109/CNIoT69547.2026.11582255.
  Heydari & Nyarko 2026: V. Heydari and K. Nyarko, "Alarm-budgeted event-level
                         evaluation of ICS anomaly detection: Lessons from SWaT
                         and WADI," IEEE Access, vol. 14, pp. 17585-17609,
                         2026, doi: 10.1109/ACCESS.2026.3659034.
"""

from __future__ import annotations

import numpy as np

from config import CUSUM, WINDOW_SIZE
from data import NominalSplit, stage_of, unwindow_scores, window_ends


def _subsample(X: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Evenly strided subsample: keeps the time coverage of A1-fit and is
    deterministic. Used where a method is O(n^2) or worse in training rows,
    and always recorded in describe()."""
    if len(X) <= n:
        return X
    rng = np.random.default_rng(seed)
    offset = int(rng.integers(0, len(X) // n))
    return X[offset :: len(X) // n][:n]


class NominalDetector:
    name = "nominal"
    objective = "unsupervised"
    family = "classical"
    papers: tuple[str, ...] = ()

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.info: dict = {}

    def fit(self, split: NominalSplit) -> "NominalDetector":
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "objective": self.objective, "family": self.family,
                "papers": list(self.papers), "seed": self.seed, **self.info}


# --------------------------------------------------------------------------
# unsupervised, classical
# --------------------------------------------------------------------------

class _Unsupervised(NominalDetector):
    """Optional PCA front end (minmax_pca profile), fit on A1-fit only."""

    def __init__(self, seed: int = 42, pca: float | None = None):
        super().__init__(seed)
        self.pca_var = pca
        self._pca = None

    def _front(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if self.pca_var is None:
            return X
        if fit:
            from sklearn.decomposition import PCA

            self._pca = PCA(n_components=self.pca_var, random_state=self.seed).fit(X)
            self.info["pca_components"] = int(self._pca.n_components_)
        return self._pca.transform(X)

    def fit(self, split):
        self._fit(self._front(split.X_fit, fit=True))
        return self

    def score(self, X):
        return self._score(self._front(X))


class PCASPE(_Unsupervised):
    """Squared prediction error (Q statistic) of a PCA model of normal data.
    Components kept: enough for `var` of the fit-row variance."""

    name, papers = "pca_spe", ("Yu et al. 2026",)

    def __init__(self, seed=42, pca=None, var: float = 0.95):
        super().__init__(seed, pca)
        self.var = var

    def _fit(self, X):
        from sklearn.decomposition import PCA

        self.model = PCA(n_components=self.var, random_state=self.seed).fit(X)
        self.info["spe_components"] = int(self.model.n_components_)

    def _score(self, X):
        recon = self.model.inverse_transform(self.model.transform(X))
        return ((X - recon) ** 2).sum(axis=1)


class IsolationForestDet(_Unsupervised):
    name, papers = "iforest", ("Yu et al. 2026",)

    def _fit(self, X):
        from sklearn.ensemble import IsolationForest

        self.model = IsolationForest(n_estimators=200, random_state=self.seed, n_jobs=-1).fit(X)

    def _score(self, X):
        return -self.model.score_samples(X)


class KMeansDet(_Unsupervised):
    """Distance to the nearest of k centroids fit on normal data. k = 8 is our
    choice, recorded."""

    name = "kmeans"

    def __init__(self, seed=42, pca=None, k: int = 8):
        super().__init__(seed, pca)
        self.k = k

    def _fit(self, X):
        from sklearn.cluster import KMeans

        self.model = KMeans(n_clusters=self.k, n_init=4, random_state=self.seed).fit(
            _subsample(X, 100_000, self.seed))
        self.info.update(k=self.k, fit_subsample=min(len(X), 100_000))

    def _score(self, X):
        return self.model.transform(X).min(axis=1)


class EllipticEnvelopeDet(_Unsupervised):
    """
    Robust (minimum covariance determinant) Mahalanobis distance. MCD is
    O(n p^2) per iteration with many restarts; fit on a 20k subsample.

    SWaT's columns are nearly collinear (P101 / FIT201 / MV201 / P203 move
    together; actuators are discrete), so the raw covariance is singular --
    condition number ~1e39 -- and its Mahalanobis distances come back NaN.
    The covariance is therefore estimated in the PCA subspace holding 99.9%
    of fit-row variance. That is still not enough: MCD keeps the most
    concentrated ~60% of rows, inside which discrete actuator directions
    barely vary (condition number ~1e16). A ridge eps*I is added to the
    robust covariance, eps = r * (mean fit-row variance), with r the
    smallest of 1e-6, 1e-5, ... that brings the condition number to <= 1e4.
    The rule is numerical and uses no labels (on SWaT it picks r = 1e-3).
    """

    name = "elliptic_envelope"

    def _fit(self, X):
        from sklearn.covariance import MinCovDet
        from sklearn.decomposition import PCA

        Xs = _subsample(X, 20_000, self.seed)
        self.sub = PCA(n_components=0.999, random_state=self.seed).fit(Xs)
        Z = self.sub.transform(Xs)
        mcd = MinCovDet(random_state=self.seed).fit(Z)
        scale = Z.var(axis=0).mean()
        for r in 10.0 ** np.arange(-6, 1):
            cov = mcd.covariance_ + r * scale * np.eye(Z.shape[1])
            if np.linalg.cond(cov) <= 1e4:
                break
        self.loc, self.prec = mcd.location_, np.linalg.inv(cov)
        self.info.update(fit_subsample=len(Xs), mcd_subspace_dims=int(Z.shape[1]),
                         ridge_ratio=float(r), cond=float(np.linalg.cond(cov)))

    def _score(self, X):
        d = self.sub.transform(X) - self.loc
        return np.einsum("ij,jk,ik->i", d, self.prec, d)


class LOFDet(_Unsupervised):
    name = "lof"

    def _fit(self, X):
        from sklearn.neighbors import LocalOutlierFactor

        self.model = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1).fit(
            _subsample(X, 20_000, self.seed))
        self.info["fit_subsample"] = min(len(X), 20_000)

    def _score(self, X):
        return -self.model.score_samples(X)


class DBSCANDet(_Unsupervised):
    """
    DBSCAN has no predict(); the standard novelty adaptation is used: cluster
    a fit subsample, keep its core samples, and score a row by its distance
    to the nearest core sample (0..eps = inside a dense normal region).
    eps = 99th percentile of the min_samples-th neighbour distance on the
    fit subsample -- our choice, recorded.
    """

    name = "dbscan"

    def _fit(self, X):
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors

        Xs = _subsample(X, 10_000, self.seed)
        k = 10
        d, _ = NearestNeighbors(n_neighbors=k).fit(Xs).kneighbors(Xs)
        eps = float(np.quantile(d[:, -1], 0.99)) or 1e-6
        db = DBSCAN(eps=eps, min_samples=k).fit(Xs)
        core = Xs[db.core_sample_indices_] if len(db.core_sample_indices_) else Xs
        self.nn = NearestNeighbors(n_neighbors=1).fit(core)
        self.info.update(eps=eps, min_samples=k, n_core=int(len(core)), fit_subsample=len(Xs))

    def _score(self, X):
        return self.nn.kneighbors(X)[0][:, 0]


class OneClassSVMDet(_Unsupervised):
    """RBF one-class SVM; libsvm is O(n^2)-O(n^3), so fit on 10k rows.
    nu = 0.01 (expected outlier fraction in normal data) is our choice."""

    name = "ocsvm"

    def _fit(self, X):
        from sklearn.svm import OneClassSVM

        self.model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01).fit(
            _subsample(X, 10_000, self.seed))
        self.info.update(nu=0.01, fit_subsample=min(len(X), 10_000))

    def _score(self, X):
        return -self.model.decision_function(X)


# --------------------------------------------------------------------------
# unsupervised, deep
# --------------------------------------------------------------------------

class DenseAEDet(NominalDetector):
    """Feed-forward autoencoder on single rows (Yu et al. 2026's AE baseline family):
    n -> 32 -> 16 -> 32 -> n, MSE, Adam, early stopping on A1-validation."""

    name, family, papers = "dense_ae", "deep", ("Yu et al. 2026",)

    def __init__(self, seed=42, epochs: int = 30, patience: int = 4, device=None):
        super().__init__(seed)
        self.epochs, self.patience, self.device = epochs, patience, device

    def fit(self, split):
        import torch
        import torch.nn as nn

        from models import _resolve_device

        torch.manual_seed(self.seed)
        self.device = _resolve_device(self.device)
        n = split.n_features
        self.model = nn.Sequential(
            nn.Linear(n, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, n),
        ).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        Xf = torch.tensor(split.X_fit, dtype=torch.float32, device=self.device)
        Xv = torch.tensor(split.X_val, dtype=torch.float32, device=self.device)
        rng = np.random.default_rng(self.seed)
        best, best_state, bad = np.inf, None, 0
        self.loss_history, self.val_loss_history = [], []
        for epoch in range(self.epochs):
            self.model.train()
            order = torch.as_tensor(rng.permutation(len(Xf)), device=self.device)
            total = 0.0
            for i in range(0, len(order), 1024):
                b = Xf[order[i : i + 1024]]
                opt.zero_grad()
                loss = ((self.model(b) - b) ** 2).mean()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            self.loss_history.append(total / len(Xf))
            v = float(self._mse(Xv).mean())
            self.val_loss_history.append(v)
            if v < best:
                best, bad, self.best_epoch = v, 0, epoch
                best_state = {k: t.detach().clone() for k, t in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        self.info.update(best_epoch=self.best_epoch, epochs_run=len(self.loss_history))
        return self

    def _mse(self, Xt):
        import torch

        self.model.eval()
        with torch.no_grad():
            return torch.cat([((self.model(Xt[i : i + 8192]) - Xt[i : i + 8192]) ** 2).mean(dim=1)
                              for i in range(0, len(Xt), 8192)]).cpu().numpy()

    def score(self, X):
        import torch

        return self._mse(torch.tensor(X, dtype=torch.float32, device=self.device))


class LSTMAEDet(NominalDetector):
    """The existing models.LSTMAutoencoderDetector, trained on A1-fit and
    early-stopped on A1-validation through NominalSplit.as_split()."""

    name, family, papers = "lstm_ae", "deep", ("Yu et al. 2026", "Heydari & Nyarko 2026")

    def __init__(self, seed=42, train_stride: int = 5, epochs: int = 8):
        super().__init__(seed)
        self.kw = {"train_stride": train_stride, "epochs": epochs}

    def fit(self, split):
        import torch

        from models import LSTMAutoencoderDetector

        torch.manual_seed(self.seed)
        self.det = LSTMAutoencoderDetector(**self.kw).fit(split.as_split())
        self.info.update(self.kw, best_epoch=getattr(self.det, "best_epoch", None))
        return self

    def score(self, X):
        return self.det.score(X)


# --------------------------------------------------------------------------
# supervised process prediction
# --------------------------------------------------------------------------

def select_mapping(split: NominalSplit, min_r2: float = 0.5) -> dict:
    """
    For every sensor, find the actuator inputs that predict it best with linear regression, and keep only sensors that the
    actuators actually explain.

    Candidates per sensor: the actuators of its own stage, and all actuators.
    The candidate with the lower A1-VALIDATION MSE wins (fit-row R^2 always
    favours the larger set; validation R^2 is meaningless for sensors that
    barely move in the validation window -- AIT401 is constant there).
    R^2 on fit rows decides inclusion: a sensor with R^2 < min_r2 is not
    explained by the actuators, its residual is just the sensor's own
    variation, and it is left to the unsupervised methods. On SWaT these are mostly the
    chemical analysers -- the same sensors whose normal level drifts by up to
    10 sd between A1 and A2. min_r2 = 0.5 is our choice, recorded.
    Returns {sensor: {...}} for every sensor, with an `included` flag.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    names = split.feature_names
    sensors, actuators = split.columns("sensor"), split.columns("actuator")
    mapping = {}
    for s in sensors:
        cands = {"all": actuators}
        same = [a for a in actuators if stage_of(names[a]) == stage_of(names[s])]
        if same:
            cands["stage"] = same
        best = None
        for key, inputs in cands.items():
            lr = LinearRegression().fit(split.X_fit[:, inputs], split.X_fit[:, s])
            fit_r2 = float(r2_score(split.X_fit[:, s], lr.predict(split.X_fit[:, inputs])))
            val_mse = float(mean_squared_error(split.X_val[:, s], lr.predict(split.X_val[:, inputs])))
            if best is None or val_mse < best["val_mse"]:
                best = {"inputs": inputs, "input_set": key, "fit_r2": fit_r2, "val_mse": val_mse,
                        "input_names": [names[a] for a in inputs]}
        mapping[names[s]] = {**best, "target": s, "included": best["fit_r2"] >= min_r2}
    return mapping


def cusum_score(z: np.ndarray, drift: float) -> np.ndarray:
    """
    Two-sided CUSUM per column of standardised residuals z, reset at the
    start of every call (block-safe), max over columns:
        S+_t = max(0, S+_{t-1} + z_t - b),  S-_t = max(0, S-_{t-1} - z_t - b)
    Target = 0 (residuals are already centred on A1 statistics), allowance
    b = `drift`.
    """
    s_hi = np.zeros(z.shape[1])
    s_lo = np.zeros(z.shape[1])
    out = np.empty(len(z))
    for t in range(len(z)):
        s_hi = np.maximum(0.0, s_hi + z[t] - drift)
        s_lo = np.maximum(0.0, s_lo - z[t] - drift)
        out[t] = max(s_hi.max(), s_lo.max())
    return out


class _ProcessPredictor(NominalDetector):
    """
    Residual pipeline shared by the classical regressors and the LSTM
    predictor: predict sensors -> residuals -> standardise with the residual
    mean/std on A1-VALIDATION (fit-row residuals are optimistic for RF/k-NN)
    -> score. score_rule = "cusum" (CUSUM over the residuals) or "max_z"
    (anomalous if any sensor's residual exceeds its own threshold).
    """

    objective = "predictive"

    def __init__(self, seed=42, score_rule: str = "cusum", drift: float = CUSUM["drift"]):
        super().__init__(seed)
        self.score_rule, self.drift = score_rule, drift

    def _finish_fit(self, split):
        r = split.X_val[:, self.targets] - self.predict(split.X_val)
        self.mu, self.sd = r.mean(axis=0), r.std(axis=0)
        # floor in scaled units: a target that is near-constant on validation
        # would otherwise turn any later movement into an enormous z
        self.sd = np.maximum(self.sd, 0.01)
        self.info.update(score_rule=self.score_rule, cusum_drift=self.drift,
                         residual_stats_from="A1-val", n_targets=len(self.targets))
        return self

    def residual_z(self, X):
        return (X[:, self.targets] - self.predict(X) - self.mu) / self.sd

    def score(self, X):
        z = self.residual_z(X)
        if self.score_rule == "max_z":
            return np.abs(z).max(axis=1)
        return cusum_score(z, self.drift)


class RegressionPredictor(_ProcessPredictor):
    """Actuator -> sensor regression with the per-sensor input mapping of select_mapping.
    Sensors sharing an input set share one multi-output regressor."""

    family = "classical"
    BASES = ("linear", "rf", "knn", "xgb")

    def __init__(self, base: str = "linear", seed=42, score_rule="cusum", min_r2: float = 0.5, **kw):
        super().__init__(seed, score_rule, **kw)
        self.base, self.min_r2 = base, min_r2
        self.name = f"{base}_{score_rule}"

    def _make(self):
        if self.base == "linear":
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        if self.base == "rf":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5,
                                         max_samples=0.3, random_state=self.seed, n_jobs=-1)
        if self.base == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            return KNeighborsRegressor(n_neighbors=10, n_jobs=-1)
        if self.base == "xgb":
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                subsample=0.8, random_state=self.seed, n_jobs=-1,
                                tree_method="hist")
        raise ValueError(self.base)

    def fit(self, split):
        self.mapping = select_mapping(split, self.min_r2)
        groups: dict[tuple, list[int]] = {}
        for m in self.mapping.values():
            if m["included"]:
                groups.setdefault(tuple(m["inputs"]), []).append(m["target"])
        self.targets = [t for ts in groups.values() for t in ts]
        Xf = split.X_fit
        if self.base == "knn":  # O(n) per query; inputs are discrete states anyway
            Xf = _subsample(Xf, 50_000, self.seed)
            self.info["fit_subsample"] = len(Xf)
        self.models = []
        for inputs, targets in groups.items():
            m = self._make().fit(Xf[:, list(inputs)], Xf[:, targets])
            self.models.append((list(inputs), targets, m))
        self.info["mapping"] = {k: {kk: v[kk] for kk in ("input_set", "fit_r2", "val_mse", "included", "input_names")}
                                for k, v in self.mapping.items()}
        self.info["min_r2"] = self.min_r2
        self.info["excluded_targets"] = [k for k, v in self.mapping.items() if not v["included"]]
        return self._finish_fit(split)

    def predict(self, X):
        out = []
        for inputs, targets, m in self.models:
            p = m.predict(X[:, inputs])
            out.append(p.reshape(len(X), len(targets)))
        return np.concatenate(out, axis=1)


class LSTMPredictor(_ProcessPredictor):
    """
    The deep arm of process prediction, MATCHED to the regressors: inputs are
    actuator states only (all of them -- the union of every regressor's
    permitted inputs), targets are the same sensors, scored through the same
    residual / CUSUM pipeline. Context: the last WINDOW_SIZE seconds of
    actuator states, so it can learn the integrating dynamics (a pump state
    changes a tank level over minutes) that a row-wise regressor cannot.
    The first window-1 rows of every scored block have no full window and
    get zero raw residual.
    """

    family = "deep"

    def __init__(self, seed=42, score_rule="cusum", window: int = WINDOW_SIZE,
                 hidden: int = 64, epochs: int = 15, train_stride: int = 5,
                 patience: int = 3, device=None, min_r2: float = 0.5, **kw):
        super().__init__(seed, score_rule, **kw)
        self.min_r2 = min_r2
        self.window, self.hidden, self.epochs = window, hidden, epochs
        self.train_stride, self.patience, self.device = train_stride, patience, device
        self.name = f"lstm_{score_rule}"

    def fit(self, split):
        import torch
        import torch.nn as nn

        from models import _gather, _resolve_device

        torch.manual_seed(self.seed)
        self.device = dev = _resolve_device(self.device)
        # same targets as the regressors (select_mapping's inclusion rule);
        # inputs: every actuator, the union of the regressors' input sets
        self.inputs = split.columns("actuator")
        self.targets = [m["target"] for m in select_mapping(split, self.min_r2).values() if m["included"]]
        n_in, n_out = len(self.inputs), len(self.targets)

        class Net(nn.Module):
            def __init__(self, hidden):
                super().__init__()
                self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
                self.out = nn.Linear(hidden, n_out)

            def forward(self, x):
                o, _ = self.lstm(x)
                return self.out(o[:, -1, :])

        self.model = Net(self.hidden).to(dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        A = lambda X: torch.tensor(X[:, self.inputs], dtype=torch.float32, device=dev)
        S = lambda X: torch.tensor(X[:, self.targets], dtype=torch.float32, device=dev)
        Af, Sf, Av, Sv = A(split.X_fit), S(split.X_fit), A(split.X_val), S(split.X_val)
        ends = window_ends(np.ones(len(Af), bool), self.window, self.train_stride)
        v_ends = torch.as_tensor(window_ends(np.ones(len(Av), bool), self.window, 10), device=dev)
        rng = np.random.default_rng(self.seed)
        best, best_state, bad = np.inf, None, 0
        self.loss_history, self.val_loss_history = [], []
        for epoch in range(self.epochs):
            self.model.train()
            order = torch.as_tensor(rng.permutation(ends), device=dev)
            total = 0.0
            for i in range(0, len(order), 256):
                e = order[i : i + 256]
                opt.zero_grad()
                loss = ((self.model(_gather(Af, e, self.window)) - Sf[e]) ** 2).mean()
                loss.backward()
                opt.step()
                total += loss.item() * len(e)
            self.loss_history.append(total / len(order))
            self.model.eval()
            with torch.no_grad():
                v = float(np.mean([((self.model(_gather(Av, v_ends[i : i + 4096], self.window))
                                     - Sv[v_ends[i : i + 4096]]) ** 2).mean().item()
                                   for i in range(0, len(v_ends), 4096)]))
            self.val_loss_history.append(v)
            if v < best:
                best, bad, self.best_epoch = v, 0, epoch
                best_state = {k: t.detach().clone() for k, t in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        self.info.update(window=self.window, hidden=self.hidden, best_epoch=self.best_epoch,
                         inputs="all actuators", train_stride=self.train_stride)
        return self._finish_fit(split)

    def predict(self, X):
        import torch

        from models import _gather

        dev = self.device
        At = torch.tensor(X[:, self.inputs], dtype=torch.float32, device=dev)
        pred = X[:, self.targets].copy()  # rows without a full window: residual 0
        ends = torch.arange(self.window - 1, len(X), device=dev)
        self.model.eval()
        with torch.no_grad():
            out = [self.model(_gather(At, ends[i : i + 4096], self.window)).cpu().numpy()
                   for i in range(0, len(ends), 4096)]
        if out:
            pred[self.window - 1 :] = np.concatenate(out)
        return pred


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def _reg(base, rule="cusum"):
    return lambda seed=42, **kw: RegressionPredictor(base, seed=seed, score_rule=rule, **kw)


NOMINAL_REGISTRY = {
    # supervised process prediction -- classical
    "linear_cusum": _reg("linear"),
    "rf_cusum": _reg("rf"),
    "knn_cusum": _reg("knn"),
    "xgb_cusum": _reg("xgb"),
    "linear_maxz": _reg("linear", "max_z"),
    "xgb_maxz": _reg("xgb", "max_z"),
    # supervised process prediction -- deep, matched inputs/targets
    "lstm_cusum": lambda seed=42, **kw: LSTMPredictor(seed=seed, **kw),
    # unsupervised -- classical
    "pca_spe": PCASPE,
    "iforest": IsolationForestDet,
    "kmeans": KMeansDet,
    "elliptic_envelope": EllipticEnvelopeDet,
    "lof": LOFDet,
    "dbscan": DBSCANDet,
    "ocsvm": OneClassSVMDet,
    # unsupervised -- deep
    "dense_ae": DenseAEDet,
    "lstm_ae": LSTMAEDet,
}

UNSUPERVISED_CLASSICAL = ("pca_spe", "iforest", "kmeans", "elliptic_envelope", "lof", "dbscan", "ocsvm")
