"""
A1 -> A2 detection baseline on SWaT 2015: supervised PROCESS PREDICTION vs
UNSUPERVISED scoring, classical models, pointwise metrics.

    python src/baseline.py               # whole pipeline -> results/baseline/

Every number this module reports is computed from our own copies of the
iTrust files (SWaT_Dataset_Normal_v0, SWaT_Dataset_Attack_v0); the file each
stream was read from is recorded in the results. notebooks/baseline.ipynb
shows them.

Pipeline:

  1. data             TRAIN = the whole normal file (from 16:00:00, no start-up
                      trim: the start-up hours are the only period in which
                      the stage-5 pumps switch, so trimming them leaves the
                      stage-5 regressions nothing to learn). TEST = the attack
                      file. Rows with missing values dropped (none).
  2. scaling          min-max, fit on TRAIN, applied to TEST; all 51 columns.
  3. features         unsupervised models: PCA keeping 99% of TRAIN variance.
                      Supervised models: per sensor, the actuator input set
                      (a single actuator, or the sensor's stage's actuators
                      together) with the highest linear-regression R^2 on TRAIN.
  4. models           supervised, no attack labels. Process prediction
                      (actuators -> sensors): Linear, RF, KNN, XGBoost on the
                      mapped actuator states; XGBoost+lags on the same states
                      plus lagged / rolling-mean / differenced copies (the
                      feature set of models.xgboost_lags); 1D-CNN (the
                      architecture of models.cnn_classifier, regression head)
                      on the last 60 s of every actuator. State prediction
                      (sensors -> actuators): LogisticR, one logistic
                      regression per actuator on every kept sensor.
                      Unsupervised: iForest, K-means, Elliptic Envelope, LOF,
                      DBSCAN, One-Class SVM on the PCA features; LSTM-AE and
                      LSTM-forecast (models.lstm_ae / lstm_forecast) on the
                      last 60 s of the min-max columns, fit on the first 80%
                      of TRAIN with early stopping on the last 20%.
  5. decisions        process prediction: per-sensor control limits on a
                      residual statistic (UCL / LCL = TRAIN extremes); alarm
                      if any sensor is outside. LogisticR and unsupervised:
                      mu + 3 sigma of TRAIN scores (LogisticR's score: 1 -
                      probability of the reported state, standardised per
                      actuator on TRAIN, max over actuators).
  6. metrics          pointwise precision, recall, F1, UDR = FN/(FN+TP),
                      FAR = FP/(FP+TN), plus threshold-free AP and ROC-AUC.
  7. per attack       recall on eight attacks located by their documented action.
  8. diagnostics      data audit; per-sensor prediction variance, residual
                      shift and false-alarm counts; an ablation excluding
                      sensors by rules decided on the normal file only
                      (R1: mapping R^2 < 0.5; R2: daily normal level drifts by
                      > 10% of range), unsupervised models and LogisticR refit
                      and thresholds recalibrated for each feature set.
  9. attack labels    SEPARATE protocol and table (run_attack_label): the
                      attack-label classifiers of models.py (xgboost_lags,
                      logistic_regression, cnn_classifier) need attack rows to
                      train on, and TRAIN above has none. They run on the
                      episode-stratified split (data.stratified_episode_split)
                      with experiment.py's threshold, the 97th percentile of
                      validation normal-row scores. Not comparable row for row
                      with steps 1-8.

Design choices (DEFAULTS, stored in the results file):

  * hyperparameters: scikit-learn / xgboost defaults. LOF, DBSCAN and
    One-Class SVM (O(n^2) or worse) are fit on evenly strided subsamples.
  * residual statistic: "pointwise" (primary) tests each sample's residual on
    its own, SH_i = max(0, x_{i-1} - T - b); "accumulating" is the textbook
    CUSUM, SH_i = max(0, SH_{i-1} + x_i - T - b), reported alongside. On this
    data the accumulating form flags 99% of normal test rows: any shift in a
    residual's mean between the two files larger than b adds up without
    bound. T = mean TRAIN residual, b = 0.5 sd of the TRAIN residual.
  * mu and sigma for the unsupervised threshold come from TRAIN scores;
    taking them from the unlabelled TEST scores is reported alongside.
  * Elliptic Envelope: SWaT's covariance is singular (cond ~1e39); a ridge is
    added by a label-free conditioning rule.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS, THRESHOLD_PERCENTILE, WINDOW_SIZE
from data import contiguous_runs, is_actuator, is_sensor, load_swat_streams, stage_of

OUT = RESULTS / "baseline"
FIG = OUT / "figures"

DEFAULTS = {
    "data_profile": "full_normal",
    "seed": 42,
    "pca_variance": 0.99,
    "k_sigma": 3.0,
    "residual_allowance_sd": 0.5,
    "residual_mode": "pointwise",         # primary; "accumulating" reported alongside
    "subsample": {"lof": 20_000, "dbscan": 10_000, "ocsvm": 10_000},
    "threshold_stats": "train",           # primary; "test" reported alongside
    # ablation rules, decided on the normal (training) file only
    "mapping_r2_rule": 0.5,               # R1: sensor not explained by its actuators
    "drift_rule": 0.10,                   # R2: daily normal level moves > 10% of its range
    "lags": [1, 5, 30],                   # XGBoost+lags: the lag set of models.xgboost_lags
    "cnn": {"window": WINDOW_SIZE, "filters": 64, "epochs": 15, "batch_size": 256, "lr": 1e-3,
            "train_stride": 5, "patience": 3, "holdout": 0.2},
    # LSTM-AE / LSTM-forecast: models.py defaults plus the stride override cv.py uses
    "lstm": {"train_stride": 5, "holdout": 0.2},
}

PROCESS_PREDICTORS = ("LinearR", "RF", "KNN", "XGBoost", "XGBoost+lags", "1D-CNN")   # actuators -> sensors
SUPERVISED = PROCESS_PREDICTORS + ("LogisticR",)                                      # + sensors -> actuator state
DEEP_UNSUPERVISED = {"LSTM-AE": "lstm_ae", "LSTM-forecast": "lstm_forecast"}    # name -> models.REGISTRY key
UNSUPERVISED = ("iForest", "K-Means", "EllipticEnvelope", "LOF", "DBSCAN", "One Class SVM",
                *DEEP_UNSUPERVISED)

# Documented actions of eight SWaT 2015 attacks (iTrust attack list), used
# to locate them among the labelled intervals of the attack file: a spoofed
# sensor reads its set value, a forced actuator switches to its set state
# (pumps / UV: 1 off, 2 on; valves: 1 closed, 2 open). Each attack's category
# (attack_type) is derived from its action.
ATTACK_ACTIONS = {
    "AIT-202 set to 6":                     {"AIT202": 6.0},
    "MV-304 closed":                        {"MV304": 1},
    "P-203, P-205 off":                     {"P203": 1, "P205": 1},
    "P-501 closed, FIT-502 set to 1.29":    {"P501": 1, "FIT502": 1.29},
    "P-101 on, LIT-301 set to 801":         {"P101": 2, "LIT301": 801.0},
    "AIT-402, AIT-502 set to 260":          {"AIT402": 260.0, "AIT502": 260.0},
    "UV-401 off, AIT-502 150, P-501 on":    {"UV401": 1, "AIT502": 150.0, "P501": 2},
    "P-101, MV-101 on, LIT-101 set to 700": {"P101": 2, "MV101": 2, "LIT101": 700.0},
}


def attack_type(action: dict) -> str:
    """
    SWaT attack category [1, Sect. 3] from the points (sensors / actuators)
    an attack acts on and the process stages they belong to:

      single stage, single point   exactly one point
      single stage, multi point    two or more points, all in one stage
      multi stage, single point    one point in each of two or more stages
      multi stage, multi point     two or more stages, two or more points in
                                   at least one of them

    The first two are [1]'s definitions. [1] words the two multi-stage ones
    almost identically ("an SSMP attack performed on multiple stages"), so
    those follow the category names. [1, Table 1]: of 36 attacks, 26 single
    stage single point, 4 single stage multi point, 2 multi stage single
    point, 4 multi stage multi point.

    [1] J. Goh, S. Adepu, K. N. Junejo, and A. Mathur, "A dataset to support
        research in the design of secure water treatment systems," in
        Critical Information Infrastructures Security (CRITIS 2016), ser.
        Lecture Notes in Computer Science, vol. 10242. Cham, Switzerland:
        Springer, 2017, pp. 88-99, doi: 10.1007/978-3-319-71368-7_8.
    """
    per_stage = pd.Series([stage_of(tag) for tag in action]).value_counts()
    stages = "single stage" if len(per_stage) == 1 else "multi stage"
    points = "single point" if per_stage.max() == 1 else "multi point"
    return f"{stages}, {points}"


# --------------------------------------------------------------------------
# 1-3. data, scaling, features
# --------------------------------------------------------------------------

def load_data():
    """TRAIN = whole normal file, TEST = attack file. Returns raw frames."""
    a1, a2, report = load_swat_streams(DEFAULTS["data_profile"])
    a1 = a1.dropna().reset_index(drop=True)        # drop rows with missing values
    a2 = a2.dropna().reset_index(drop=True)
    return a1, a2, report


class Preprocess:
    """Min-max on TRAIN; PCA (99% of TRAIN variance) for the unsupervised models."""

    def __init__(self, columns: list[str], pca_variance: float, seed: int):
        self.columns, self.pca_variance, self.seed = columns, pca_variance, seed

    def fit(self, train: pd.DataFrame):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import MinMaxScaler

        self.scaler = MinMaxScaler().fit(train[self.columns].to_numpy(float))
        self.pca = PCA(n_components=self.pca_variance, random_state=self.seed).fit(
            self.minmax(train))
        return self

    def minmax(self, df):
        return self.scaler.transform(df[self.columns].to_numpy(float))

    def pca_features(self, df):
        return self.pca.transform(self.minmax(df))


def select_mapping(X: np.ndarray, columns: list[str]) -> dict:
    """
    Actuator input set per sensor. Candidates: every single actuator, and the
    sensor's stage's actuators together. The candidate with the highest
    linear-regression R^2 on TRAIN (then lowest MSE) is kept.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    act = [i for i, c in enumerate(columns) if is_actuator(c)]
    mapping = {}
    for s, name in enumerate(columns):
        if not is_sensor(name):
            continue
        cands = [[a] for a in act]
        stage = [a for a in act if stage_of(columns[a]) == stage_of(name)]
        if len(stage) > 1:
            cands.append(stage)
        best = None
        for inp in cands:
            if np.ptp(X[:, inp], axis=0).max() == 0:
                continue                                  # a constant input explains nothing
            lr = LinearRegression().fit(X[:, inp], X[:, s])
            p = lr.predict(X[:, inp])
            key = (r2_score(X[:, s], p), -mean_squared_error(X[:, s], p))
            if best is None or key > best[0]:
                best = (key, inp)
        mapping[name] = {"target": s, "inputs": best[1],
                         "input_names": [columns[a] for a in best[1]],
                         "r2": float(best[0][0]), "mse": float(-best[0][1])}
    return mapping


def lag_features(A: np.ndarray, lags: list[int] = DEFAULTS["lags"]) -> np.ndarray:
    """
    Lagged, rolling-mean and differenced copies of every column of A -- the
    feature set of models.xgboost_lags (GradientBoostingDetector._featurise),
    without the raw columns. Blocks are ordered (shift, roll, diff) per lag,
    each block one column per column of A. Backward-looking only: row i uses
    rows <= i.
    """
    from models import GradientBoostingDetector

    return GradientBoostingDetector(lags=list(lags))._featurise(A)[:, A.shape[1]:]


def lagged_mapping(mapping: dict, lagged: list[int], n_cols: int, n_lags: int) -> dict:
    """The mapping with every input actuator joined by its lag features, for a
    matrix [X | lag_features(X[:, lagged])] with X of n_cols columns."""
    pos = {a: k for k, a in enumerate(lagged)}
    blocks = 3 * n_lags
    return {s: {**m, "inputs": list(m["inputs"]) + [n_cols + b * len(lagged) + pos[a]
                                                   for a in m["inputs"] for b in range(blocks)]}
            for s, m in mapping.items()}


# --------------------------------------------------------------------------
# 4-5. models and decisions
# --------------------------------------------------------------------------

def _subsample(X, n, seed):
    if n is None or len(X) <= n:
        return X
    step = len(X) // n
    off = int(np.random.default_rng(seed).integers(0, step))
    return X[off::step][:n]


def residual_statistic(r: np.ndarray, target: np.ndarray, b: np.ndarray, mode: str = "pointwise"):
    """
    Upper / lower residual statistics per sensor (rows x sensors):

    pointwise     SH_i = max(0, x_{i-1} - T - b), SL_i = min(0, x_{i-1} - T + b)
                  -- each sample tested on its own (x_0 = 0).
    accumulating  SH_i = max(0, SH_{i-1} + x_i - T - b), SL likewise -- the
                  textbook CUSUM; never forgets a sustained offset.
    """
    if mode == "pointwise":
        x = np.vstack([np.zeros((1, r.shape[1])), r[:-1]])     # x_{i-1}, x_0 = 0
        return np.maximum(0.0, x - target - b), np.minimum(0.0, x - target + b)
    sh = np.zeros_like(r)
    sl = np.zeros_like(r)
    hi = np.zeros(r.shape[1])
    lo = np.zeros(r.shape[1])
    for t in range(len(r)):
        hi = np.maximum(0.0, hi + r[t] - target - b)
        lo = np.minimum(0.0, lo + r[t] - target + b)
        sh[t], sl[t] = hi, lo
    return sh, sl


class ProcessPredictor:
    """Actuator -> sensor regression + per-sensor control-limit decision."""

    def __init__(self, name: str, mapping: dict, seed: int):
        self.name, self.mapping, self.seed = name, mapping, seed

    def _regressor(self):
        if self.name == "LinearR":
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        if self.name == "RF":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(random_state=self.seed, n_jobs=-1)
        if self.name == "KNN":
            from sklearn.neighbors import KNeighborsRegressor
            return KNeighborsRegressor(n_jobs=-1)
        if self.name in ("XGBoost", "XGBoost+lags"):
            from xgboost import XGBRegressor
            return XGBRegressor(random_state=self.seed, n_jobs=-1)
        raise ValueError(self.name)

    def _fit_models(self, X):
        groups = {}
        for m in self.mapping.values():
            groups.setdefault(tuple(m["inputs"]), []).append(m["target"])
        self.models = [(list(i), t, self._regressor().fit(X[:, list(i)], X[:, t]))
                       for i, t in groups.items()]
        self.targets = [t for _, ts, _ in self.models for t in ts]

    def regressor_params(self) -> dict:
        return {k: str(v) for k, v in self.models[0][2].get_params().items()}

    def fit(self, X):
        self._fit_models(X)
        r = self.residuals(X)
        y = X[:, self.targets]
        self.target_ = r.mean(axis=0)
        self.resid_sd_ = np.maximum(r.std(axis=0), 1e-12)
        self.b_ = DEFAULTS["residual_allowance_sd"] * r.std(axis=0)
        # share of each sensor's training variance the prediction carries:
        # ~0 means the prediction is close to a constant
        self.pred_var_ratio_ = (y - r).var(axis=0) / np.maximum(y.var(axis=0), 1e-12)
        self.limits_ = {}
        for mode in ("pointwise", "accumulating"):
            sh, sl = residual_statistic(r, self.target_, self.b_, mode)
            self.limits_[mode] = (sh.max(axis=0), sl.min(axis=0))   # UCL, LCL from TRAIN
        return self

    def residuals(self, X):
        out = []
        for i, t, m in self.models:
            # inputs are discrete actuator states: a few hundred distinct rows
            # out of ~500k. Predicting each distinct row once is exact (a
            # regressor's output depends only on its input) and makes k-NN
            # feasible on the full training set.
            uniq, inv = np.unique(X[:, i], axis=0, return_inverse=True)
            out.append(m.predict(uniq).reshape(len(uniq), len(t))[inv.ravel()])
        return X[:, self.targets] - np.concatenate(out, axis=1)

    def margins(self, X, mode: str = DEFAULTS["residual_mode"]) -> np.ndarray:
        """(rows x sensors) how far each statistic is outside UCL/LCL, in
        training-residual sd; > 0 is an alarm. The max over sensors is the
        continuous score used for threshold-free AP / ROC-AUC."""
        ucl, lcl = self.limits_[mode]
        sh, sl = residual_statistic(self.residuals(X), self.target_, self.b_, mode)
        return np.maximum(sh - ucl, lcl - sl) / self.resid_sd_

    def exceedances(self, X, mode: str = DEFAULTS["residual_mode"]) -> np.ndarray:
        """(rows x sensors) True where that sensor's statistic is outside UCL/LCL."""
        return self.margins(X, mode) > 0

    def predict(self, X, mode: str = DEFAULTS["residual_mode"]):
        return self.exceedances(X, mode).any(axis=1).astype(int)

    def record(self, columns) -> dict:
        """Everything that defines the decision, per sensor, for the results file."""
        names = [columns[t] for t in self.targets]
        rec = {n: {"target": float(self.target_[j]), "allowance_b": float(self.b_[j]),
                   **{f"{mode}_{lim}": float(self.limits_[mode][k][j])
                      for mode in self.limits_ for k, lim in enumerate(("UCL", "LCL"))}}
               for j, n in enumerate(names)}
        return {"per_sensor": rec, "regressor_params": self.regressor_params(),
                "recurrence": {"pointwise": "SH_i = max(0, x_{i-1} - T - b); SL_i = min(0, x_{i-1} - T + b)",
                               "accumulating": "SH_i = max(0, SH_{i-1} + x_i - T - b); SL_i = min(0, SL_{i-1} + x_i - T + b)"},
                "alarm": "any sensor with SH > UCL or SL < LCL; UCL/LCL = max/min on TRAIN"}


class CNNPredictor(ProcessPredictor):
    """
    Deep process predictor: the 1D-CNN of models.CNNClassifierDetector (three
    Conv1d(64, kernel 3) + BatchNorm + ReLU blocks, global average pooling)
    with a linear regression head in place of the attack logit, MSE loss.
    Input: the last `window` seconds of every actuator in the mapping's inputs
    (run() passes every actuator that moves in TRAIN); output: every mapped
    sensor at the window's last second. Early stopping on the last `holdout`
    share of TRAIN, in time order. Rows without a full window (the first
    window - 1 of a scored array) get zero residual, the convention of
    nominal_models.LSTMPredictor. Same residual statistic and control limits
    as the other process predictors.
    """

    def __init__(self, mapping: dict, seed: int, cfg: dict | None = None):
        super().__init__("1D-CNN", mapping, seed)
        self.cfg = {**DEFAULTS["cnn"], **(cfg or {})}
        self.inputs = sorted({a for m in mapping.values() for a in m["inputs"]})

    def _fit_models(self, X):
        import torch
        import torch.nn as nn

        from models import _gather, _resolve_device

        c, w = self.cfg, self.cfg["window"]
        torch.manual_seed(self.seed)
        self.device = dev = _resolve_device(None)
        self.targets = [m["target"] for m in self.mapping.values()]
        n_in, n_out, f = len(self.inputs), len(self.targets), c["filters"]

        def block(i, o):
            return nn.Sequential(nn.Conv1d(i, o, kernel_size=3, padding="same"), nn.BatchNorm1d(o), nn.ReLU())

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(block(n_in, f), block(f, f), block(f, f))
                self.out = nn.Linear(f, n_out)

            def forward(self, x):                              # (batch, window, n_in)
                return self.out(self.conv(x.transpose(1, 2)).mean(dim=-1))

        net = Net().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=c["lr"])
        A = torch.tensor(X[:, self.inputs], dtype=torch.float32, device=dev)
        S = torch.tensor(X[:, self.targets], dtype=torch.float32, device=dev)
        cut = int(len(X) * (1 - c["holdout"]))
        tr = torch.arange(w - 1, cut, c["train_stride"], device=dev)
        va = torch.arange(cut + w - 1, len(X), 10, device=dev)     # windows wholly inside the hold-out
        rng = np.random.default_rng(self.seed)
        best, state, bad, self.val_mse_history = np.inf, None, 0, []
        for epoch in range(c["epochs"]):
            net.train()
            order = tr[torch.as_tensor(rng.permutation(len(tr)), device=dev)]
            for i in range(0, len(order), c["batch_size"]):
                e = order[i:i + c["batch_size"]]
                opt.zero_grad()
                loss = ((net(_gather(A, e, w)) - S[e]) ** 2).mean()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                sse = sum(((net(_gather(A, va[i:i + 4096], w)) - S[va[i:i + 4096]]) ** 2).sum().item()
                          for i in range(0, len(va), 4096))
            v = sse / (len(va) * n_out)
            self.val_mse_history.append(v)
            if v < best:
                best, bad, self.best_epoch = v, 0, epoch
                state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= c["patience"]:
                    break
        net.load_state_dict(state)
        self.net = net

    def residuals(self, X):
        import torch

        from models import _gather

        w = self.cfg["window"]
        A = torch.tensor(X[:, self.inputs], dtype=torch.float32, device=self.device)
        pred = X[:, self.targets].copy()                       # rows without a full window: zero residual
        ends = torch.arange(w - 1, len(X), device=self.device)
        self.net.eval()
        with torch.no_grad():
            out = [self.net(_gather(A, ends[i:i + 4096], w)).cpu().numpy()
                   for i in range(0, len(ends), 4096)]
        if out:
            pred[w - 1:] = np.concatenate(out)
        return X[:, self.targets] - pred

    def regressor_params(self) -> dict:
        return {**self.cfg, "n_inputs": len(self.inputs), "device": self.device,
                "best_epoch": self.best_epoch, "val_mse_history": self.val_mse_history}


class StatePredictor:
    """
    Sensors -> actuator state, the reverse direction of process prediction.
    One logistic regression (multinomial over the states seen in TRAIN) per
    actuator that moves in TRAIN, on every sensor that is not excluded. Per
    actuator, s = 1 - P(reported state | sensors), a state never seen in TRAIN
    scoring 1, standardised by its TRAIN mean and sd. Row score = max over
    actuators; alarm above mu + 3 sigma of TRAIN row scores, the rule of the
    unsupervised models (a probability, not a residual, so the control-limit
    rule does not apply).

    Two choices were revised on TRAIN evidence alone. Own-stage sensors as
    inputs: P101 feeds stage 2, so stage 1's sensors cannot see its state
    (s > 0.5 on 8.6% of TRAIN rows). Unstandardised s: the max over actuators
    was dominated by the least predictable one, and mu + 3 sigma of it
    exceeded 1, the largest possible score, so the rule could never alarm.
    """

    name = "LogisticR"

    def __init__(self, columns: list[str], exclude: set, seed: int):
        self.columns, self.exclude, self.seed = list(columns), set(exclude), seed

    def fit(self, X, S):
        """X: min-max features; S: raw values of the same columns (actuator states)."""
        from sklearn.linear_model import LogisticRegression

        self.inputs = [i for i, s in enumerate(self.columns) if is_sensor(s) and s not in self.exclude]
        self.models = [(j, self.inputs, LogisticRegression(max_iter=1000).fit(X[:, self.inputs], S[:, j].astype(int)))
                       for j, c in enumerate(self.columns)
                       if is_actuator(c) and len(np.unique(S[:, j])) > 1]
        a = self.actuator_scores(X, S)
        self.mu_, self.sd_ = a.mean(axis=0), np.maximum(a.std(axis=0), 1e-12)
        s = self.score(X, S)
        self.threshold_ = float(s.mean() + DEFAULTS["k_sigma"] * s.std())
        self.n_inputs = len(self.inputs)
        return self

    def actuator_scores(self, X, S) -> np.ndarray:
        """(rows x actuators) 1 - P(reported state | sensors), unstandardised."""
        out = np.ones((len(X), len(self.models)))
        for k, (j, inputs, m) in enumerate(self.models):
            p = m.predict_proba(X[:, inputs])
            state = S[:, j].astype(int)
            col = np.searchsorted(m.classes_, state).clip(max=len(m.classes_) - 1)
            seen = m.classes_[col] == state
            out[seen, k] = 1 - p[np.flatnonzero(seen), col[seen]]
        return out

    def score(self, X, S) -> np.ndarray:
        return ((self.actuator_scores(X, S) - self.mu_) / self.sd_).max(axis=1)

    def predict(self, X, S) -> np.ndarray:
        return (self.score(X, S) > self.threshold_).astype(int)

    def record(self) -> dict:
        return {"inputs": [self.columns[i] for i in self.inputs],
                "per_actuator": {self.columns[j]: {"states": [int(v) for v in m.classes_],
                                                   "train_mean": float(self.mu_[k]), "train_sd": float(self.sd_[k])}
                                 for k, (j, _, m) in enumerate(self.models)},
                "threshold_train_stats": self.threshold_,
                "params": _params(self.models[0][2]) if self.models else {},
                "alarm": "max over actuators of standardised 1 - P(reported state) > mu + 3 sigma of TRAIN"}


class EllipticEnvelope:
    """MCD Mahalanobis distance; ridge by the label-free conditioning rule."""

    def __init__(self, seed):
        self.seed = seed

    def fit(self, Z):
        from sklearn.covariance import MinCovDet

        mcd = MinCovDet(random_state=self.seed).fit(Z)
        scale = Z.var(axis=0).mean()
        for r in 10.0 ** np.arange(-6, 1):
            cov = mcd.covariance_ + r * scale * np.eye(Z.shape[1])
            if np.linalg.cond(cov) <= 1e4:
                break
        self.loc, self.prec, self.ridge = mcd.location_, np.linalg.inv(cov), float(r)
        return self

    def score(self, Z):
        d = Z - self.loc
        return np.einsum("ij,jk,ik->i", d, self.prec, d)


def _params(m) -> dict:
    return {k: str(v) for k, v in m.get_params().items()}


def unsupervised_scorer(name: str, Z_train: np.ndarray, seed: int):
    """Fit one unsupervised model on PCA features of TRAIN; return score(Z),
    higher = more anomalous, and the settings actually used."""
    sub = DEFAULTS["subsample"]
    if name == "iForest":
        from sklearn.ensemble import IsolationForest
        m = IsolationForest(random_state=seed, n_jobs=-1).fit(Z_train)
        return (lambda Z: -m.score_samples(Z)), {"params": _params(m)}
    if name == "K-Means":
        from sklearn.cluster import KMeans
        m = KMeans(n_clusters=8, random_state=seed, n_init="auto").fit(Z_train)
        return (lambda Z: m.transform(Z).min(axis=1)), {"params": _params(m)}
    if name == "EllipticEnvelope":
        m = EllipticEnvelope(seed).fit(Z_train)
        return m.score, {"params": {"estimator": "MinCovDet", "random_state": seed}, "ridge_ratio": m.ridge}
    if name == "LOF":
        from sklearn.neighbors import LocalOutlierFactor
        m = LocalOutlierFactor(novelty=True, n_jobs=-1).fit(_subsample(Z_train, sub["lof"], seed))
        return (lambda Z: -m.score_samples(Z)), {"params": _params(m), "fit_rows": sub["lof"]}
    if name == "DBSCAN":
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
        Zs = _subsample(Z_train, sub["dbscan"], seed)
        db = DBSCAN().fit(Zs)                               # eps 0.5, min_samples 5
        core = Zs[db.core_sample_indices_] if len(db.core_sample_indices_) else Zs
        nn = NearestNeighbors(n_neighbors=1).fit(core)
        # DBSCAN has no predict(): distance to the nearest core sample
        return (lambda Z: nn.kneighbors(Z)[0][:, 0]), {"params": _params(db), "fit_rows": len(Zs),
                                                       "n_core": len(core)}
    if name == "One Class SVM":
        from sklearn.svm import OneClassSVM
        m = OneClassSVM().fit(_subsample(Z_train, sub["ocsvm"], seed))   # rbf, nu 0.5
        return (lambda Z: -m.decision_function(Z)), {"params": _params(m), "fit_rows": sub["ocsvm"]}
    raise ValueError(name)


def deep_scorer(name: str, X_train: np.ndarray):
    """
    Fit one of models.py's unsupervised LSTMs (lstm_ae: reconstruction error
    of a window's last second; lstm_forecast: one-step prediction error) on
    min-max TRAIN columns, no PCA: the network learns its own compression.
    The first `holdout` share of TRAIN (in time order) is the fit part, the
    rest the early-stopping part; all of it is normal. Returns
    score(X, context=None), higher = more anomalous, and the settings used.
    `context`: rows that immediately precede X, prepended so X's first
    window - 1 rows get a full window rather than inheriting the first score.
    """
    from data import Split
    from models import REGISTRY

    c = DEFAULTS["lstm"]
    det = REGISTRY[DEEP_UNSUPERVISED[name]](train_stride=c["train_stride"])
    cut = int(len(X_train) * (1 - c["holdout"]))
    zeros = lambda n: np.zeros(n, dtype=int)                       # every row normal
    det.fit(Split(X_train[:cut], zeros(cut), X_train[cut:], zeros(len(X_train) - cut),
                  X_train[:0], zeros(0), feature_names=[], name="baseline TRAIN"))
    w = det.window

    def score(X, context=None):
        if context is None:
            return det.score(X)
        return det.score(np.vstack([context[-(w - 1):], X]))[w - 1:]

    return score, {"params": {"model": det.name, "epochs": det.epochs, "train_stride": det.train_stride,
                              "best_epoch": getattr(det, "best_epoch", None),
                              "val_loss_history": det.val_loss_history},
                   "fit_rows": cut}


# --------------------------------------------------------------------------
# 6-7. metrics and per-attack recall
# --------------------------------------------------------------------------

def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """Pointwise precision, recall, F1, UDR = FN/(FN+TP), FAR = FP/(FP+TN)."""
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12),
            "udr": fn / max(fn + tp, 1), "far": fp / max(fp + tn, 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _ranked(y, score) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    s = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max)
    return {"ap": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s))}


def match_attacks(test: pd.DataFrame, train: pd.DataFrame, actions: dict = None) -> pd.DataFrame:
    """
    Locate each attack in `actions` (default ATTACK_ACTIONS) among the
    labelled intervals of the attack file by its documented ACTION (the
    official attack list with start times is not in this repository): a
    spoofed sensor reads its set value (|median - value| in TRAIN standard
    deviations), a forced actuator switches to its set state (share of the
    interval in that state, minus the share in the 10 minutes before).
    Lowest total deviation wins; the evidence is returned.
    """
    actions = ATTACK_ACTIONS if actions is None else actions
    y = test["label"].to_numpy().astype(bool)
    runs = contiguous_runs(y)
    sd = train.std(numeric_only=True).replace(0, 1)
    rows = []
    for name, spec in actions.items():
        best = None
        for k, (s, e) in enumerate(runs):
            seg, pre = test.iloc[s:e], test.iloc[max(0, s - 600):s]
            dev, evidence = 0.0, {}
            for tag, v in spec.items():
                if is_sensor(tag):
                    med = float(seg[tag].median())
                    dev += abs(med - v) / sd[tag]
                    evidence[tag] = round(med, 3)
                else:
                    frac = float((seg[tag] == v).mean()) - (float((pre[tag] == v).mean()) if len(pre) else 0.0)
                    dev += 1 - frac
                    evidence[tag] = f"{(seg[tag] == v).mean():.2f} in state {v}"
            if best is None or dev < best[0]:
                best = (dev, k, s, e, evidence)
        rows.append({"attack": name, "type": attack_type(spec), "interval": best[1],
                     "start": best[2], "end": best[3],
                     "match_deviation": round(best[0], 3), "evidence": best[4]})
    return pd.DataFrame(rows).set_index("attack")


# --------------------------------------------------------------------------
# 8. diagnostics
# --------------------------------------------------------------------------

def data_audit(train: pd.DataFrame, test: pd.DataFrame, report: dict) -> dict:
    """Source files, row counts, time ranges, raw labels and attack intervals."""
    from config import DATA_RAW

    runs = contiguous_runs(test["label"].to_numpy().astype(bool))
    lengths = np.array([e - s for s, e in runs])
    gaps = test["timestamp"].diff().dt.total_seconds().iloc[1:]
    audit = {
        "train_source": report["A1"].get("source"), "test_source": report["A2"].get("source"),
        "train_rows": len(train), "train_start": str(train.timestamp.iloc[0]),
        "train_end": str(train.timestamp.iloc[-1]),
        "test_rows": len(test), "test_start": str(test.timestamp.iloc[0]),
        "test_end": str(test.timestamp.iloc[-1]),
        "test_attack_rows": int(test.label.sum()), "test_attack_rate": float(test.label.mean()),
        "labelled_intervals": len(runs),
        "interval_length_s": {"min": int(lengths.min()), "median": float(np.median(lengths)),
                              "max": int(lengths.max())},
        "test_timestamp_gaps_not_1s": {str(k): int(v) for k, v in gaps[gaps != 1].value_counts().items()},
    }
    src = report["A2"].get("source")
    if src and src.endswith(".csv") and (DATA_RAW / src).exists():   # raw label strings, before 0/1
        raw = pd.read_csv(DATA_RAW / src, usecols=[52]).iloc[:, 0].astype(str).str.strip()
        audit["raw_label_strings"] = {k: int(v) for k, v in raw.value_counts().items()}
    return audit


def nominal_level_drift(train: pd.DataFrame, prep: Preprocess) -> pd.Series:
    """
    Day-to-day movement of each column's NORMAL level, from the training
    (normal) file only: range of the daily means over the five full days
    (23-27 Dec), in min-max units, i.e. as a share of the column's normal
    range. Rule R2 flags a sensor whose normal level moves by more than
    DEFAULTS["drift_rule"] of its range from one normal day to another.
    """
    x = pd.DataFrame(prep.minmax(train), columns=prep.columns, index=train["timestamp"])
    days = x[(x.index >= "2015-12-23") & (x.index < "2015-12-28")].resample("1D").mean()
    return days.max() - days.min()


ALL = "none (all sensors)"


def exclusion_rules(mapping: dict, drift: pd.Series) -> dict[str, set]:
    """Sensor sets to exclude, each decided on the normal (training) file only."""
    sensors = list(mapping)
    r1 = {n for n in sensors if mapping[n]["r2"] < DEFAULTS["mapping_r2_rule"]}
    r2 = {n for n in sensors if drift[n] > DEFAULTS["drift_rule"]}
    return {ALL: set(), "R1: mapping R^2 < 0.5": r1,
            "R2: normal-level drift > 10% of range": r2, "R1 + R2": r1 | r2}


def per_sensor_table(det: ProcessPredictor, columns, Xte, y, mapping, drift, rules) -> pd.DataFrame:
    """Per-sensor evidence for one supervised detector on TEST."""
    names = [columns[t] for t in det.targets]
    exc = det.exceedances(Xte)
    r_te = det.residuals(Xte)
    normal = y == 0
    fa_events = [sum(1 for s, e in contiguous_runs(exc[:, j]) if not y[s:e].any())
                 for j in range(len(names))]
    t = pd.DataFrame({
        "mapping_r2": [mapping[n]["r2"] for n in names],
        "inputs": [",".join(mapping[n]["input_names"]) for n in names],
        "pred_var_ratio": det.pred_var_ratio_,
        "resid_sd_train": det.resid_sd_,
        "resid_shift_test_normal_sd": (r_te[normal].mean(axis=0) - det.target_) / det.resid_sd_,
        "nominal_level_drift": [drift[n] for n in names],
        "false_alarm_rows": exc[normal].sum(axis=0),
        "false_alarm_rate": exc[normal].mean(axis=0),
        "false_alarm_events": fa_events,
        "attack_rows_flagged": exc[~normal].sum(axis=0),
    }, index=names)
    for label, excl in rules.items():
        if excl:
            t[label.split(":")[0]] = [n in excl for n in names]
    return t.sort_values("false_alarm_rows", ascending=False)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

C_SUP, C_UNS = "#F7DC6F", "#7DCEA0"                    # yellow, green
RULE_COLOURS = {"none": "#555555", "R1:": C_SUP, "R2:": C_UNS, "R1 +": "#B8721F"}   # by label prefix


def fig_scatter(test, prep, path, pca=False, n=6000, seed=42):
    """Two features of the attack file against sample index, attack rows red."""
    idx = np.sort(np.random.default_rng(seed).choice(len(test), size=min(n, len(test)), replace=False))
    sub = test.iloc[idx]
    if pca:
        Z = prep.pca_features(sub)[:, :2]
        cols = [("PC1", Z[:, 0]), ("PC2", Z[:, 1])]
    else:
        X = prep.minmax(sub)
        cols = [(c, X[:, prep.columns.index(c)]) for c in ("LIT101", "LIT301")]
    att = sub["label"].to_numpy() == 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (name, v) in zip(axes, cols):
        ax.scatter(v[~att], idx[~att], s=2, c="#2E64B0", label="Normal")
        ax.scatter(v[att], idx[att], s=2, c="#D62728", label="Attack")
        ax.set_xlabel(name); ax.set_ylabel("sample index (attack file)")
        ax.legend(loc="lower right", markerscale=4)
    fig.suptitle("Attack file in the first two PCA components (fit on the normal file)" if pca else
                 "Attack file: min-max LIT101 and LIT301 (scaler fit on the normal file)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_metric_bars(ours: pd.DataFrame, metric: str, path, title: str, ascending: bool):
    """One bar per model, sorted; supervised hatched, unsupervised solid."""
    vals = ours[metric].astype(float).sort_values(ascending=ascending)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (m, v) in enumerate(vals.items()):
        sup = m in SUPERVISED
        ax.bar(i, v, color=C_SUP if sup else C_UNS, hatch="///" if sup else None,
               edgecolor="#555555", width=0.75)
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=8)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(vals.index, rotation=40, ha="right")
    ax.set_ylabel(metric.upper() if metric in ("udr", "far") else metric.capitalize())
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=C_SUP, hatch="///", edgecolor="#555"),
               plt.Rectangle((0, 0), 1, 1, facecolor=C_UNS, edgecolor="#555")]
    ax.legend(handles, ["Supervised, no attack labels (process / state prediction)", "Unsupervised"], fontsize=8)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_ablation(abl: pd.DataFrame, path,
                 title: str = "Ablation: sensors excluded by rules decided on the normal file only"):
    """F1 (thresholded) and AP (threshold-free) per model under each exclusion
    rule. If excluded sensors limit detection, both rise; if only F1 moves,
    the threshold rule is the bottleneck."""
    d = abl.reset_index().rename(columns={"level_0": "rule", "level_1": "model"})
    rules = list(dict.fromkeys(d["rule"]))
    models = list(dict.fromkeys(d["model"]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    w = 0.8 / len(rules)
    for ax, metric in zip(axes, ("f1", "ap")):
        for k, rule in enumerate(rules):
            sub = d[d["rule"] == rule].set_index("model").reindex(models)
            colour = next(v for p, v in RULE_COLOURS.items() if rule.startswith(p))
            ax.bar(np.arange(len(models)) + k * w, sub[metric], width=w, label=rule, color=colour,
                   edgecolor="#555555", linewidth=0.5)
        ax.set_xticks(np.arange(len(models)) + w * (len(rules) - 1) / 2)
        ax.set_xticklabels(models, rotation=40, ha="right")
        ax.set_title({"f1": "F1 (control limits / mu + 3 sigma)", "ap": "average precision (threshold-free)"}[metric])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(rules), fontsize=9, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.07, 1, 1]); fig.savefig(path, dpi=150); plt.close(fig)


def bar_figures(abl: pd.DataFrame, fig_dir=None):
    """Every bar chart, from the ablation table alone (results/baseline/ablation.csv
    read back with index_col=[0, 1] gives the same table), so they can be
    redrawn without rerunning the pipeline."""
    fig_dir = FIG if fig_dir is None else fig_dir
    r1_label = next(k for k in abl.index.get_level_values(0) if k.startswith("R1:"))
    r2_label = next(k for k in abl.index.get_level_values(0) if k.startswith("R2:"))
    for metric, title, asc in (("precision", "Precision", False), ("recall", "Recall", False),
                               ("f1", "F1 score", False), ("udr", "Undetected rate (UDR)", True),
                               ("far", "False alarm rate (FAR)", True)):
        fig_metric_bars(abl.loc[ALL], metric, fig_dir / f"{metric}.png", f"{title}, all sensors", asc)
        fig_metric_bars(abl.loc[r1_label], metric, fig_dir / f"{metric}_R1.png",
                        f"{title}, sensors the actuators do not explain excluded (R1)", asc)
        fig_metric_bars(abl.loc[r2_label], metric, fig_dir / f"{metric}_R2.png",
                        f"{title}, drifting analysers excluded (R2)", asc)
    fig_ablation(abl, fig_dir / "ablation_f1_ap.png")
    fig_ablation(abl.loc[[r1_label, r2_label]], fig_dir / "r1_vs_r2_f1_ap.png",
                 "Iteration 1 (R1: mapping R^2 < 0.5) vs iteration 2 (R2: normal-level drift > 10%)")


# --------------------------------------------------------------------------
# the whole pipeline
# --------------------------------------------------------------------------

def run(seed: int = DEFAULTS["seed"], verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    train, test, report = load_data()
    columns = [c for c in train.columns if c not in ("timestamp", "label")]
    y = test["label"].to_numpy().astype(int)
    audit = data_audit(train, test, report)
    log(f"train {len(train):,} rows from {audit['train_source']}")
    log(f"test  {len(test):,} rows from {audit['test_source']}, {y.mean():.2%} attack, "
        f"{audit['labelled_intervals']} labelled intervals")

    prep = Preprocess(columns, DEFAULTS["pca_variance"], seed).fit(train)
    Xtr, Xte = prep.minmax(train), prep.minmax(test)
    log(f"PCA keeps {prep.pca.n_components_} components for {DEFAULTS['pca_variance']:.0%} variance")

    mapping = select_mapping(Xtr, columns)
    drift = nominal_level_drift(train, prep)
    rules = exclusion_rules(mapping, drift)
    log("exclusion rules (normal file only): " + "; ".join(f"{k} -> {sorted(v)}" for k, v in rules.items() if v))

    primary, alternative, ablation, records, per_sensor, rule_predictions = {}, {}, {}, {}, {}, {}
    r1_label = next(k for k in rules if k.startswith("R1"))
    r2_label = next(k for k in rules if k.startswith("R2"))

    # XGBoost+lags: lag features of every actuator (a stage's input set can
    # hold one that never moves), over TRAIN then TEST in one pass: the attack
    # file starts one second after the normal file ends, and the features only
    # look back. 1D-CNN input: every actuator that moves in TRAIN.
    act = [i for i, c in enumerate(columns) if is_actuator(c)]
    moving = [i for i in act if np.ptp(Xtr[:, i]) > 0]
    F = lag_features(np.vstack([Xtr[:, act], Xte[:, act]]))
    Xtr_lag, Xte_lag = np.hstack([Xtr, F[:len(Xtr)]]), np.hstack([Xte, F[len(Xtr):]])
    mapping_lag = lagged_mapping(mapping, act, len(columns), len(DEFAULTS["lags"]))
    mapping_cnn = {s: {**m, "inputs": moving, "input_names": [columns[a] for a in moving]}
                   for s, m in mapping.items()}

    # ---- supervised: process prediction + control limits -----------------
    for name in PROCESS_PREDICTORS:
        t = time.time()
        if name == "XGBoost+lags":
            det, Xs, mp = ProcessPredictor(name, mapping_lag, seed).fit(Xtr_lag), Xte_lag, mapping
        elif name == "1D-CNN":
            det, Xs, mp = CNNPredictor(mapping_cnn, seed).fit(Xtr), Xte, mapping_cnn
        else:
            det, Xs, mp = ProcessPredictor(name, mapping, seed).fit(Xtr), Xte, mapping
        marg = det.margins(Xs)                        # rows x sensors, > 0 = alarm
        names = [columns[i] for i in det.targets]
        for label, excl in rules.items():
            keep = np.array([n not in excl for n in names])
            score = marg[:, keep].max(axis=1)
            pred = (score > 0).astype(int)
            res = {**metrics(y, pred), **_ranked(y, score), "n_sensors": int(keep.sum())}
            ablation.setdefault(label, {})[name] = res
            rule_predictions.setdefault(label, {})[name] = pred
            if label == ALL:
                primary[name] = res
        alternative[name] = metrics(y, det.predict(Xs, "accumulating"))
        records[name] = det.record(columns)
        per_sensor[name] = per_sensor_table(det, columns, Xs, y, mp, drift, rules)
        log(f"  {name:<17} F1 {primary[name]['f1']:.3f}  AP {primary[name]['ap']:.3f}  "
            f"FAR {primary[name]['far']:.3%}  ({time.time() - t:.0f}s)")

    # ---- supervised: actuator-state prediction, refit per feature set -----
    S_tr, S_te = train[columns].to_numpy(), test[columns].to_numpy()
    for label, excl in rules.items():
        t = time.time()
        det = StatePredictor(columns, excl, seed).fit(Xtr, S_tr)
        score = det.score(Xte, S_te)
        pred = (score > det.threshold_).astype(int)
        res = {**metrics(y, pred), **_ranked(y, score), "n_sensors": det.n_inputs,
               "threshold": det.threshold_}
        ablation.setdefault(label, {})["LogisticR"] = res
        rule_predictions.setdefault(label, {})["LogisticR"] = pred
        if label == ALL:
            primary["LogisticR"] = res
            thr_t = score.mean() + DEFAULTS["k_sigma"] * score.std()     # labels never used
            alternative["LogisticR"] = metrics(y, (score > thr_t).astype(int))
            records["LogisticR"] = {**det.record(), "threshold_test_stats": float(thr_t)}
        log(f"  [{label[:2]}] LogisticR         F1 {res['f1']:.3f}  AP {res['ap']:.3f}  "
            f"FAR {res['far']:.3%}  ({time.time() - t:.0f}s)")

    # ---- unsupervised: refit per feature set, mu + k sigma on TRAIN -------
    for label in (ALL, r1_label, r2_label):
        cols = [c for c in columns if c not in rules[label]]
        p = Preprocess(cols, DEFAULTS["pca_variance"], seed).fit(train)
        Ztr, Zte = p.pca_features(train), p.pca_features(test)
        Mtr, Mte = p.minmax(train), p.minmax(test)
        for name in UNSUPERVISED:
            t = time.time()
            if name in DEEP_UNSUPERVISED:
                score, info = deep_scorer(name, Mtr)
                s_tr, s_te = score(Mtr), score(Mte, context=Mtr)    # the attack file continues TRAIN's clock
            else:
                score, info = unsupervised_scorer(name, Ztr, seed)
                s_tr, s_te = score(Ztr), score(Zte)
            thr = s_tr.mean() + DEFAULTS["k_sigma"] * s_tr.std()      # recalibrated per feature set
            pred = (s_te > thr).astype(int)
            res = {**metrics(y, pred), **_ranked(y, s_te),
                   "threshold": float(thr), "pca_components": int(p.pca.n_components_)}
            ablation.setdefault(label, {})[name] = res
            rule_predictions.setdefault(label, {})[name] = pred
            if label == ALL:
                primary[name] = res
                thr_t = s_te.mean() + DEFAULTS["k_sigma"] * s_te.std()   # labels never used
                alternative[name] = metrics(y, (s_te > thr_t).astype(int))
                records[name] = {**info, "threshold_train_stats": float(thr),
                                 "threshold_test_stats": float(thr_t)}
            log(f"  [{label[:2]}] {name:<17} F1 {res['f1']:.3f}  AP {res['ap']:.3f}  "
                f"FAR {res['far']:.3%}  ({time.time() - t:.0f}s)")

    ours = pd.DataFrame(primary).T
    ours_alt = pd.DataFrame(alternative).T
    abl = pd.concat({k: pd.DataFrame(v).T for k, v in ablation.items()})

    # ---- per-attack recall -----------------------------------------------
    # With all sensors the supervised detectors alarm on ~76% of normal rows,
    # so per-attack recall is ~1 everywhere and says little; the tables under
    # R1 and R2 are the informative ones. All three are saved.
    attacks = match_attacks(test, train)
    per_attack = {
        label: pd.DataFrame({m: {a: float(rule_predictions[label][m][r.start:r.end].mean())
                                 for a, r in attacks.iterrows()}
                             for m in SUPERVISED + UNSUPERVISED if m in rule_predictions[label]})
        for label in (ALL, r1_label, r2_label)
    }

    # ---- figures ---------------------------------------------------------
    fig_scatter(test, prep, FIG / "features_scatter.png")
    fig_scatter(test, prep, FIG / "pca_scatter.png", pca=True)
    bar_figures(abl)

    # ---- save ------------------------------------------------------------
    out = {
        "pipeline": "src/baseline.py -- A1 -> A2 supervised process prediction vs unsupervised scoring",
        "created": datetime.now().isoformat(timespec="seconds"),
        "defaults": DEFAULTS, "data_report": report, "data_audit": audit,
        "pca_components": int(prep.pca.n_components_),
        "mapping": {k: {kk: v[kk] for kk in ("input_names", "r2", "mse")} for k, v in mapping.items()},
        "nominal_level_drift": drift.to_dict(),
        "exclusion_rules": {k: sorted(v) for k, v in rules.items()},
        "metrics": ours.to_dict(orient="index"),
        "metrics_alternative_reading": ours_alt.to_dict(orient="index"),
        "ablation": {k: pd.DataFrame(v).T.to_dict(orient="index") for k, v in ablation.items()},
        "decision_records": records,
        "per_sensor": {k: v.to_dict(orient="index") for k, v in per_sensor.items()},
        "attacks": attacks.drop(columns=["evidence"]).to_dict(orient="index"),
        "attack_evidence": attacks["evidence"].to_dict(),
        "per_attack_recall": {k: v.to_dict(orient="index") for k, v in per_attack.items()},
        "seconds": time.time() - t0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=str))
    ours.to_csv(OUT / "metrics.csv")
    ours_alt.to_csv(OUT / "metrics_alternative_reading.csv")
    abl.to_csv(OUT / "ablation.csv")
    pd.concat(per_sensor).to_csv(OUT / "per_sensor.csv")
    per_attack[ALL].to_csv(OUT / "per_attack_recall.csv")
    per_attack[r1_label].to_csv(OUT / "per_attack_recall_R1.csv")
    per_attack[r2_label].to_csv(OUT / "per_attack_recall_R2.csv")
    log(f"saved -> {OUT}  ({time.time() - t0:.0f}s)")
    return {"metrics": ours, "alternative": ours_alt, "ablation": abl, "per_sensor": per_sensor,
            "per_attack": per_attack, "attacks": attacks, "audit": audit,
            "rules": rules, "results": out}


# --------------------------------------------------------------------------
# 9. attack-label classifiers: separate protocol, separate table
# --------------------------------------------------------------------------

ATTACK_LABEL_MODELS = {"XGBoost+lags": "xgboost_lags", "LogisticR": "logistic_regression",
                       "1D-CNN": "cnn_classifier"}


def run_attack_label(verbose: bool = True) -> pd.DataFrame:
    """
    The attack-label classifiers of models.py, trained on labelled attacks.
    run()'s TRAIN (the normal file) has no attack rows, so they run on the
    repository's episode-stratified split instead: the normal file and the
    attack file in time order, whole attack incidents assigned to train / val
    / test by the stage they target, purged boundaries
    (data.stratified_episode_split). Threshold: experiment.fit_and_score's
    default, the THRESHOLD_PERCENTILE of validation normal-row scores; alarm
    where score >= threshold (evaluate.py's convention). Metrics: the same
    functions as run(). A different test set from run(), so the two tables
    are not comparable row for row.
    """
    from data import load
    from experiment import fit_and_score
    from models import REGISTRY

    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    split = load("swat", protocol="stratified")
    y = split.y_test.astype(int)
    rows, names = {}, {}
    for label, key in ATTACK_LABEL_MODELS.items():
        t = time.time()
        det = REGISTRY[key]()
        thr, s = fit_and_score(split, det)
        rows[label] = {**metrics(y, (s >= thr).astype(int)), **_ranked(y, s), "threshold": float(thr)}
        names[label] = det.name
        log(f"  {label:<13} ({det.name}) F1 {rows[label]['f1']:.3f}  AP {rows[label]['ap']:.3f}  "
            f"FAR {rows[label]['far']:.3%}  ({time.time() - t:.0f}s)")
    table = pd.DataFrame(rows).T
    out = {
        "pipeline": "src/baseline.py run_attack_label -- attack-label classifiers, episode-stratified split",
        "created": datetime.now().isoformat(timespec="seconds"),
        "models": names,
        "split": split.summary(),
        "test_rows": int(len(y)), "test_attack_rows": int(y.sum()), "test_attack_rate": float(y.mean()),
        "threshold": f"{THRESHOLD_PERCENTILE}th percentile of validation normal-row scores",
        "metrics": table.to_dict(orient="index"),
        "seconds": time.time() - t0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attack_label.json").write_text(json.dumps(out, indent=1, default=str))
    table.to_csv(OUT / "attack_label.csv")
    log(f"saved -> {OUT / 'attack_label.csv'}  ({time.time() - t0:.0f}s)")
    return table


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["baseline", "attack-label"],
                    help="run one part; default runs both")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    if args.only != "attack-label":
        r = run()
        print(r["metrics"][["precision", "recall", "f1", "udr", "far", "ap", "roc_auc"]].round(4).to_string())
        print()
        print(r["ablation"][["precision", "recall", "f1", "far", "ap", "roc_auc"]].round(3).to_string())
    if args.only != "baseline":
        print(run_attack_label()[["precision", "recall", "f1", "far", "ap", "roc_auc"]].round(3).to_string())
