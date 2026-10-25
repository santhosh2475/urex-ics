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
from data import Split, unwindow_scores, window_ends


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
# classical / shallow supervised baselines
# --------------------------------------------------------------------------
# Six lightweight supervised classifiers, binary, over SWaT's Normal/Attack
# label -- binary only, since that label carries no per-attack-type
# breakdown. A per-attack-type variant would need iTrust's separate
# List_of_attacks metadata file, which this project does not have (see
# swat-preprocess/README.md).
#
# These use the same chronological train/val/test split as every other
# detector in the registry -- not a shuffled k-fold split, which would leak
# future rows into training (module docstring, rule 1) -- so the comparison
# stays apples-to-apples with xgboost rather than reintroducing a different,
# invalid protocol just for this one family.

class NaiveBayesDetector(Detector):
    """Gaussian Naive Bayes: per-feature likelihood ratio, no iteration."""

    supervised = True
    name = "naive_bayes"

    def __init__(self, **params):
        self.params = params
        self.model = None

    def fit(self, split: Split) -> "NaiveBayesDetector":
        from sklearn.naive_bayes import GaussianNB

        self.model = GaussianNB(**self.params)
        self.model.fit(split.X_train, split.y_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() first")
        return self.model.predict_proba(X)[:, 1]


class RandomForestSCADADetector(Detector):
    """Random Forest (Breiman, 2001): bagged decision trees, majority vote.

    Reference: L. Breiman, "Random forests," Mach. Learn., vol. 45, no. 1, pp.
               5-32, 2001, doi: 10.1023/A:1010933404324.
    """

    supervised = True
    name = "random_forest"

    def __init__(self, n_estimators: int = 100, **params):
        self.params = {
            "n_estimators": n_estimators,
            "class_weight": "balanced",  # SWaT's attack rate is a few percent
            "random_state": SEED,
            "n_jobs": -1,
            **params,
        }
        self.model = None

    def fit(self, split: Split) -> "RandomForestSCADADetector":
        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(**self.params)
        self.model.fit(split.X_train, split.y_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() first")
        return self.model.predict_proba(X)[:, 1]


class OneRDetector(Detector):
    """
    OneR (Holte, 1993) [1]: the simplest possible rule-based baseline, worth
    building by hand rather than importing -- there is no standard library
    implementation, and the algorithm really is three steps.

    For EACH feature: bin its values, then assign each bin the majority
    class of the training rows that fall in it. That gives one candidate
    rule per feature. Keep the single feature/rule pair with the lowest
    training error. At score time, a row's anomaly score is the attack rate
    of the bin its chosen feature falls into -- continuous, so it still
    supports the project's percentile-threshold protocol, though the
    original OneR only ever outputs a hard 0/1 call.

    [1] R. C. Holte, "Very simple classification rules perform well on most
        commonly used datasets," Mach. Learn., vol. 11, no. 1, pp. 63-90, 1993,
        doi: 10.1023/A:1022631118932.
    """

    supervised = True
    name = "oner"

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.best_feature: int | None = None
        self.bin_edges: np.ndarray | None = None
        self.bin_attack_rate: np.ndarray | None = None

    def _fit_one_feature(self, x: np.ndarray, y: np.ndarray):
        edges = np.unique(np.quantile(x, np.linspace(0, 1, self.n_bins + 1)))
        if len(edges) < 3:  # a near-constant feature: one bin only
            edges = np.array([x.min(), x.max()])
        bin_idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
        rate = np.zeros(len(edges) - 1)
        errors = 0
        for b in range(len(edges) - 1):
            mask = bin_idx == b
            if not mask.any():
                continue
            rate[b] = y[mask].mean()
            majority = round(rate[b])  # 0 or 1
            errors += (y[mask] != majority).sum()
        return edges, rate, errors

    def fit(self, split: Split) -> "OneRDetector":
        best_errors = None
        for f in range(split.n_features):
            edges, rate, errors = self._fit_one_feature(
                split.X_train[:, f], split.y_train
            )
            if best_errors is None or errors < best_errors:
                best_errors, self.best_feature = errors, f
                self.bin_edges, self.bin_attack_rate = edges, rate
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.best_feature is None:
            raise RuntimeError("call fit() first")
        x = X[:, self.best_feature]
        bin_idx = np.clip(
            np.digitize(x, self.bin_edges[1:-1]), 0, len(self.bin_edges) - 2
        )
        return self.bin_attack_rate[bin_idx]


class J48Detector(Detector):
    """
    Decision tree, standing in for Weka's J48 (its C4.5 implementation).
    Not identical: C4.5 splits on gain RATIO and post-prunes via a
    confidence-based procedure; scikit-learn's tree splits on information
    gain (entropy) and only prunes if ccp_alpha is set. Close enough to
    represent "a single decision tree" as a family -- unmaintained py-c45
    packages exist if the exact algorithm ever matters, but they buy no
    accuracy difference that shows up in practice.
    """

    supervised = True
    name = "j48"

    def __init__(self, max_depth: int | None = None, **params):
        self.params = {
            "criterion": "entropy",  # matches C4.5's information-gain family
            "max_depth": max_depth,
            "class_weight": "balanced",
            "random_state": SEED,
            **params,
        }
        self.model = None

    def fit(self, split: Split) -> "J48Detector":
        from sklearn.tree import DecisionTreeClassifier

        self.model = DecisionTreeClassifier(**self.params)
        self.model.fit(split.X_train, split.y_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() first")
        return self.model.predict_proba(X)[:, 1]


class NNgeDetector(Detector):
    """
    Stands in for NNge (Martin, 1995) [1]: nearest-neighbour classification
    against generalised rectangular exemplars merged from training points,
    rather than every individual point. That merging step is a nontrivial
    algorithm in its own right with no maintained Python implementation.

    Uses plain k-NN instead -- the same "instance-based" family, and
    cheaper. The honest name for this is a stand-in, not a reproduction;
    the real merging-exemplar version is a bigger undertaking than the
    other five put together.

    [1] B. Martin, "Instance-based learning: Nearest neighbour with
        generalisation," M.Sc. thesis, Dept. Comput. Sci., Univ. Waikato,
        Hamilton, New Zealand, 1995.
    """

    supervised = True
    name = "nnge_stub_knn"

    def __init__(self, n_neighbors: int = 5, **params):
        self.params = {"n_neighbors": n_neighbors, "n_jobs": -1, **params}
        self.model = None

    def fit(self, split: Split) -> "NNgeDetector":
        from sklearn.neighbors import KNeighborsClassifier

        self.model = KNeighborsClassifier(**self.params)
        self.model.fit(split.X_train, split.y_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() first")
        return self.model.predict_proba(X)[:, 1]


class SVMDetector(Detector):
    """
    SVM trained by sequential minimal optimization (libsvm's SMO solver,
    under scikit-learn's SVC).

    CAUTION: SMO is O(n^2)-O(n^3) in training rows. SWaT's training window
    is ~270k rows; an SVC.fit on all of it would run for hours. `max_train`
    subsamples (stratified, so rare attack rows survive) before fitting --
    exactly the kind of thing that would go unnoticed until a run had
    already burned the night, so it is a parameter, not a silent default.
    """

    supervised = True
    name = "svm"

    def __init__(self, max_train: int = 20_000, **params):
        self.max_train = max_train
        self.params = {
            "kernel": "rbf",
            "class_weight": "balanced",
            "probability": True,  # needed for predict_proba scoring
            "random_state": SEED,
            **params,
        }
        self.model = None

    def fit(self, split: Split) -> "SVMDetector":
        from sklearn.svm import SVC

        X, y = split.X_train, split.y_train
        if len(X) > self.max_train:
            rng = np.random.default_rng(SEED)
            normal_idx = np.flatnonzero(y == 0)
            attack_idx = np.flatnonzero(y == 1)
            frac = self.max_train / len(X)
            keep = np.concatenate(
                [
                    rng.choice(
                        normal_idx, size=int(len(normal_idx) * frac), replace=False
                    ),
                    rng.choice(
                        attack_idx,
                        size=max(1, int(len(attack_idx) * frac)),
                        replace=False,
                    ),
                ]
            )
            X, y = X[keep], y[keep]

        self.model = SVC(**self.params)
        self.model.fit(X, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() first")
        return self.model.predict_proba(X)[:, 1]


class LogisticRegressionDetector(Detector):
    """
    Logistic regression: one linear score per row, squashed by a sigmoid,
    fit by minimising binary cross-entropy (log loss) plus an L2 penalty.

    The simplest model whose training objective IS the BCE loss every
    other supervised probabilistic model here also optimises (xgboost's
    binary:logistic, the MLP and CNN's BCEWithLogitsLoss), so it is the
    reference point for "how much does non-linearity buy". Its coefficients
    are directly readable: a positive weight on a standardised sensor means
    "higher than usual reading -> more likely attack".
    """

    supervised = True
    name = "logistic_regression"

    def __init__(self, C: float = 1.0, **params):
        self.params = {
            "C": C,
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": SEED,
            **params,
        }
        self.model = None

    def fit(self, split: Split) -> "LogisticRegressionDetector":
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(**self.params)
        self.model.fit(split.X_train, split.y_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() first")
        return self.model.predict_proba(X)[:, 1]


class AdaBoostDetector(Detector):
    """
    AdaBoost (Freund & Schapire, 1997) [1] over shallow trees.

    Each round reweights the training rows the ensemble so far gets wrong,
    so later trees concentrate on the hard cases. Note the loss: AdaBoost
    minimises EXPONENTIAL loss, not cross-entropy -- it is the boosting
    model that is not BCE-trained, which makes it a useful contrast to
    xgboost (gradient boosting on the BCE/log loss). Both are reported.

    Imbalance is handled the only way AdaBoost allows: initial sample
    weights balanced between the classes.

    [1] Y. Freund and R. E. Schapire, "A decision-theoretic generalization of
        on-line learning and an application to boosting," J. Comput. Syst.
        Sci., vol. 55, no. 1, pp. 119-139, 1997, doi: 10.1006/jcss.1997.1504.
    """

    supervised = True
    name = "adaboost"

    def __init__(self, n_estimators: int = 200, max_depth: int = 2,
                 learning_rate: float = 0.5, **params):
        self.max_depth = max_depth
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "random_state": SEED,
            **params,
        }
        self.model = None

    def fit(self, split: Split) -> "AdaBoostDetector":
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.utils.class_weight import compute_sample_weight

        self.model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=self.max_depth),
            **self.params,
        )
        self.model.fit(
            split.X_train, split.y_train,
            sample_weight=compute_sample_weight("balanced", split.y_train),
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() first")
        return self.model.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------
# shared training machinery for the torch models
# --------------------------------------------------------------------------
# Windows are never materialised. `window_ends` gives the index of each
# window's last row, and a batch is gathered on the fly as
# X[ends[:, None] + offsets]. Stacking every window up front, which this
# module used to do, costs ~13 GB for SWaT's training window in float64.

def _offsets(window: int):
    import torch

    return torch.arange(-window + 1, 1)


def _gather(Xt, ends, window: int):
    """(len(ends), window, n_features) batch of windows ending at `ends`."""
    return Xt[ends[:, None] + _offsets(window).to(ends.device)]


def _balanced_bce(logits, y) -> float:
    """Mean BCE over positives and over negatives, averaged: comparable
    between a subsampled training epoch and the full validation set, which
    have very different class ratios."""
    import torch.nn.functional as F

    per = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pos, neg = y > 0.5, y <= 0.5
    parts = [per[m].mean() for m in (pos, neg) if m.any()]
    return float(sum(parts) / len(parts))


class _BCETrainer:
    """
    Mixin: the supervised BCE training loop shared by the MLP and the CNN.

    What it does differently from a bare `for epoch: for batch: step()`,
    and why each one improves the loss curve:

      negative subsampling  every epoch uses ALL attack items plus a fresh
                            random draw of `neg_ratio` times as many normal
                            items. With ~1% attacks, pos_weight=n_neg/n_pos
                            is ~100, so one misclassified attack row moves
                            the loss as much as 100 normal rows -- that is
                            what made the CNN's loss jump around (0.0073 ->
                            0.0109 between epochs). Subsampling brings the
                            ratio to ~neg_ratio, and since a fresh sample is
                            drawn each epoch, all normal data is still seen.
      AdamW                 Adam with decoupled weight decay: penalises
                            large weights, which the 1% minority class
                            otherwise invites the model to memorise.
      OneCycle LR           warm up, then anneal: big steps early, small
                            steps late, so the loss settles instead of
                            bouncing at a fixed learning rate.
      gradient clipping     caps a single bad batch's update.
      validation tracking   balanced BCE and PR-AUC on validation every
                            epoch, stored in val_loss_history / val_pr_auc.
      best-epoch restore    the weights kept are those of the epoch with the
                            best validation PR-AUC, and training stops after
                            `patience` epochs without improvement.

    The last two matter most. Training loss is not the goal: the CNN in the
    chronological run reached training loss 0.007 and test ROC-AUC 0.44.
    A falling training loss with a rising validation loss is overfitting,
    and only the validation curve can show it.
    """

    def _fit_bce(self, net, gather, y_items, val_gather, y_val_items):
        import torch
        from sklearn.metrics import average_precision_score

        device = self.device
        rng = np.random.default_rng(SEED)
        pos = np.flatnonzero(y_items == 1)
        neg = np.flatnonzero(y_items == 0)
        if len(pos) == 0:
            raise ValueError("Supervised model cannot train on a split with no anomalies.")
        n_neg = len(neg) if self.neg_ratio is None else min(len(neg), int(self.neg_ratio * len(pos)))
        n_items = len(pos) + n_neg
        steps = int(np.ceil(n_items / self.batch_size))

        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr, total_steps=self.epochs * steps
        )
        pos_weight = torch.tensor(n_neg / len(pos), dtype=torch.float32, device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        y_all = torch.tensor(y_items, dtype=torch.float32)

        has_val = val_gather is not None and 0 < y_val_items.sum() < len(y_val_items)
        best_score, best_state, bad = -np.inf, None, 0
        self.loss_history, self.val_loss_history, self.val_pr_auc = [], [], []

        for epoch in range(self.epochs):
            net.train()
            items = np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)])
            rng.shuffle(items)
            logits_all, y_seen = [], []
            for i in range(0, n_items, self.batch_size):
                idx = items[i : i + self.batch_size]
                xb = gather(idx)
                yb = y_all[idx].to(device)
                opt.zero_grad()
                logits = net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.clip)
                opt.step()
                sched.step()
                logits_all.append(logits.detach())
                y_seen.append(yb)
            train_loss = _balanced_bce(torch.cat(logits_all), torch.cat(y_seen))
            self.loss_history.append(train_loss)

            msg = f"    epoch {epoch:>3}  train bce {train_loss:.4f}"
            if has_val:
                v_logits = self._predict_logits(net, val_gather, len(y_val_items))
                v_loss = _balanced_bce(v_logits, torch.tensor(y_val_items, dtype=torch.float32, device=v_logits.device))
                v_ap = float(average_precision_score(y_val_items, v_logits.cpu().numpy()))
                self.val_loss_history.append(v_loss)
                self.val_pr_auc.append(v_ap)
                msg += f"  val bce {v_loss:.4f}  val PR-AUC {v_ap:.4f}"
                if v_ap > best_score:
                    best_score, bad = v_ap, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    self.best_epoch = epoch
                else:
                    bad += 1
            print(msg)
            if has_val and bad >= self.patience:
                print(f"    early stop: no val PR-AUC gain for {self.patience} epochs; "
                      f"restoring epoch {self.best_epoch}")
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        return net

    def _predict_logits(self, net, gather, n: int):
        import torch

        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, n, 4096):
                out.append(net(gather(np.arange(i, min(i + 4096, n)))))
        return torch.cat(out)


class BCEMLPDetector(_BCETrainer, Detector):
    """
    Feed-forward network on single rows, trained on BCE with Adam(W).

    `hidden=()` gives NO hidden layer: one linear layer and a sigmoid, i.e.
    logistic regression, but trained by Adam on mini-batches through the
    same `_BCETrainer` loop as the CNN. Registered as `logreg_adam`, next to
    sklearn's `logistic_regression` (same model, exact lbfgs solver): if the
    two disagree, the training loop is the problem, not the model.

    `hidden=(64, 32)` (registered as `mlp_bce`) adds two ReLU layers with
    dropout -- the non-linear, row-wise counterpart to xgboost.
    """

    supervised = True

    def __init__(
        self,
        hidden: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        epochs: int = 30,
        batch_size: int = 1024,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        neg_ratio: float | None = 20.0,
        patience: int = 5,
        clip: float = 1.0,
        device: str | None = None,
    ):
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.neg_ratio = neg_ratio
        self.patience = patience
        self.clip = clip
        self.device = device
        self.name = "logreg_adam" if not self.hidden else f"mlp_bce(h={'-'.join(map(str, self.hidden))})"
        self.model = None
        self.loss_history: list[float] = []

    def _build(self, n_features: int):
        import torch
        import torch.nn as nn

        torch.manual_seed(SEED)
        layers, d = [], n_features
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        body = nn.Sequential(*layers)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = body

            def forward(self, x):
                return self.body(x).squeeze(-1)

        return Net()

    def fit(self, split: Split) -> "BCEMLPDetector":
        import torch

        self.device = _resolve_device(self.device)
        Xt = torch.tensor(split.X_train, dtype=torch.float32, device=self.device)
        Xv = torch.tensor(split.X_val, dtype=torch.float32, device=self.device)
        net = self._build(split.n_features).to(self.device)
        self.model = self._fit_bce(
            net,
            lambda idx: Xt[torch.as_tensor(idx, device=self.device)],
            split.y_train,
            lambda idx: Xv[torch.as_tensor(idx, device=self.device)],
            split.y_val,
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("call fit() first")
        Xs = torch.tensor(X, dtype=torch.float32, device=self.device)
        logits = self._predict_logits(
            self.model, lambda idx: Xs[torch.as_tensor(idx, device=self.device)], len(X)
        )
        return torch.sigmoid(logits).cpu().numpy()


# --------------------------------------------------------------------------
# deep learning
# --------------------------------------------------------------------------

class _WindowedMSE:
    """
    Mixin: memory-safe training loop for the two unsupervised LSTMs.

    Same objective as before (MSE on normal windows, Adam), plus a
    validation curve on validation's NORMAL windows and best-epoch restore.
    For an unsupervised model the validation loss is the only honest stop
    signal: the training loss keeps falling as the model memorises the
    normal week, while the validation loss shows whether it still
    generalises to normal operation on other days.
    """

    def _fit_mse(self, split: Split, step_loss):
        import torch

        self.device = _resolve_device(self.device)
        dev = self.device
        Xt = torch.tensor(split.X_train, dtype=torch.float32, device=dev)
        # Unsupervised: normal training rows only, windowed per contiguous
        # normal run and never across a block join -- masking first and
        # windowing second would splice the rows either side of every
        # removed attack together and teach the model jumps that never
        # happen in deployment.
        ends = window_ends(
            split.normal_mask() & split.continuity_mask("train"),
            window=self.window, stride=self.train_stride,
        )
        val_ends = None
        if (split.y_val == 0).sum() >= self.window:
            Xv = torch.tensor(split.X_val, dtype=torch.float32, device=dev)
            try:
                val_ends = window_ends(
                    (split.y_val == 0) & split.continuity_mask("val"),
                    window=self.window, stride=max(self.train_stride, 5),
                )
            except ValueError:
                val_ends = None

        net = self._build(split.n_features).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        rng = np.random.default_rng(SEED)
        self.loss_history, self.val_loss_history = [], []
        best, best_state, bad = np.inf, None, 0

        for epoch in range(self.epochs):
            net.train()
            order = rng.permutation(ends)  # safe: windows carry their own time order
            total = 0.0
            for i in range(0, len(order), self.batch_size):
                batch = _gather(Xt, torch.as_tensor(order[i : i + self.batch_size], device=dev), self.window)
                opt.zero_grad()
                loss = step_loss(net, batch)
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            self.loss_history.append(total / len(order))
            msg = f"    epoch {epoch:>3}  loss {self.loss_history[-1]:.5f}"

            if val_ends is not None:
                net.eval()
                v_total = 0.0
                with torch.no_grad():
                    for i in range(0, len(val_ends), 4096):
                        batch = _gather(Xv, torch.as_tensor(val_ends[i : i + 4096], device=dev), self.window)
                        v_total += step_loss(net, batch).item() * len(batch)
                v = v_total / len(val_ends)
                self.val_loss_history.append(v)
                msg += f"  val-normal loss {v:.5f}"
                if v < best:
                    best, bad, self.best_epoch = v, 0, epoch
                    best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
            print(msg)
            if val_ends is not None and bad >= self.patience:
                print(f"    early stop; restoring epoch {self.best_epoch}")
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.model = net
        return self

    def _score_windows(self, X: np.ndarray, err_fn) -> np.ndarray:
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
        dev = self.device
        Xs = torch.tensor(X, dtype=torch.float32, device=dev)
        ends = np.arange(self.window - 1, len(X))
        self.model.eval()
        errors = []
        with torch.no_grad():
            for i in range(0, len(ends), self.batch_size * 4):
                batch = _gather(Xs, torch.as_tensor(ends[i : i + self.batch_size * 4], device=dev), self.window)
                errors.append(err_fn(self.model, batch).cpu().numpy())
        return unwindow_scores(np.concatenate(errors), len(X), window=self.window)


class LSTMAutoencoderDetector(_WindowedMSE, Detector):
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
        hidden1: int = 128,
        hidden2: int = 64,
        epochs: int = 8,
        batch_size: int = 256,
        lr: float = 1e-3,
        train_stride: int = 1,
        patience: int = 3,
        device: str | None = None,
    ):
        self.window = window
        self.hidden1 = hidden1  # outer LSTM width (encoder in / decoder out)
        self.hidden2 = hidden2  # latent width (the bottleneck both sides share)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        # consecutive windows overlap in window-1 of their rows; stride>1
        # trains on every k-th window, k times faster, at little cost
        self.train_stride = train_stride
        self.patience = patience
        self.device = device
        self.name = f"lstm_ae(w={window},h={hidden1}-{hidden2})"
        self.model = None
        self.loss_history: list[float] = []  # one average loss per epoch, for plotting

    def _build(self, n_features: int):
        import torch
        import torch.nn as nn

        class LSTMAE(nn.Module):
            """
            Encoder: LSTM(128) -> LSTM(64), the second collapses to one
            vector per window (the latent/bottleneck). Decoder: that vector
            repeated across the window -> LSTM(64) -> LSTM(128) -> a Linear
            applied at every timestep (PyTorch has no TimeDistributed
            wrapper; Linear already broadcasts over the time dimension of a
            [batch, time, features] tensor, which is the same operation).

            PyTorch's nn.LSTM always uses tanh for its internal gates (it's
            fixed in the cuDNN kernel, unlike Keras where `activation=`
            is a free parameter) -- so the ReLU from the reference
            architecture is applied explicitly to each LSTM layer's output
            here instead, which is the standard way this Keras pattern gets
            ported to PyTorch.
            """

            def __init__(self, n_features, h1, h2, window):
                super().__init__()
                self.window = window
                self.enc1 = nn.LSTM(n_features, h1, batch_first=True)
                self.enc2 = nn.LSTM(h1, h2, batch_first=True)
                self.dec1 = nn.LSTM(h2, h2, batch_first=True)
                self.dec2 = nn.LSTM(h2, h1, batch_first=True)
                self.relu = nn.ReLU()
                self.out = nn.Linear(h1, n_features)

            def forward(self, x):
                e1, _ = self.enc1(x)
                e1 = self.relu(e1)
                _, (h, _) = self.enc2(e1)
                z = self.relu(h[-1])  # latent vector, one per window

                z = z.unsqueeze(1).repeat(1, self.window, 1)
                d1, _ = self.dec1(z)
                d1 = self.relu(d1)
                d2, _ = self.dec2(d1)
                d2 = self.relu(d2)
                return self.out(d2)

        torch.manual_seed(SEED)
        return LSTMAE(n_features, self.hidden1, self.hidden2, self.window)

    def fit(self, split: Split) -> "LSTMAutoencoderDetector":
        import torch

        mse = torch.nn.MSELoss()
        return self._fit_mse(split, lambda net, b: mse(net(b), b))

    def score(self, X: np.ndarray) -> np.ndarray:
        # mean squared error over the final timestep of each window,
        # so the score aligns with the label at that same index
        return self._score_windows(
            X, lambda net, b: ((net(b)[:, -1, :] - b[:, -1, :]) ** 2).mean(dim=1)
        )


class LSTMForecastDetector(_WindowedMSE, Detector):
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

    This is the Malhotra et al. (2015) [1] / Hundman et al. (2018) [2]
    formulation, and it is the standard deep baseline in the telemetry and
    ICS anomaly detection literature.

    Fair-comparison note: still unsupervised (normal training rows only),
    still WINDOW_SIZE context, still scored at the window's last timestep, so
    its score vector aligns with every other model's against the same y_true.
    It sees one fewer timestep of context than the autoencoder because the
    last one is the target -- that is a handicap, not an advantage.

    [1] P. Malhotra, L. Vig, G. Shroff, and P. Agarwal, "Long short term memory
        networks for anomaly detection in time series," in Proc. Eur. Symp.
        Artif. Neural Netw. (ESANN), Bruges, Belgium, 2015.
    [2] K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstrom,
        "Detecting spacecraft anomalies using LSTMs and nonparametric dynamic
        thresholding," in Proc. 24th ACM SIGKDD Int. Conf. Knowl. Discov. Data
        Mining, 2018, pp. 387-395, doi: 10.1145/3219819.3219845.
    """

    supervised = False

    def __init__(
        self,
        window: int = WINDOW_SIZE,
        hidden: int = 64,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        train_stride: int = 1,
        patience: int = 3,
        device: str | None = None,
    ):
        self.window = window
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.train_stride = train_stride
        self.patience = patience
        self.device = device
        self.name = f"lstm_forecast(w={window},h={hidden})"
        self.model = None
        self.loss_history: list[float] = []

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

        mse = torch.nn.MSELoss()
        # context is everything but the last step; target is the last
        return self._fit_mse(split, lambda net, b: mse(net(b[:, :-1, :]), b[:, -1, :]))

    def score(self, X: np.ndarray) -> np.ndarray:
        # error on the window's final timestep, so the score lands on the
        # same index make_windows assigns the window's label
        return self._score_windows(
            X, lambda net, b: ((net(b[:, :-1, :]) - b[:, -1, :]) ** 2).mean(dim=1)
        )


# --------------------------------------------------------------------------
# supervised, windowed: a 1D-CNN classifier
# --------------------------------------------------------------------------

class CNNClassifierDetector(_BCETrainer, Detector):
    """
    1D convolutional classifier, supervised, over fixed-length windows.

    Architecture: three Conv1d(filters=64, kernel=3) blocks, each followed
    by BatchNorm and ReLU, then global average pooling over the window,
    dropout, and one linear output layer -- the FCN (Fully Convolutional
    Network) design of [1], the standard baseline for
    the UCR/UEA time-series classification benchmarks (e.g. FordA). That
    architecture is univariate (one input channel); this generalises it to
    WINDOW_SIZE x n_features multivariate windows by treating every sensor
    as its own input channel, exactly the extension the original authors
    flag as the natural next step.

    This is a genuinely different kind of model from everything else in the
    registry. xgboost is supervised but sees one row with no memory of
    before or after it. The two LSTM detectors see a whole window but are
    UNsupervised -- trained on normal windows only, never shown what an
    attack actually looks like. This one is supervised AND windowed: it is
    trained directly on labelled examples of both classes, window-shaped,
    which neither of the other two families gets to do.

    Classification, not regression, so the loss is binary cross-entropy
    (BCEWithLogitsLoss, the numerically-stable form that takes a raw logit)
    rather than the LSTMs' MSE. Training goes through `_BCETrainer`: see
    its docstring for the negative subsampling, AdamW, OneCycle schedule,
    clipping and validation-based best-epoch restore, and why each helps.

    [1] Z. Wang, W. Yan, and T. Oates, "Time series classification from
        scratch with deep neural networks: A strong baseline," in Proc. Int.
        Joint Conf. Neural Netw. (IJCNN), 2017, doi: 10.1109/IJCNN.2017.7966039.
    """

    supervised = True

    def __init__(
        self,
        window: int = WINDOW_SIZE,
        filters: int = 64,
        dropout: float = 0.3,
        epochs: int = 15,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        neg_ratio: float | None = 10.0,
        patience: int = 4,
        clip: float = 1.0,
        device: str | None = None,
    ):
        self.window = window
        self.filters = filters
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.neg_ratio = neg_ratio
        self.patience = patience
        self.clip = clip
        self.device = device
        self.name = f"cnn_classifier(w={window},f={filters})"
        self.model = None
        self.loss_history: list[float] = []

    def _build(self, n_features: int):
        import torch
        import torch.nn as nn

        class FCN(nn.Module):
            def __init__(self, n_features, filters, dropout):
                super().__init__()

                def block(c_in, c_out):
                    return nn.Sequential(
                        nn.Conv1d(c_in, c_out, kernel_size=3, padding="same"),
                        nn.BatchNorm1d(c_out),
                        nn.ReLU(),
                    )

                self.conv1 = block(n_features, filters)
                self.conv2 = block(filters, filters)
                self.conv3 = block(filters, filters)
                self.drop = nn.Dropout(dropout)
                self.out = nn.Linear(filters, 1)

            def forward(self, x):
                # x arrives (batch, window, n_features) -- same layout every
                # other detector here uses -- Conv1d wants channels first.
                x = x.transpose(1, 2)              # -> (batch, n_features, window)
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = x.mean(dim=-1)                 # global average pooling
                return self.out(self.drop(x)).squeeze(-1)  # one raw logit per window

        torch.manual_seed(SEED)
        return FCN(n_features, self.filters, self.dropout)

    def fit(self, split: Split) -> "CNNClassifierDetector":
        import torch

        self.device = _resolve_device(self.device)
        dev = self.device
        if split.y_train.sum() == 0:
            raise ValueError("Supervised model cannot train on a split with no anomalies.")

        # Supervised AND windowed: nothing is masked to normal-only, but
        # windows still never cross a block join. Each window is labelled by
        # its LAST timestep, same convention as every other windowed model.
        Xt = torch.tensor(split.X_train, dtype=torch.float32, device=dev)
        ends = window_ends(split.continuity_mask("train"), window=self.window)
        Xv = torch.tensor(split.X_val, dtype=torch.float32, device=dev)
        v_ends = window_ends(split.continuity_mask("val"), window=self.window)
        ends_t = torch.as_tensor(ends, device=dev)
        v_ends_t = torch.as_tensor(v_ends, device=dev)

        net = self._build(split.n_features).to(dev)
        self.model = self._fit_bce(
            net,
            lambda idx: _gather(Xt, ends_t[torch.as_tensor(idx, device=dev)], self.window),
            split.y_train[ends],
            lambda idx: _gather(Xv, v_ends_t[torch.as_tensor(idx, device=dev)], self.window),
            split.y_val[v_ends],
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("call fit() first")
        if len(X) < self.window:
            raise ValueError(
                f"{self.name}: got {len(X)} rows, need at least {self.window}."
            )
        dev = self.device
        Xs = torch.tensor(X, dtype=torch.float32, device=dev)
        ends_t = torch.arange(self.window - 1, len(X), device=dev)
        logits = self._predict_logits(
            self.model,
            lambda idx: _gather(Xs, ends_t[torch.as_tensor(idx, device=dev)], self.window),
            len(ends_t),
        )
        return unwindow_scores(torch.sigmoid(logits).cpu().numpy(), len(X), window=self.window)

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
#
# PROTOCOL STATUS. The supervised entries here (xgboost, naive_bayes,
# logistic_regression, mlp_bce, cnn_classifier, ...) train on ATTACK LABELS.
# They belong to the attack-label-trained EXTENSION, evaluated only under the
# secondary chronological / stratified / cross-validation protocols. They are
# never pooled with the A1 -> A2 primary protocol, whose detectors
# (nominal_models.py) see no attack label at all. The unsupervised lstm_ae and
# lstm_forecast are reused by the primary protocol through NominalSplit.
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
    # Six classical/shallow supervised baselines, binary, on the same
    # chronological split -- see the section above for what's an exact
    # implementation and what's a deliberate stand-in.
    "naive_bayes": NaiveBayesDetector,
    "random_forest": RandomForestSCADADetector,
    "oner": OneRDetector,
    "j48": J48Detector,
    "nnge_stub_knn": NNgeDetector,
    "svm": SVMDetector,
    # Supervised AND windowed -- a third family alongside row-independent
    # boosting and unsupervised sequence models. See CNNClassifierDetector's
    # docstring for exactly what that combination buys.
    "cnn_classifier": CNNClassifierDetector,
    # BCE-trained models, from linear to non-linear: sklearn logistic
    # regression (exact solver), the same model trained by Adam through the
    # shared torch loop, and a two-layer MLP. AdaBoost is the exponential-
    # loss boosting contrast to xgboost's log-loss (BCE) boosting.
    "logistic_regression": LogisticRegressionDetector,
    "logreg_adam": partial(BCEMLPDetector, hidden=()),
    "mlp_bce": BCEMLPDetector,
    "adaboost": AdaBoostDetector,
}

DEEP_MODELS = ("lstm_ae", "lstm_forecast")
BOOSTING_MODELS = ("xgboost", "xgboost_lags")
CLASSICAL_MODELS = (
    "naive_bayes",
    "random_forest",
    "oner",
    "j48",
    "nnge_stub_knn",
    "svm",
)
CONV_MODELS = ("cnn_classifier",)
BCE_MODELS = ("logistic_regression", "logreg_adam", "mlp_bce", "xgboost", "cnn_classifier")
