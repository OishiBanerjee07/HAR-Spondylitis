#!/usr/bin/env python3
"""In-memory hyperparameter sweep for the HYBRID physics-aware reducer.

Loads the cached feature matrix + leak-free session split written by
HAR_4_PCA.py (DUMP_FEATURES=1) and benchmarks selector-K x RF-regularisation
combos on the IDENTICAL split — no feature re-extraction, so a full grid runs in
seconds.  Reports, for each combo:

  val   = StratifiedGroupKFold(5) session-grouped CV accuracy on the TRAIN rows
          (leak-free — the honest generalisation signal)
  test  = held-out-session TEST accuracy (single split, high variance)
  gap   = train - val  (overfitting gauge)

The hybrid selector here is byte-identical in behaviour to RawFeatureSelector
(kind='hybrid') in HAR_4_PCA.py: SelectKBest(mutual_info_classif) ranks every
feature by MI, then the physics descriptors (_is_physics_feature) are taken first
in MI order (MI>0), remaining budget filled by next-best MI.
"""
import os, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.metrics import accuracy_score

CACHE = os.environ.get("DUMP_FEATURES_PATH", "generated/feature_cache.npz")
d = np.load(CACHE, allow_pickle=True)
X, y, win_session = d["X"], d["y"], d["win_session"]
tr_idx, te_idx = d["tr_idx"], d["te_idx"]
FEATURES = list(d["features"]); CLASSES = list(d["classes"])
X_train, X_test = X[tr_idx], X[te_idx]
y_train, y_test = y[tr_idx], y[te_idx]
groups_train = win_session[tr_idx]
print(f"cache: X={X.shape}  train={X_train.shape}  test={X_test.shape}  "
      f"classes={len(CLASSES)}  sessions={len(set(win_session))}")


def _is_physics_feature(name: str) -> bool:
    n = name.lower()
    return (
        "ang_disp" in n or "crossprod" in n
        or ("gyro" in n and "energy" in n) or "energy_ratio" in n or "band_ratio" in n
        or "rms_angvel" in n or ("gyro" in n and "rms" in n)
        or "corr__" in n or "xcorr" in n
        or "accel_mag" in n or "linacc_mag_var" in n
        or ("accel" in n and ("__std" in n or "__var" in n))
        or "orient" in n or "gyro_accel" in n
        or "linacc_mag_peak" in n or "p2p" in n
        or "pospeak" in n or "negpeak" in n or "__range" in n)


class Hybrid(BaseEstimator, TransformerMixin):
    def __init__(self, k=40, random_state=42):
        self.k = k; self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, np.float64)
        k = int(max(1, min(self.k, X.shape[1])))
        scores = np.nan_to_num(
            SelectKBest(mutual_info_classif, k="all").fit(X, y).scores_)
        order = np.argsort(scores)[::-1]
        phys = [i for i in order if _is_physics_feature(FEATURES[i]) and scores[i] > 1e-9]
        pset = set(phys)
        rest = [i for i in order if i not in pset]
        self.sel_ = np.asarray(sorted((phys + rest)[:k]), int)
        self.n_phys_ = int(sum(1 for i in self.sel_ if i in pset))
        return self

    def transform(self, X):
        return np.asarray(X, np.float64)[:, self.sel_]


# Tempered inverse-frequency class weights — identical to HAR_4_PCA.py so the sweep
# reproduces the deployed pipeline's numbers (class_weight is a big lever here).
_counts = np.bincount(y_train, minlength=len(CLASSES)).astype(float)
_inv = np.sqrt(_counts.sum() / (len(_counts) * np.maximum(_counts, 1)))
_inv = _inv / _inv.mean()
CLASS_WEIGHT = {i: float(w) for i, w in enumerate(_inv)}


def rf(depth, leaf, trees, split):
    return RandomForestClassifier(
        n_estimators=trees, max_depth=depth, min_samples_leaf=leaf,
        min_samples_split=split, max_features="sqrt", class_weight=CLASS_WEIGHT,
        random_state=42, n_jobs=-1)


def evaluate(k, depth, leaf, trees=18, split=4):
    pipe = Pipeline([("sc", StandardScaler()),
                     ("sel", Hybrid(k=k)),
                     ("clf", rf(depth, leaf, trees, split))])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    val = float(np.mean(cross_validate(pipe, X_train, y_train, groups=groups_train,
                                       cv=cv, scoring="accuracy", n_jobs=-1)["test_score"]))
    pipe.fit(X_train, y_train)
    tr = accuracy_score(y_train, pipe.predict(X_train))
    te = accuracy_score(y_test, pipe.predict(X_test))
    nphys = pipe.named_steps["sel"].n_phys_
    return val, te, tr, nphys


GRID = [
    # (k, depth, leaf, trees)  — higher K clearly helps; explore the top end + trees/depth
    (40,  8, 2, 18),    # current deployed baseline (should reproduce full-run ~0.29 val)
    (60,  8, 2, 30),
    (80,  8, 2, 40),
    (100, 8, 2, 40),
    (120, 8, 2, 60),
    (80, 10, 2, 60),
    (100,10, 2, 80),
    (120,10, 3, 80),
    (100,12, 2, 80),
    (127,10, 2, 100),   # int8 cap
]
extra = os.environ.get("GRID_EXTRA")
if extra:  # "k,depth,leaf,trees; k,depth,leaf,trees"
    for row in extra.split(";"):
        GRID.append(tuple(int(x) for x in row.split(",")))

print(f"\n{'k':>3} {'depth':>5} {'leaf':>4} {'trees':>5} | {'val':>6} {'test':>6} "
      f"{'train':>6} {'gap':>6} {'nphys':>5}")
print("-" * 62)
results = []
for (k, depth, leaf, trees) in GRID:
    val, te, tr, nphys = evaluate(k, depth, leaf, trees)
    gap = tr - val
    results.append((val, te, k, depth, leaf, trees, gap, nphys))
    print(f"{k:>3} {depth:>5} {leaf:>4} {trees:>5} | {val:>6.3f} {te:>6.3f} "
          f"{tr:>6.3f} {gap:>6.3f} {nphys:>5}")

print("\nBy CV val (honest, leak-free):")
for r in sorted(results, key=lambda r: -r[0])[:3]:
    print(f"  val={r[0]:.3f} test={r[1]:.3f}  k={r[2]} depth={r[3]} leaf={r[4]} trees={r[5]} gap={r[6]:.3f}")
print("By held-out-session test:")
for r in sorted(results, key=lambda r: -r[1])[:3]:
    print(f"  test={r[1]:.3f} val={r[0]:.3f}  k={r[2]} depth={r[3]} leaf={r[4]} trees={r[5]} gap={r[6]:.3f}")
