#!/usr/bin/env python3
"""
Repair / QA for the PCA stage of the engineered IMU feature pipeline.

WHY THIS EXISTS
---------------
Symptom reported: the "principal components" were ~perfectly correlated with one
another and the loading matrix looked like an IDENTITY map (a single 1.0 loading
on one feature, e.g. an energy metric).  Genuine PCA can never do that — its
components are orthonormal by construction and its scores are uncorrelated.  That
pattern appears for exactly two reasons:

  (a) the matrix was fed to PCA WITHOUT standardising first, so a few high-variance
      columns (raw energy terms dwarf everything) dominate every component, and/or
  (b) what was being inspected was not PCA at all but a one-hot FEATURE SELECTOR
      (rf_importance) whose "components_" really are identity rows — its loadings
      are 1.0 on a single feature by design.

This script does PCA the correct way and proves the components come out orthogonal
and multi-feature:

  1. load the raw engineered feature matrix          (accepts .pkl / .csv / .npy)
  2. drop near-zero-variance (constant) features      (VarianceThreshold)
  3. StandardScaler  -> zero mean, unit variance
  4. PCA             -> orthogonal components, verified by a correlation heatmap
  5. cumulative explained-variance ratio -> #PCs for 95%
  6. top-5 original-feature loadings for the first 3 PCs (multi-feature check)

USAGE
-----
  python pca_fix.py                              # uses generated/feature_cache.pkl
  python pca_fix.py --features path/to/X.csv     # any CSV (numeric feature cols)
  python pca_fix.py --features X.npy --variance-target 0.95
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys

import numpy as np

# Windows consoles default to cp1252 and choke on the box-drawing / check glyphs.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ── 1. LOAD THE RAW ENGINEERED FEATURE MATRIX ────────────────────────────────
_NON_FEATURE_COLS = {
    "activity_label", "activity_id", "label", "y", "target",
    "device_id", "device_encoded", "session_id", "participant_id", "timestamp",
}


def load_feature_matrix(path):
    """Return (X float64 [n, d], feature_names list[str]).

    Supports:
      • .pkl  – the project's generated/feature_cache.pkl  (dict with X / FEATURES)
      • .csv  – numeric feature columns (non-feature/meta columns are dropped)
      • .npy  – a plain (n, d) array (names auto-generated)
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pkl":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, dict) and "X" in obj:
            X = np.asarray(obj["X"], dtype=np.float64)
            names = list(obj.get("FEATURES") or [f"feat_{i}" for i in range(X.shape[1])])
        else:                                           # a bare array pickled
            X = np.asarray(obj, dtype=np.float64)
            names = [f"feat_{i}" for i in range(X.shape[1])]

    elif ext == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        feat_cols = [c for c in df.columns
                     if c not in _NON_FEATURE_COLS
                     and np.issubdtype(df[c].dtype, np.number)]
        X = df[feat_cols].to_numpy(dtype=np.float64)
        names = list(feat_cols)

    elif ext == ".npy":
        X = np.asarray(np.load(path), dtype=np.float64)
        names = [f"feat_{i}" for i in range(X.shape[1])]

    else:
        raise ValueError(f"unsupported feature-matrix extension: {ext!r}")

    # PCA cannot run on NaN/Inf; scrub defensively (the engineered matrix is finite,
    # but a CSV round-trip or a degenerate window can introduce them).
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {X.shape}")
    return X, names


# ── 2. DROP NEAR-ZERO-VARIANCE FEATURES ──────────────────────────────────────
def drop_low_variance(X, names, threshold):
    """Remove (near-)constant columns.  A feature with ~0 variance carries no
    information, inflates the condition number, and lets StandardScaler divide by an
    almost-zero std -> the exact collinearity that sabotages PCA + the RF."""
    vt = VarianceThreshold(threshold=threshold).fit(X)
    keep = vt.get_support()
    dropped = [n for n, k in zip(names, keep) if not k]
    X_kept = X[:, keep]
    kept_names = [n for n, k in zip(names, keep) if k]
    print(f"[2] Near-zero-variance cleaning (threshold={threshold:g})")
    print(f"      features in : {X.shape[1]}")
    print(f"      dropped     : {len(dropped)}"
          + (f"  e.g. {dropped[:6]}" if dropped else ""))
    print(f"      features out: {X_kept.shape[1]}")
    return X_kept, kept_names


# ── 4 (verify). ORTHOGONALITY CHECK + HEATMAP ────────────────────────────────
def verify_orthogonal(scores, out_png, n_keep, n_show=15):
    """Correlation matrix of the PCA SCORES.  Correct PCA => identity (off-diagonal
    ~0).  We check only the n_keep *retained* components: the discarded tail PCs have
    near-zero variance, so Pearson r (which divides by their std~0) is numerically
    meaningless there — including them would invent spurious correlations and is NOT
    evidence of non-orthogonality (the loading vectors are exactly orthonormal; see
    the Gram check in run()).  Returns max off-diagonal |corr| over the kept PCs."""
    Zk = scores[:, :max(2, n_keep)]
    corr = np.corrcoef(Zk, rowvar=False)
    off = corr - np.eye(corr.shape[0])
    max_off = float(np.abs(off).max()) if corr.shape[0] > 1 else 0.0

    k = min(n_show, corr.shape[0])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr[:k, :k], cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(f"PCA score correlation (first {k} of {Zk.shape[1]} retained PCs)\n"
                 f"max off-diagonal |corr| = {max_off:.2e}  (≈0 ⇒ orthogonal)")
    ax.set_xlabel("component"); ax.set_ylabel("component")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=130); plt.close(fig)
    return max_off, out_png


def run(path, variance_target=0.95, var_threshold=1e-8,
        heatmap_png="plots/pca_component_corr_heatmap.png"):
    print("=" * 70)
    print("PCA PIPELINE — repaired (clean -> scale -> PCA -> verify)")
    print("=" * 70)

    # 1 ────────────────────────────────────────────────────────────────────────
    X, names = load_feature_matrix(path)
    print(f"[1] Loaded feature matrix: {X.shape[0]} samples x {X.shape[1]} features"
          f"  ({os.path.normpath(path)})")

    # 2 ────────────────────────────────────────────────────────────────────────
    X, names = drop_low_variance(X, names, var_threshold)

    # 3 ── StandardScaler: THE fix.  Without this, raw *energy* columns (variance
    #      orders of magnitude larger than, say, a correlation in [-1, 1]) hijack
    #      every component, which is what made the loadings look like identity rows.
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    print("[3] StandardScaler applied -> per-feature mean~0, std~1")
    print(f"      mean abs(mean) = {np.abs(Xs.mean(0)).max():.2e}   "
          f"max|std-1| = {np.abs(Xs.std(0) - 1).max():.2e}")

    # 4 ── PCA on the SCALED matrix.  Components are orthonormal by construction;
    #      we keep the full rank here so the 95% search (step 5) sees every PC.
    pca = PCA(svd_solver="full", random_state=42).fit(Xs)
    scores = pca.transform(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_target = int(np.searchsorted(cum, variance_target) + 1)

    # Verify orthogonality on the RETAINED components (see verify_orthogonal docstring
    # for why the near-zero-variance tail is excluded), and prove the loading vectors
    # are exactly orthonormal via the Gram matrix CᵀC ?= I (the rigorous test).
    max_off, png = verify_orthogonal(scores, heatmap_png, n_keep=n_target)
    gram = pca.components_ @ pca.components_.T
    orthonormal_err = float(np.abs(gram - np.eye(gram.shape[0])).max())
    ok = max_off < 1e-6 and orthonormal_err < 1e-6
    print(f"[4] PCA fit: {pca.n_components_} components")
    print(f"      retained-PC score max off-diagonal |corr| = {max_off:.2e}")
    print(f"      loading orthonormality error  CᵀC vs I     = {orthonormal_err:.2e}")
    print(f"      components genuinely orthogonal: {'PASS ✓' if ok else 'FAIL ✗'}")
    print(f"      correlation heatmap saved -> {os.path.normpath(png)}")

    # 5 ── cumulative explained variance -> #PCs for the target.
    print(f"[5] Cumulative explained variance")
    print(f"      components for {variance_target*100:.0f}% variance: {n_target}"
          f"  (of {pca.n_components_})")
    for frac in (0.80, 0.90, 0.95, 0.99):
        n = int(np.searchsorted(cum, frac) + 1)
        print(f"        {frac*100:4.0f}% -> {n:3d} PCs")

    # Scree / cumulative-variance plot.
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, len(cum) + 1), cum * 100, "o-", ms=3, color="steelblue")
    ax.axhline(variance_target * 100, color="red", ls="--",
               label=f"{variance_target*100:.0f}% target")
    ax.axvline(n_target, color="green", ls="--", label=f"n={n_target}")
    ax.set_xlabel("number of components"); ax.set_ylabel("cumulative variance (%)")
    ax.set_title("PCA cumulative explained variance"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    scree_png = os.path.join(os.path.dirname(heatmap_png) or ".", "pca_scree_fixed.png")
    fig.savefig(scree_png, dpi=130); plt.close(fig)
    print(f"      scree plot saved -> {os.path.normpath(scree_png)}")

    # 6 ── top-5 original-feature loadings for the first 3 PCs.  A healthy PC blends
    #      MANY features (no single 1.0); we print each loading + that PC's variance.
    print("[6] Top-5 original-feature loadings (first 3 PCs)")
    for i in range(min(3, pca.n_components_)):
        load = pca.components_[i]
        top = np.argsort(np.abs(load))[::-1][:5]
        share = pca.explained_variance_ratio_[i] * 100
        # how spread is this PC?  (participation ratio ~ effective #features)
        pr = (load @ load) ** 2 / (np.sum(load ** 4) + 1e-300)
        print(f"  PC{i+1}  (explains {share:5.2f}% var,  ~{pr:.0f} effective features)")
        for j in top:
            bar = "█" * int(round(abs(load[j]) * 40))
            print(f"      {load[j]:+.4f}  {names[j]:<34} {bar}")
    print("=" * 70)
    print("Done.  Scaled-then-PCA produces orthogonal, multi-feature components —")
    print("feed `scores[:, :n_target]` (or a Pipeline of the three steps) to the RF.")
    return pca, n_target


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="generated/feature_cache.pkl",
                    help="path to the engineered feature matrix (.pkl/.csv/.npy)")
    ap.add_argument("--variance-target", type=float, default=0.95,
                    help="cumulative variance to retain (default 0.95)")
    ap.add_argument("--var-threshold", type=float, default=1e-8,
                    help="drop features whose variance is <= this (default 1e-8)")
    ap.add_argument("--heatmap", default="plots/pca_component_corr_heatmap.png")
    args = ap.parse_args()

    if not os.path.exists(args.features):
        sys.exit(f"feature matrix not found: {args.features}\n"
                 f"  run HAR_4.py once to populate generated/feature_cache.pkl, "
                 f"or pass --features <path-to-your-matrix>")
    run(args.features, args.variance_target, args.var_threshold, args.heatmap)


if __name__ == "__main__":
    main()
