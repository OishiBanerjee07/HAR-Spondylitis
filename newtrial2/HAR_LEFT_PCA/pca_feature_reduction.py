"""
PCA-based feature reduction with AUTO-K via the elbow / scree-plot method.

This is a self-contained, leak-aware module that complements the *supervised*
top-K PCA already in HAR_4.py (TopPCAComponents, which ranks components by
RandomForest importance).  Here the selection is purely UNSUPERVISED and driven
by the variance geometry of the scree curve:

  1. Fit PCA on (standardised) features.
  2. Auto-select K from the scree plot using the Kneedle "max-distance-to-chord"
     elbow on the per-component explained-variance-ratio curve.
  3. Rank the kept components by explained variance ratio (highest first).
  4. Rank the ORIGINAL features by the magnitude of their loadings across the
     selected K components (variance-weighted, so high-variance PCs count more).
  5. Emit a scree plot with the elbow marked, a feature-importance chart, and a
     JSON record documenting cumulative explained variance + per-feature scores.

Design notes
------------
* Reproducible: `random_state` is threaded into PCA.
* Leak-safe usage: fit on TRAIN ONLY (pass X_train), then `.transform(X_test)`.
  The returned object is a fitted transformer you can reuse.
* No third-party knee library is required — the elbow finder is implemented here.

Integrate into HAR_4.py with:

    from pca_feature_reduction import run_pca_feature_reduction
    result = run_pca_feature_reduction(X_train, FEATURES)     # fit on train only
    X_train_red = result.transform(X_train)
    X_test_red  = result.transform(X_test)

or run this file directly for a synthetic-data smoke test.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field

import numpy as np
import matplotlib
matplotlib.use("Agg")               # headless-safe (no display needed)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ── ELBOW / KNEE DETECTION ────────────────────────────────────────────────────
def find_elbow(values: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Kneedle-style elbow on a monotonically DECREASING convex curve (the scree
    plot's per-component explained-variance-ratio).

    The knee is the point of maximum perpendicular distance from the straight
    chord joining the first and last points of the (min-max normalised) curve —
    i.e. where the curve stops dropping steeply and flattens out.

    Returns
    -------
    k     : 1-based index of the elbow component (== number of components to keep)
    dist  : the per-point distance array (handy for plotting / debugging)
    """
    y = np.asarray(values, dtype=float)
    n = y.size
    if n <= 2:                      # nothing to bend — keep everything
        return n, np.zeros(n)

    x = np.arange(1, n + 1, dtype=float)
    # Min-max normalise both axes so x- and y-scales are comparable.
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())

    p1 = np.array([xn[0], yn[0]])
    p2 = np.array([xn[-1], yn[-1]])
    chord = p2 - p1
    chord_len = np.hypot(*chord) + 1e-12
    # Perpendicular distance of every point to the p1→p2 chord.
    dist = np.abs(chord[0] * (p1[1] - yn) - (p1[0] - xn) * chord[1]) / chord_len
    k = int(np.argmax(dist)) + 1    # +1 → 1-based component count
    return max(k, 1), dist


# ── RESULT CONTAINER ──────────────────────────────────────────────────────────
@dataclass
class PCAReductionResult:
    """Fitted reducer + everything the task asks us to preserve/report."""
    k: int
    feature_names: list[str]
    scaler: StandardScaler
    pca: PCA
    explained_variance_ratio: np.ndarray          # all components (var-ordered)
    cumulative_variance: np.ndarray               # cumulative, all components
    cumulative_variance_topk: float               # variance captured by top-K
    component_ranking: list[dict] = field(default_factory=list)   # top-K, by EVR
    feature_importance: list[dict] = field(default_factory=list)  # by loading mag
    elbow_distances: np.ndarray = None

    # ---- reuse as a transformer (leak-safe: scaler+PCA fit on train only) ----
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto the selected top-K components."""
        Xs = self.scaler.transform(np.asarray(X, dtype=np.float64))
        return self.pca.transform(Xs)[:, : self.k]


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────
def run_pca_feature_reduction(
    X: np.ndarray,
    feature_names: list[str],
    *,
    k: int | None = None,                # None → auto-select via elbow method
    variance_cap: float = 0.99,          # don't auto-pick beyond this cum. variance
    top_features_chart: int = 25,        # how many features to show on the chart
    random_state: int = 42,
    out_dir: str = "generated",
    plot_dir: str = "plots",
    prefix: str = "pca_elbow",
    verbose: bool = True,
) -> PCAReductionResult:
    """
    Fit PCA, auto-select K (elbow), rank components by variance and features by
    loading magnitude, and write the scree plot + feature-importance chart.

    Parameters
    ----------
    X              : (n_samples, n_features) numeric matrix.
    feature_names  : list of the original feature names (len == n_features).
    k              : force a specific K; None auto-selects via the elbow method.
    variance_cap   : safety ceiling so the elbow K can't exceed the #components
                     needed to reach this cumulative variance.
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names ({len(feature_names)}) must match X columns ({n_features})")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Standardise (PCA is variance-based → must be on a common scale) + fit PCA.
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    max_comp = min(n_samples, n_features)
    pca = PCA(n_components=max_comp, random_state=random_state).fit(Xs)

    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)

    # 2. Auto-select K from the scree elbow (unless K is pinned).
    elbow_k, dist = find_elbow(evr)
    cap_k = int(np.searchsorted(cumvar, variance_cap) + 1)   # #comps for variance_cap
    if k is None:
        k = int(min(elbow_k, cap_k, max_comp))
        k = max(k, 1)
        k_source = f"elbow (knee={elbow_k}, variance-cap={cap_k} @ {variance_cap:.0%})"
    else:
        k = int(min(k, max_comp))
        k_source = "user-specified"

    cum_topk = float(cumvar[k - 1])

    # 3. Rank the kept components by explained variance ratio (already var-ordered,
    #    so component 0 is highest — we record it explicitly for the output).
    comp_ranking = [
        {"component": int(i),
         "explained_variance_ratio": float(evr[i]),
         "cumulative_variance": float(cumvar[i])}
        for i in range(k)
    ]

    # 4. Rank ORIGINAL features by loading magnitude across the top-K components.
    #    Weight each component's |loadings| by its explained variance ratio so a
    #    feature that drives a high-variance PC outranks one driving a tail PC.
    loadings = pca.components_[:k]                       # (k, n_features)
    abs_load = np.abs(loadings)
    weights = evr[:k] / (evr[:k].sum() + 1e-12)         # renormalised over kept PCs
    importance = (weights[:, None] * abs_load).sum(axis=0)   # (n_features,)
    importance = importance / (importance.sum() + 1e-12)     # → fractions sum to 1
    order = np.argsort(importance)[::-1]
    feat_importance = [
        {"name": feature_names[i],
         "importance": float(importance[i]),
         # the single component this feature loads onto most strongly, for context
         "dominant_component": int(np.argmax(abs_load[:, i])),
         "max_abs_loading": float(abs_load[:, i].max())}
        for i in order
    ]

    result = PCAReductionResult(
        k=k,
        feature_names=list(feature_names),
        scaler=scaler,
        pca=pca,
        explained_variance_ratio=evr,
        cumulative_variance=cumvar,
        cumulative_variance_topk=cum_topk,
        component_ranking=comp_ranking,
        feature_importance=feat_importance,
        elbow_distances=dist,
    )

    # ── OUTPUT: scree plot with the elbow marked ──────────────────────────────
    scree_path = os.path.join(plot_dir, f"{prefix}_scree.png")
    _plot_scree(evr, cumvar, k, elbow_k, cap_k, variance_cap, scree_path)

    # ── OUTPUT: feature-importance chart (by loading magnitude) ───────────────
    chart_path = os.path.join(plot_dir, f"{prefix}_feature_importance.png")
    _plot_feature_importance(feat_importance, top_features_chart, k, cum_topk, chart_path)

    # ── OUTPUT: JSON documentation ────────────────────────────────────────────
    json_path = os.path.join(out_dir, f"{prefix}_reduction.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "k_selected": int(k),
            "k_source": k_source,
            "elbow_k": int(elbow_k),
            "variance_cap": variance_cap,
            "variance_cap_k": int(cap_k),
            "cumulative_variance_topk": cum_topk,
            "random_state": random_state,
            "component_ranking_by_variance": comp_ranking,
            "feature_importance_by_loading": feat_importance,
        }, fh, indent=2)

    if verbose:
        print("=" * 68)
        print("PCA FEATURE REDUCTION — elbow / scree auto-K")
        print("=" * 68)
        print(f"  Input                : {n_samples} samples × {n_features} features")
        print(f"  K selected           : {k}   [{k_source}]")
        print(f"  Cumulative variance  : {cum_topk * 100:.2f}% captured by top-{k} PCs")
        print(f"  Top components (EVR) : "
              + ", ".join(f"PC{c['component']}={c['explained_variance_ratio']*100:.1f}%"
                          for c in comp_ranking[:min(k, 6)])
              + (" …" if k > 6 else ""))
        print(f"  Top features (by |loading|):")
        for f in feat_importance[:10]:
            bar = "#" * int(round(f["importance"] / feat_importance[0]["importance"] * 30))
            print(f"    {f['name']:<30} {f['importance']*100:5.2f}%  {bar}")
        print(f"  Scree plot           : {scree_path}")
        print(f"  Importance chart     : {chart_path}")
        print(f"  JSON record          : {json_path}")

    return result


# ── PLOT HELPERS ──────────────────────────────────────────────────────────────
def _plot_scree(evr, cumvar, k, elbow_k, cap_k, variance_cap, path):
    n = len(evr)
    x = np.arange(1, n + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Per-component explained variance (the scree curve the elbow is read from).
    ax1.bar(x, evr * 100, color="#9ecae1", label="Explained variance (per comp.)")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance (%)", color="#3182bd")
    ax1.tick_params(axis="y", labelcolor="#3182bd")

    # Cumulative variance on a twin axis.
    ax2 = ax1.twinx()
    ax2.plot(x, cumvar * 100, "o-", color="#e6550d", lw=2, ms=3,
             label="Cumulative variance")
    ax2.set_ylabel("Cumulative variance (%)", color="#e6550d")
    ax2.tick_params(axis="y", labelcolor="#e6550d")
    ax2.set_ylim(0, 105)

    # Mark the elbow / selected K.
    ax1.axvline(k, color="green", ls="--", lw=2,
                label=f"selected K={k} ({cumvar[k-1]*100:.1f}% var)")
    ax1.scatter([elbow_k], [evr[elbow_k - 1] * 100], color="green", zorder=5, s=80,
                marker="D", label=f"elbow knee @ {elbow_k}")
    ax2.axhline(variance_cap * 100, color="grey", ls=":", lw=1,
                label=f"{variance_cap:.0%} variance cap (K≤{cap_k})")

    # Combined legend.
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)
    ax1.set_title("PCA Scree Plot with Elbow-selected K", fontsize=13)
    ax1.set_xlim(0.5, min(n, max(k * 3, 30)) + 0.5)   # zoom to the informative head
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_feature_importance(feat_importance, top_n, k, cum_topk, path):
    top = feat_importance[:top_n]
    names = [f["name"] for f in top][::-1]          # reverse → highest on top
    vals = [f["importance"] * 100 for f in top][::-1]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(top) + 1)))
    bars = ax.barh(names, vals, color=plt.cm.viridis(np.linspace(0.15, 0.9, len(top))))
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
    ax.set_xlabel("Importance  (variance-weighted Σ |loading|, %)")
    ax.set_title(f"Top {len(top)} Features by PCA Loading Magnitude\n"
                 f"(across top-{k} components — {cum_topk*100:.1f}% variance)",
                 fontsize=12)
    ax.margins(x=0.12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── STANDALONE SMOKE TEST ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic dataset: a handful of informative directions + noise, so the scree
    # curve has a clear elbow we can confirm the detector lands on.
    rng = np.random.default_rng(42)
    n, p, latent = 800, 40, 5
    Z = rng.standard_normal((n, latent))
    mixing = rng.standard_normal((latent, p))
    X_demo = Z @ mixing + 0.25 * rng.standard_normal((n, p))
    names = [f"feat_{i:02d}" for i in range(p)]

    res = run_pca_feature_reduction(X_demo, names, prefix="pca_elbow_demo")
    print(f"\nReduced shape: {res.transform(X_demo).shape} (from {X_demo.shape})")
