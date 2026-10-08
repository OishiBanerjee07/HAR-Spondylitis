#!/usr/bin/env python3
"""
HAR_ROT_HIER.py — 12-way (Body x Direction x Side) arm-rotation HAR trainer
(merged spec v2).  Hierarchical multi-head (default) + flat baseline.

RELATIONSHIP TO THE EXISTING MODEL
----------------------------------
This REPLACES the flat locomotion model (HAR_4_PCA.py) for the rotational task.
Carried over UNCHANGED (do not re-litigate): RF hyperparameters (18/8/7/4/sqrt),
window=100/step=50, the 79+66+UCI feature layers + exact-duplicate dedup (in
har_rot_features.py), tempered class weights, session-grouped StratifiedGroupKFold
+ held-out-session test, the reducer shared contract + per-fold leak-free fitting,
and the int8 feature-index cap (127) for the C export.

CHANGED for this task (see the analysis handed to the user):
  * flat single forest      -> 3 independent heads (Body/Direction/Side) + soft
                               combine; flat kept as a BASELINE only.
  * PRIORITIZE_MAGNITUDE... -> PROTECTED_DIRECTION_FEATURES (signed features are
                               first-class for CW/CCW; magnitude cannot separate it)
  * PCA default reducer     -> DEMOTED to a candidate; expect selection reducers to
                               win.  New candidate: mi_rfe_hybrid.
  * single-wrist mirror aug -> real dual-wrist synchronised windows + canonical
                               left-wrist remap (Head S needs both real devices)
  * locomotion/static block -> DROPPED (rotational task)

WHAT NEEDS YOUR DATASET (prompt §12 — do NOT guess): everything in the
`DATASET CONFIG` block below (column names, session/device keys, the label->factor
parser, device side values, rest threshold, flash budget).  The engine below the
config block is fixed.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, re, itertools
# ── BLAS THREAD CAP (must be set BEFORE numpy/scipy/sklearn import) ────────────
# The per-head benchmark runs cross_val_predict(n_jobs=-1); each worker process then
# had OpenBLAS spawn one thread PER CORE, and the per-thread allocations multiply
# across workers on the 20k×598 matrix → "OpenBLAS: Memory allocation still failed
# after 10 retries" and killed workers (TerminatedWorkerError → 'no reducer
# succeeded for head body').  Pinning each process to a single BLAS thread removes
# the explosion; joblib still parallelises across worker PROCESSES, so results are
# byte-identical — only thread fan-out changes.  (Same guard HAR_4_PCA.py carries.)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import har_rot_features as F

os.makedirs("generated", exist_ok=True)
os.makedirs("plots",     exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# ░░░  DATASET CONFIG  ░░░  — the ONLY block to edit when the dataset lands  ░░░
# ══════════════════════════════════════════════════════════════════════════════
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DATASET_CANDIDATES = [
    os.path.join(_here, "ROT_activities_master.csv"),
]

# --- column names / order (raw contract = accel_x..gyro_z; rename here if different)
COL_ACCEL = ["accel_x", "accel_y", "accel_z"]
COL_GYRO  = ["gyro_x",  "gyro_y",  "gyro_z"]
COL_TIME  = "timestamp"          # per-sample time (for the nearest-sample dual join)
COL_LABEL = "activity_label"
COL_SESSION = "session_id"       # leak-free CV grouping key
COL_DEVICE  = "device_id"        # LEFT/RIGHT device stream key
COL_PARTICIPANT = "participant"  # for the participant x class coverage table (opt.)

DEVICE_LEFT_VALUES  = {"LEFT", "L"}
DEVICE_RIGHT_VALUES = {"RIGHT", "R"}

# --- sampling / units (50 Hz confirmed). Gyro units drive the Madgwick conversion.
F.GYRO_IN_DEG = True             # set False if the dataset gyro is already rad/s

# --- LABEL -> (body, direction, side) parser.  Classes differ from the named 12,
#     so factor membership is keyword-driven and fully editable here.  A factor
#     that returns None for EVERY class disables that head automatically.
# DATASET NOTE: this dataset's labels follow {SIDE}_{BODY}_{DIR} with SIDE in B/L/R,
# BODY in HAND/SHOU, DIR in CLK/ACLK (e.g. "B_SHOU_ACLK", "L_HAND_CLK").  The default
# keywords ("shoulder"/"both"/"left"/"right") never matched this vocabulary, so Body
# collapsed to {hand, nan} and Side to all-None — disabling those two heads and
# crashing the combined-label split.  The keyword lists below match the ACTUAL tokens:
#   body 'shou'/'hand', side 'b_'/'l_'/'r_' prefixes (labels lowercased before match).
# 'aclk' is still checked before 'clk' (FACTOR_PRIORITY) so "..._aclk" is not caught by
# the "clk" substring it contains.  The 'b_'/'l_'/'r_' prefixes only ever appear at the
# start of a label, so there is no spurious cross-match between the three sides.
FACTOR_KEYWORDS = {
    "body":      {"shoulder": ["shou", "shoulder"], "hand": ["hand", "wrist"]},
    "direction": {"clk": ["clk", "clockwise", "_cw", " cw"],
                  "aclk": ["aclk", "anti", "counter", "ccw", "anticlock"]},
    "side":      {"both": ["b_", "both"], "left": ["l_", "left"], "right": ["r_", "right"]},
}
# 'aclk' keywords are checked BEFORE 'clk' so "anticlockwise" is not caught by "clk".
FACTOR_PRIORITY = {"direction": ["aclk", "clk"]}

def parse_label(label: str) -> dict:
    """Map a raw class label to its three rotational factors (or None per factor)."""
    s = str(label).strip().lower()
    out = {}
    for factor, classes in FACTOR_KEYWORDS.items():
        order = FACTOR_PRIORITY.get(factor, list(classes.keys()))
        order = order + [c for c in classes if c not in order]
        val = None
        for cls in order:
            if any(kw in s for kw in classes[cls]):
                val = cls; break
        out[factor] = val
    return out

# --- Head S rest threshold source (active-device count).  None -> learn the 20th
#     percentile of training-window RMS ang-vel (prompt §4 Group S).
REST_THRESHOLD = None

# --- canonical LEFT-wrist remap (OPEN item — depends on physical mounting).  The
#     default mirrors about X; validate with the canonical self-check (§2 / §9).
F.LEFT_CANONICAL_REMAP = [(0, -1), (1, 1), (2, 1), (3, 1), (4, -1), (5, -1)]

# --- embedded target
C_EXPORT        = True
FLASH_BUDGET_KB = 2048           # ESP32 flash budget (drives the summary only)
# ══════════════════════════════════════════════════════════════════════════════
# ░░░  END DATASET CONFIG  —  fixed engine below  ░░░
# ══════════════════════════════════════════════════════════════════════════════

# ── RF hyperparameters (PARITY with the probe RF — do NOT change silently) ─────
RF_KW = dict(n_estimators=18, max_depth=8, min_samples_leaf=7,
             min_samples_split=4, max_features="sqrt", random_state=42, n_jobs=-1)

CLASS_WEIGHT_MODE = "tempered"   # none | balanced | tempered(default)

# ── Reducer policy (spec v2 §5) ───────────────────────────────────────────────
PRIORITIZE_MAGNITUDE_FEATURES = False    # RETIRED — magnitude can't separate CW/CCW
FEATURE_SELECTOR = "auto"                # benchmark per head + flat; deploy winner
# PCA is DEMOTED to a candidate (sign indeterminacy mixes signed direction axes).
# Selection-shape reducers are expected to win.  mi_rfe_hybrid is the new candidate.
HEAD_SELECTORS = ["rf_importance", "mi_rfe_hybrid", "rfe", "mutual_info", "pca"]
# Per-head feature budgets (starting points; tuned by the benchmark)
HEAD_BUDGET = {"body": 24, "direction": 16, "side": 16}
FLAT_BUDGET = 64
FEATURE_INDEX_CAP = 127          # int8 gather-index cap for the C export

# PROTECTED_DIRECTION_FEATURES: the Group-D signed features force-included in Head D
# (and the flat candidate set) and only ever droppable via RFE, never by a
# magnitude/variance heuristic.  Prefix "D" is the per-window Group-D namespace.
PROTECTED_DIRECTION_FEATURES = set(F.group_D_feature_names("D"))


def make_class_weight(y):
    if CLASS_WEIGHT_MODE == "none":
        return None
    if CLASS_WEIGHT_MODE == "balanced":
        return "balanced"
    counts = np.bincount(y, minlength=int(y.max()) + 1).astype(float)
    inv = np.sqrt(counts.sum() / (len(counts) * np.maximum(counts, 1)))
    inv = inv / inv.mean()
    return {i: float(w) for i, w in enumerate(inv)}


# ══════════════════════════════════════════════════════════════════════════════
# REDUCERS — shared contract (mean_, components_, transform==(X-mean_)@comp.T,
# n_components_, orig_index_, selector_kind_, reducer_shape_).  Fit inside the
# pipeline per fold -> leak-free.
# ══════════════════════════════════════════════════════════════════════════════
class TopPCAComponents(BaseEstimator, TransformerMixin):
    """Supervised top-K PCA (DEMOTED candidate).  Variance pool -> top-K by RF
    importance.  Projection shape."""
    def __init__(self, k=20, variance_target=0.99, random_state=42):
        self.k = k; self.variance_target = variance_target
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, np.float64)
        sc = PCA(svd_solver="full").fit(X)
        cum = np.cumsum(sc.explained_variance_ratio_)
        pool = int(np.searchsorted(cum, self.variance_target) + 1)
        pool = max(1, min(pool, X.shape[1]))
        Z = sc.transform(X)[:, :pool]
        probe = RandomForestClassifier(**RF_KW).fit(Z, y)
        order = np.argsort(probe.feature_importances_)[::-1]
        keep = order[:min(self.k, pool)]
        self.mean_ = sc.mean_.astype(np.float64)
        self.components_ = sc.components_[keep].astype(np.float64)
        self.orig_index_ = keep.astype(int)
        self.explained_variance_ratio_ = sc.explained_variance_ratio_[keep]
        self.n_components_ = len(keep)
        self.selector_kind_ = "pca"; self.reducer_shape_ = "projection"
        return self

    def transform(self, X):
        return (np.asarray(X, np.float64) - self.mean_) @ self.components_.T


class RawFeatureSelector(BaseEstimator, TransformerMixin):
    """Top-K RAW feature selection (rf_importance | mutual_info | mrmr | rfe |
    mi_rfe_hybrid).  Selection shape (C export = index gather, no float matrix).
    Force-includes `protected` feature indices before the k-budget fills; those may
    only be dropped by RFE, never by a magnitude/variance heuristic."""
    def __init__(self, kind, k, feature_names=None, protected=None,
                 class_weight=None, random_state=42):
        # sklearn clone requires params be stored VERBATIM — do not transform here.
        self.kind = kind; self.k = k
        self.feature_names = feature_names
        self.protected = protected
        self.class_weight = class_weight
        self.random_state = random_state

    def _probe_rf(self):
        kw = dict(RF_KW); kw["class_weight"] = self.class_weight
        return RandomForestClassifier(**kw)

    def _protected_idx(self, n_feat):
        prot = self.protected or set()
        if not self.feature_names or not prot:
            return []
        return [i for i, nm in enumerate(self.feature_names)
                if nm in prot and i < n_feat]

    def _mrmr_select(self, X, y, k):
        n_feat = X.shape[1]; k = int(max(1, min(k, n_feat)))
        rel = np.nan_to_num(mutual_info_classif(X, y, random_state=self.random_state))
        Xc = X - X.mean(0); sd = Xc.std(0); sd[sd < 1e-12] = 1.0
        Xn = Xc / sd; n = X.shape[0]
        first = int(np.argmax(rel)); selected = [first]
        remaining = np.ones(n_feat, bool); remaining[first] = False
        red = np.abs(Xn.T @ Xn[:, first]) / n
        while len(selected) < k and remaining.any():
            score = rel / np.maximum(red / len(selected), 1e-6)
            score[~remaining] = -np.inf
            nxt = int(np.argmax(score)); selected.append(nxt)
            remaining[nxt] = False; red += np.abs(Xn.T @ Xn[:, nxt]) / n
        return np.asarray(selected, int)

    def fit(self, X, y):
        X = np.asarray(X, np.float64); n_feat = X.shape[1]
        cw = self.class_weight
        prot = self._protected_idx(n_feat)
        budget = int(max(1, min(self.k, n_feat)))
        # protected features consume budget first (unless the reducer is RFE, which
        # is allowed to prune redundant protected ones).
        if self.kind == "rf_importance":
            imp = self._probe_rf().fit(X, y).feature_importances_
            order = [i for i in np.argsort(imp)[::-1]]
            idx = self._fill(prot, order, budget)
        elif self.kind == "mutual_info":
            sc = np.nan_to_num(SelectKBest(mutual_info_classif, k="all").fit(X, y).scores_)
            order = [i for i in np.argsort(sc)[::-1]]
            idx = self._fill(prot, order, budget)
        elif self.kind == "rfe":
            rfe = RFE(self._probe_rf(), n_features_to_select=budget, step=0.1).fit(X, y)
            idx = list(np.where(rfe.support_)[0])
        elif self.kind == "mi_rfe_hybrid":
            m = min(200, n_feat)
            mi = np.nan_to_num(mutual_info_classif(X, y, random_state=self.random_state))
            cand = np.argsort(mi)[::-1][:m]
            cand = np.array(sorted(set(cand.tolist()) | set(prot)))
            sub = X[:, cand]
            rfe = RFE(self._probe_rf(), n_features_to_select=min(budget, len(cand)),
                      step=0.1).fit(sub, y)
            idx = list(cand[np.where(rfe.support_)[0]])
        else:
            raise ValueError(f"unknown reducer {self.kind!r}")
        idx = np.asarray(sorted(int(i) for i in dict.fromkeys(idx)), int)[:budget]
        self.sel_indices_ = idx
        self.mean_ = np.zeros(n_feat, np.float64)
        comp = np.zeros((len(idx), n_feat), np.float64); comp[np.arange(len(idx)), idx] = 1.0
        self.components_ = comp; self.orig_index_ = idx
        self.n_components_ = len(idx); self.n_features_in_ = n_feat
        self.selector_kind_ = ("rfe" if self.kind == "mi_rfe_hybrid" else self.kind)
        self.reducer_shape_ = "selection"
        return self

    @staticmethod
    def _fill(protected, order, budget):
        out = list(dict.fromkeys(protected))          # protected first, de-duped
        for i in order:
            if len(out) >= budget:
                break
            if i not in out:
                out.append(int(i))
        return out[:budget]

    def transform(self, X):
        return np.asarray(X, np.float64)[:, self.sel_indices_]


def make_selector(kind, k, feature_names, protected, class_weight):
    if kind == "pca":
        return TopPCAComponents(k=k)
    return RawFeatureSelector(kind, k, feature_names, protected, class_weight)


def build_pipeline(kind, k, feature_names, protected, class_weight):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    make_selector(kind, k, feature_names, protected, class_weight)),
        ("clf",    RandomForestClassifier(class_weight=class_weight, **RF_KW)),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# LOAD + SCHEMA NORMALISE
# ══════════════════════════════════════════════════════════════════════════════
CSV_PATH = next((p for p in DATASET_CANDIDATES if os.path.exists(p)), None)
if CSV_PATH is None:
    raise FileNotFoundError("dataset not found; searched:\n  " +
                            "\n  ".join(os.path.normpath(p) for p in DATASET_CANDIDATES))
df = pd.read_csv(CSV_PATH)
print(f"Dataset : {os.path.normpath(CSV_PATH)}   shape={df.shape}")

RAW6 = COL_ACCEL + COL_GYRO
for c in RAW6:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=RAW6).reset_index(drop=True)

def _side_of(v):
    u = str(v).upper().strip()
    if u in DEVICE_LEFT_VALUES:  return "LEFT"
    if u in DEVICE_RIGHT_VALUES: return "RIGHT"
    return u
df["_side"] = df[COL_DEVICE].map(_side_of)

# ── LABEL -> factors ──────────────────────────────────────────────────────────
_parsed = df[COL_LABEL].map(parse_label)
for factor in ("body", "direction", "side"):
    df[f"_{factor}"] = _parsed.map(lambda d: d[factor])
ACTIVE_HEADS = [h for h in ("body", "direction", "side")
                if df[f"_{h}"].notna().any() and df[f"_{h}"].nunique() >= 2]
print(f"Active heads (>=2 classes present): {ACTIVE_HEADS}")
for h in ("body", "direction", "side"):
    vc = df[f"_{h}"].value_counts(dropna=False).to_dict()
    print(f"  factor {h:<9}: {vc}")
_unparsed = df[df[[f"_{h}" for h in ACTIVE_HEADS]].isna().any(axis=1)][COL_LABEL].unique()
if len(_unparsed):
    print(f"  WARNING: labels not fully parsed by FACTOR_KEYWORDS (edit config): "
          f"{list(_unparsed)[:10]}")

# ══════════════════════════════════════════════════════════════════════════════
# COVERAGE TABLES (spec §7) — abort/flag single-session or single-participant class
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65 + "\nCLASS COVERAGE (report up front)\n" + "=" * 65)
def _coverage(df, by):
    if by not in df.columns:
        print(f"  ({by} column absent — skipped)"); return None
    tab = df.groupby([COL_LABEL, by]).size().unstack(fill_value=0)
    n_groups = (tab > 0).sum(axis=1)
    print(f"\n  {COL_LABEL} x {by}  (#{by} carrying each class):")
    print(n_groups.to_string())
    singles = n_groups[n_groups <= 1].index.tolist()
    if singles:
        print(f"  ⚠ classes covered by a SINGLE {by} (confound risk): {singles}")
    return n_groups

cov_session = _coverage(df, COL_SESSION)
cov_part    = _coverage(df, COL_PARTICIPANT)
cov_dev     = _coverage(df, COL_DEVICE)
SINGLE_SESSION_CLASSES = ([] if cov_session is None
                          else cov_session[cov_session <= 1].index.tolist())
if SINGLE_SESSION_CLASSES:
    print(f"\n  ACCEPTANCE-CHECK RISK: {SINGLE_SESSION_CLASSES} covered by one "
          f"session — cross-session accuracy for them is unmeasurable (spec §7/§9).")

# ══════════════════════════════════════════════════════════════════════════════
# CANONICAL FRAME — remap LEFT device into the common body frame (spec §2)
# ══════════════════════════════════════════════════════════════════════════════
df = df.sort_values([COL_SESSION, COL_DEVICE, COL_TIME]).reset_index(drop=True)
for (sid, dev), grp in df.groupby([COL_SESSION, COL_DEVICE], sort=False):
    side = _side_of(dev)
    can = F.canonicalize_device(grp[RAW6].values, side)
    df.loc[grp.index, RAW6] = can
print("\nCanonical frame: LEFT remap applied "
      f"(LEFT_CANONICAL_REMAP={F.LEFT_CANONICAL_REMAP}).")

# ══════════════════════════════════════════════════════════════════════════════
# DUAL-DEVICE WINDOWING (spec §1/§2) — window each (session,device) stream, then
# join L+R per window index for the Side head.  Single-device sessions are FLAGGED
# and excluded from Head S but still used for Heads B/D.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"WINDOWING  size={F.WINDOW_SIZE} step={F.STEP_SIZE} @ {F.FS_HZ:.0f} Hz "
      f"(per session+device, time-sorted)")
print("=" * 65)

def _window_stream(raw6, labels):
    """Yield (start, window6, majority_label, purity) for a single stream."""
    n = len(raw6)
    for s in range(0, n - F.WINDOW_SIZE + 1, F.STEP_SIZE):
        e = s + F.WINDOW_SIZE
        vals, cnts = np.unique(labels[s:e], return_counts=True)
        maj = vals[cnts.argmax()]
        yield s, raw6[s:e], maj, cnts.max() / F.WINDOW_SIZE

rows, meta = [], []       # per-device window rows (base + B + D + P features)
sample_windows = {}       # one pure raw window per class (golden vectors / selfcheck)
# store per (session, side) preprocessed windows so Head S can join L/R by index
per_stream = {}
for (sid, dev), grp in df.groupby([COL_SESSION, COL_DEVICE], sort=False):
    side = _side_of(dev)
    raw6 = grp[RAW6].values.astype(np.float64)
    labels = grp[COL_LABEL].values
    part = grp[COL_PARTICIPANT].iloc[0] if COL_PARTICIPANT in grp.columns else ""
    stream_windows = []
    for widx, (s, win6, maj, purity) in enumerate(_window_stream(raw6, labels)):
        if purity < 1.0:            # keep only 100%-pure windows (spec §1)
            continue
        prep = F.preprocess_device_stream(win6)
        feat = F.extract_base_stats(win6)
        feat.update(F.group_B_features(prep, "B"))
        feat.update(F.group_D_features(prep, "D"))
        feat.update(F.family_P_features(prep, "P"))
        fac = parse_label(maj)
        rec = dict(feat)
        rec.update({"_label": maj, "_session": sid, "_side": side,
                    "_part": part, "_widx": widx,
                    "_body": fac["body"], "_direction": fac["direction"],
                    "_side_lbl": fac["side"]})
        rows.append(rec)
        meta.append((sid, side, widx))
        stream_windows.append((widx, prep, win6, maj))
        sample_windows.setdefault(maj, win6.copy())
    per_stream[(sid, side)] = {w: (prep, win6, lbl)
                               for (w, prep, win6, lbl) in stream_windows}

dfw = pd.DataFrame(rows)
print(f"  per-device windows : {len(dfw):,}")
print(f"  label balance      :\n{dfw['_label'].value_counts().to_string()}")

# ── Head-S dual-device join: for each session, pair LEFT & RIGHT windows by _widx ─
rest_thr = REST_THRESHOLD
if rest_thr is None:              # learn 20th-pct RMS ang-vel over all windows (§4)
    _rms = []
    for (sid, side), wins in per_stream.items():
        for w, (prep, win6, lbl) in wins.items():
            _rms.append(float(np.sqrt(np.mean(np.sum(prep["gyro"] ** 2, axis=1)))))
    rest_thr = float(np.percentile(_rms, 20)) if _rms else 0.0
print(f"  Head-S rest threshold (RMS ang-vel) : {rest_thr:.4f}")

side_rows = []
sessions = sorted({sid for (sid, _s) in per_stream})
n_single = 0
for sid in sessions:
    L = per_stream.get((sid, "LEFT")); R = per_stream.get((sid, "RIGHT"))
    if not L or not R:
        n_single += 1; continue          # single-device session -> excluded from Head S
    for w in sorted(set(L) & set(R)):     # nearest-sample = same window index
        prepL, _, lblL = L[w]; prepR, _, _ = R[w]
        sfeat = F.group_S_features(prepL, prepR, rest_thresh=rest_thr, prefix="S")
        side_lbl = parse_label(lblL)["side"]     # side factor of the paired class
        side_rows.append({**sfeat, "_session": sid, "_widx": w,
                          "_label": lblL, "_side_lbl": side_lbl})
print(f"  dual-device sessions : {len(sessions) - n_single} / {len(sessions)} "
      f"({n_single} single-device excluded from Head S)")
dfs = pd.DataFrame(side_rows)

# ── DEDUP exact-duplicate features (spec §2), protecting the 79+66 prefix ───────
feat_cols = [c for c in dfw.columns if not c.startswith("_")]
dfw[feat_cols] = dfw[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
kept, dup_of = F.dedup_exact(dfw, feat_cols, F.EXPECTED_TOTAL_FEAT)
dropped = [c for c in feat_cols if c not in kept]
if dropped:
    # never drop a protected direction feature via dedup silently
    dropped = [c for c in dropped if c not in PROTECTED_DIRECTION_FEATURES]
    dfw = dfw.drop(columns=dropped)
    print(f"  dedup: dropped {len(dropped)} exact-duplicate feature(s)")
    with open("generated/rot_dedup_dropped.json", "w") as fh:
        json.dump({c: dup_of[c] for c in dropped}, fh, indent=2)
FEATURES = [c for c in dfw.columns if not c.startswith("_")]
print(f"  final per-device feature count : {len(FEATURES)}")

# per-device gyro energy + active flag (a resting arm during a one-sided gesture is
# NOT a valid body/direction example — filter it out of Heads B/D).
dfw["_gyro_energy"] = dfw["gyro_mag__energy"].astype(float) * F.WINDOW_SIZE \
    if "gyro_mag__energy" in dfw.columns else 0.0
_active_floor = float(np.percentile(dfw["_gyro_energy"], 20)) if len(dfw) else 0.0
dfw["_active"] = dfw["_gyro_energy"] > _active_floor

# ══════════════════════════════════════════════════════════════════════════════
# SHARED session-level held-out split (spec §7) — test sessions never touched in
# any sweep/benchmark.  One split, reused by every head + flat + combine.
# ══════════════════════════════════════════════════════════════════════════════
_combo = (dfw["_body"].astype(str) + "|" + dfw["_direction"].astype(str)
          + "|" + dfw["_side_lbl"].astype(str))
_sess = dfw["_session"].values
_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
_tr, _te = next(_sgkf.split(dfw[FEATURES].values, _combo.values, groups=_sess))
TEST_SESSIONS = set(np.unique(_sess[_te]))
print(f"\nHeld-out TEST sessions ({len(TEST_SESSIONS)}): {sorted(TEST_SESSIONS)}")
# assert overlap discipline: no test session appears in train (guaranteed by grouping)
assert not (set(np.unique(_sess[_tr])) & TEST_SESSIONS), "session leaked across split"


def _split_masks(sessions):
    te = np.isin(sessions, list(TEST_SESSIONS))
    return ~te, te


def benchmark_and_fit(head, Xdf, y_str, sessions, protected, budget, active_mask=None):
    """Benchmark HEAD_SELECTORS under session-grouped CV on the TRAIN sessions,
    deploy the winner (CV acc, tie-break fewer output features), report held-out
    test accuracy.  Leak-free: reducers fit per fold inside the pipeline."""
    le = LabelEncoder(); y = le.fit_transform(y_str)
    tr, te = _split_masks(sessions)
    if active_mask is not None:
        # a resting arm carries no body/direction signal — exclude it from BOTH
        # training and scoring so the head is measured only where the gesture is.
        tr = tr & active_mask
        te = te & active_mask
    Xall = Xdf.values.astype(np.float32)
    fnames = list(Xdf.columns)
    Xtr, ytr, gtr = Xall[tr], y[tr], sessions[tr]
    Xte, yte = Xall[te], y[te]
    cw = make_class_weight(ytr)
    n_splits = max(2, min(5, len(np.unique(gtr))))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    for kind in HEAD_SELECTORS:
        k = min(budget, Xtr.shape[1], FEATURE_INDEX_CAP)
        pipe = build_pipeline(kind, k, fnames, protected, cw)
        try:
            yp = cross_val_predict(pipe, Xtr, ytr, groups=gtr, cv=cv, n_jobs=-1)
            acc = accuracy_score(ytr, yp)
            pipe.fit(Xtr, ytr)
            nout = int(pipe.named_steps["pca"].n_components_)
            results.append((kind, acc, nout, pipe))
        except Exception as e:
            print(f"    [{head}] reducer {kind:<14} failed: {e}")
    if not results:
        raise RuntimeError(f"no reducer succeeded for head {head}")
    results.sort(key=lambda r: (-r[1], r[2]))            # CV acc desc, fewer feats
    if FEATURE_SELECTOR != "auto":
        results = [r for r in results if r[0] == FEATURE_SELECTOR] or results
    kind, cv_acc, nout, pipe = results[0]
    test_acc = accuracy_score(yte, pipe.predict(Xte)) if te.sum() else float("nan")
    print(f"  [{head:<9}] winner={kind:<14} CVacc={cv_acc:.3f} "
          f"testacc={test_acc:.3f} nfeat={nout}  "
          f"(ranked: {', '.join(f'{k}:{a:.3f}' for k,a,_,_ in results)})")
    return {"head": head, "le": le, "pipe": pipe, "kind": kind,
            "cv_acc": cv_acc, "test_acc": test_acc, "features": fnames,
            "y": y, "test_mask": te}


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN HEADS  (B/D on active per-device windows; S on joined dual-device windows)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65 + "\nHEAD TRAINING (session-grouped, leak-free)\n" + "=" * 65)
HEADS = {}
Xdw = dfw[FEATURES]
if "body" in ACTIVE_HEADS:
    HEADS["body"] = benchmark_and_fit(
        "body", Xdw, dfw["_body"].astype(str).values, _sess,
        protected=set(), budget=HEAD_BUDGET["body"], active_mask=dfw["_active"].values)
if "direction" in ACTIVE_HEADS:
    HEADS["direction"] = benchmark_and_fit(
        "direction", Xdw, dfw["_direction"].astype(str).values, _sess,
        protected=PROTECTED_DIRECTION_FEATURES, budget=HEAD_BUDGET["direction"],
        active_mask=dfw["_active"].values)
if "side" in ACTIVE_HEADS and len(dfs):
    Sfeat = [c for c in dfs.columns if not c.startswith("_")]
    dfs[Sfeat] = dfs[Sfeat].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    HEADS["side"] = benchmark_and_fit(
        "side", dfs[Sfeat], dfs["_side_lbl"].astype(str).values,
        dfs["_session"].values, protected=set(), budget=HEAD_BUDGET["side"])

# ── ORACLE bound + hierarchical/flat evaluation on JOINED dual-device windows ──
print("\n" + "=" * 65 + "\nHIERARCHICAL vs FLAT (12-class) on held-out sessions\n" + "=" * 65)
oracle = 1.0
for h in ("body", "direction", "side"):
    if h in HEADS:
        oracle *= HEADS[h]["test_acc"]
print(f"Oracle upper bound (product of per-head test acc) : {oracle:.3f}")

# Build joined dual-device evaluation table: pick the ACTIVE device's B/D/P/base
# features per window, attach the Side features, form the 12-class combo label.
dfw_idx = dfw.set_index(["_session", "_side", "_widx"])
joined = []
for _, sr in dfs.iterrows():
    sid, w = sr["_session"], sr["_widx"]
    lbl = sr["_label"]
    act = "LEFT" if sr.get("S__L_gyro_energy", 0) >= sr.get("S__R_gyro_energy", 0) else "RIGHT"
    key = (sid, act, w)
    if key not in dfw_idx.index:
        key = (sid, "LEFT" if act == "RIGHT" else "RIGHT", w)
        if key not in dfw_idx.index:
            continue
    bd = dfw_idx.loc[key]
    fac = parse_label(lbl)
    combined = {c: float(bd[c]) for c in FEATURES}
    combined.update({c: float(sr[c]) for c in dfs.columns if not c.startswith("_")})
    combined.update({"_session": sid, "_label": lbl,
                     "_combo": f"{fac['body']}|{fac['direction']}|{fac['side']}"})
    joined.append(combined)
dfj = pd.DataFrame(joined)
if len(dfj):
    jte = np.isin(dfj["_session"].values, list(TEST_SESSIONS))
    dfj_te = dfj[jte]
    print(f"Joined dual-device windows: {len(dfj)}  (held-out: {int(jte.sum())})")

    # soft-combine: P(body)*P(direction)*P(side) over VALID observed 12 combos
    def _proba(head, cols_df):
        h = HEADS[head]
        X = cols_df[h["features"]].values.astype(np.float32)
        return h["le"].classes_, h["pipe"].predict_proba(X)

    combos = sorted(dfj["_combo"].unique())
    soft_pred, y_true = [], dfj_te["_combo"].values
    if all(h in HEADS for h in ("body", "direction", "side")) and len(dfj_te):
        cB, pB = _proba("body", dfj_te)
        cD, pD = _proba("direction", dfj_te)
        cS, pS = _proba("side", dfj_te)
        iB = {c: i for i, c in enumerate(cB)}
        iD = {c: i for i, c in enumerate(cD)}
        iS = {c: i for i, c in enumerate(cS)}
        for n in range(len(dfj_te)):
            best, bestc = -1.0, combos[0]
            for combo in combos:
                b, d, s = combo.split("|")
                if b not in iB or d not in iD or s not in iS:
                    continue
                p = pB[n, iB[b]] * pD[n, iD[d]] * pS[n, iS[s]]
                if p > best:
                    best, bestc = p, combo
            soft_pred.append(bestc)
        hier_acc = accuracy_score(y_true, soft_pred)
        print(f"Hierarchical soft-combine 12-class acc : {hier_acc:.3f}")
    else:
        hier_acc = float("nan")
        print("Hierarchical combine skipped (a head is inactive).")

    # flat 12-class baseline on the joined feature set
    flat_feats = [c for c in dfj.columns if not c.startswith("_")]
    le_f = LabelEncoder(); yf = le_f.fit_transform(dfj["_combo"].astype(str))
    cwf = make_class_weight(yf[~jte])
    flat_pipe = build_pipeline("rf_importance", min(FLAT_BUDGET, len(flat_feats),
                               FEATURE_INDEX_CAP), flat_feats,
                               PROTECTED_DIRECTION_FEATURES, cwf)
    flat_pipe.fit(dfj[flat_feats].values[~jte].astype(np.float32), yf[~jte])
    flat_acc = (accuracy_score(yf[jte], flat_pipe.predict(
        dfj[flat_feats].values[jte].astype(np.float32))) if jte.sum() else float("nan"))
    print(f"Flat 12-class baseline acc             : {flat_acc:.3f}")
else:
    hier_acc = flat_acc = float("nan")
    print("No dual-device joined windows — cannot evaluate 12-class combine.")

# ══════════════════════════════════════════════════════════════════════════════
# ACCEPTANCE CHECKS (spec §9) — run before declaring done
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65 + "\nACCEPTANCE CHECKS\n" + "=" * 65)
checks = []

# 1. canonical-frame unit test (§2): after canonicalisation a 'Both'+'hand'+CLK
#    window must show SAME-SIGN mean gyro on the LEFT and RIGHT devices of the same
#    session.  per_stream holds already-canonicalised windows, so compare directly.
def _both_hand_clk(lbl):
    p = parse_label(lbl)
    return p["side"] == "both" and p["body"] == "hand" and p["direction"] == "clk"
_canon_pass = None
for sid in sessions:
    L = per_stream.get((sid, "LEFT")); R = per_stream.get((sid, "RIGHT"))
    if not L or not R:
        continue
    pairs = [(w, L[w], R[w]) for w in (set(L) & set(R)) if _both_hand_clk(L[w][2])]
    if not pairs:
        continue
    w, (pl, wl, _), (pr, wr, _) = pairs[0]
    lg = pl["gyro"].mean(0); rg = pr["gyro"].mean(0)
    dom = int(np.argmax(np.abs(lg) + np.abs(rg)))
    _canon_pass = bool(np.sign(lg[dom]) == np.sign(rg[dom]))
    break
checks.append(("canonical-frame same-sign gyro (L vs R, Both-Hand-CLK)", _canon_pass))

# 2. threshold-on-D alone >= 85% Direction accuracy on held-out sessions
if "direction" in HEADS:
    d = HEADS["direction"]
    dom_col = "D__ang_disp_dominant"
    if dom_col in dfw.columns:
        # score direction only on ACTIVE-arm windows (resting arm has no direction)
        te = np.isin(_sess, list(TEST_SESSIONS)) & dfw["_active"].values
        ydir = (dfw["_direction"].astype(str).values == "clk")
        thr_pred = dfw[dom_col].values < 0                  # D<0 -> CW (spec §4)
        # allow either sign polarity (mounting-dependent) — take the better
        acc_a = accuracy_score(ydir[te], thr_pred[te])
        acc_b = accuracy_score(ydir[te], ~thr_pred[te])
        d_thr = max(acc_a, acc_b)
        checks.append((f"threshold-on-D Direction acc >= 0.85 (got {d_thr:.3f})",
                       d_thr >= 0.85))

# 3. gyro_accel_energy_ratio in top-5 importances of Head B
if "body" in HEADS:
    b = HEADS["body"]
    red = b["pipe"].named_steps["pca"]
    clf = b["pipe"].named_steps["clf"]
    if hasattr(red, "sel_indices_"):
        kept_names = [b["features"][i] for i in red.sel_indices_]
        imp = clf.feature_importances_
        top5 = [kept_names[i] for i in np.argsort(imp)[::-1][:5]]
        checks.append(("gyro_accel_energy_ratio in Head-B top-5",
                       "B__gyro_accel_energy_ratio" in top5))
    else:
        checks.append(("Head-B top-5 (projection reducer — names opaque)", None))

# 4. hierarchical soft-combine >= flat baseline, both >= 0.90; Head B >= 0.92
if not np.isnan(hier_acc) and not np.isnan(flat_acc):
    checks.append((f"hierarchical ({hier_acc:.3f}) >= flat ({flat_acc:.3f})",
                   hier_acc >= flat_acc))
    checks.append((f"hierarchical >= 0.90 (got {hier_acc:.3f})", hier_acc >= 0.90))
    checks.append((f"flat >= 0.90 (got {flat_acc:.3f})", flat_acc >= 0.90))
if "body" in HEADS:
    hb = HEADS["body"]["test_acc"]
    checks.append((f"Head B >= 0.92 (got {hb:.3f}) — else iterate Group B first",
                   hb >= 0.92))

# 5. no class covered by a single session
checks.append((f"no single-session class (offenders: {SINGLE_SESSION_CLASSES})",
               len(SINGLE_SESSION_CLASSES) == 0))

for name, ok in checks:
    mark = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
    print(f"  [{mark}] {name}")
with open("generated/rot_acceptance.json", "w") as fh:
    json.dump({name: ok for name, ok in checks}, fh, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDED C EXPORT (spec §8) — per-head compact trees + ONE shared feature gather
# table (int8 indices, hence the 127 cap) + scaler/reducer arrays + golden vectors.
#   Selection-shape reducers export as an index gather only (no float matrix).
#   Projection-shape reducers (PCA) also export mean_/components_ rows.
# The C feature extractor must MIRROR har_rot_features §3-§4 exactly (median ->
# Butterworth 10Hz -> Madgwick @50Hz per device -> Group B/D/S + Family P); the
# golden vectors below are the firmware parity contract.  Direction/Group-S
# features require runtime cumtrapz + Madgwick state on-device and both wrists
# streamed to the classifying node.
# ══════════════════════════════════════════════════════════════════════════════
def _serialize_tree(tree):
    """sklearn tree -> list of IMUNode dicts {threshold,left,right,feature,pred}."""
    t = tree.tree_
    nodes = []
    for i in range(t.node_count):
        leaf = t.children_left[i] == -1
        nodes.append({
            "threshold": float(t.threshold[i]) if not leaf else 0.0,
            "left":  int(t.children_left[i]),
            "right": int(t.children_right[i]),
            "feature": int(t.feature[i]) if not leaf else -1,   # idx into REDUCED vec
            "pred": int(np.argmax(t.value[i][0])) if leaf else 0,
        })
    return nodes


def export_head(head_obj):
    """Model arrays for one head: scaler, reducer, per-tree IMUNode arrays."""
    pipe = head_obj["pipe"]
    sc  = pipe.named_steps["scaler"]
    red = pipe.named_steps["pca"]
    clf = pipe.named_steps["clf"]
    out = {
        "head": head_obj["head"],
        "classes": head_obj["le"].classes_.tolist(),
        "scaler_mean":  sc.mean_.astype(float).tolist(),
        "scaler_scale": sc.scale_.astype(float).tolist(),
        "reducer_shape": red.reducer_shape_,
        "reducer_kind":  red.selector_kind_,
        "n_out": int(red.n_components_),
        "trees": [_serialize_tree(t) for t in clf.estimators_],
    }
    if red.reducer_shape_ == "selection":
        out["gather_index"] = [int(i) for i in red.sel_indices_]      # raw-feat idx
    else:
        out["reducer_mean"] = red.mean_.astype(float).tolist()
        out["reducer_components"] = red.components_.astype(float).tolist()
    return out


if C_EXPORT and HEADS:
    export = {"fs_hz": F.FS_HZ, "window": F.WINDOW_SIZE, "step": F.STEP_SIZE,
              "feature_names": FEATURES,
              "left_canonical_remap": F.LEFT_CANONICAL_REMAP,
              "gyro_in_deg": F.GYRO_IN_DEG,
              "madgwick_beta": F.MADGWICK_BETA,
              "protected_direction_features": sorted(PROTECTED_DIRECTION_FEATURES),
              "heads": {h: export_head(HEADS[h]) for h in HEADS}}

    # shared gather table = union of every selection head's raw-feature indices
    shared = sorted({i for h in HEADS.values()
                     for i in (h["pipe"].named_steps["pca"].__dict__.get("sel_indices_", []))})
    export["shared_gather_index"] = [int(i) for i in shared]
    assert len(shared) <= FEATURE_INDEX_CAP, \
        f"shared gather width {len(shared)} exceeds int8 cap {FEATURE_INDEX_CAP}"

    with open("generated/imu_rot_model.json", "w") as fh:
        json.dump(export, fh)
    print(f"\nC-export model : generated/imu_rot_model.json "
          f"(shared gather width={len(shared)} <= {FEATURE_INDEX_CAP})")

    # ── golden vectors: one pure window per class -> features -> expected preds ──
    golden = []
    for lbl, win6 in sample_windows.items():
        prep = F.preprocess_device_stream(win6)
        feat = F.extract_base_stats(win6)
        feat.update(F.group_B_features(prep, "B"))
        feat.update(F.group_D_features(prep, "D"))
        feat.update(F.family_P_features(prep, "P"))
        fvec = [float(feat.get(c, 0.0)) for c in FEATURES]
        exp = {}
        for h, hobj in HEADS.items():
            if h == "side":       # side needs a dual-device window — skip single
                continue
            x = np.array([[feat.get(c, 0.0) for c in hobj["features"]]], np.float32)
            exp[h] = hobj["le"].classes_[int(hobj["pipe"].predict(x)[0])]
        golden.append({"label": lbl, "features": fvec, "expected": exp})
    with open("generated/imu_rot_golden.json", "w") as fh:
        json.dump(golden, fh)
    print(f"Golden vectors : generated/imu_rot_golden.json ({len(golden)} windows)")

    # ── flash budget summary ──────────────────────────────────────────────────
    node_bytes = 10
    total_nodes = sum(len(t) for h in export["heads"].values() for t in h["trees"])
    flash = total_nodes * node_bytes
    for h in export["heads"].values():
        flash += (len(h["scaler_mean"]) + len(h.get("reducer_mean", []))) * 4
        flash += sum(len(r) for r in h.get("reducer_components", [])) * 4
    print("\n=== FLASH BUDGET SUMMARY (ESP32) ===")
    print(f"  heads={len(HEADS)}  trees={sum(len(h['trees']) for h in export['heads'].values())}"
          f"  nodes={total_nodes}")
    print(f"  model flash ~= {flash/1024:.1f} KB / {FLASH_BUDGET_KB} KB budget "
          f"({100*flash/1024/FLASH_BUDGET_KB:.1f}%)")

print("\nDone.  Outputs in generated/ and plots/.")
print("Reminder: fill the DATASET CONFIG block + finalise LEFT_CANONICAL_REMAP "
      "(run the canonical self-check) before trusting these numbers.")
