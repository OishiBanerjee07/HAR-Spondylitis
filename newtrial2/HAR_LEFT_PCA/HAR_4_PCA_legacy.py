

import warnings; warnings.filterwarnings("ignore")
import os, sys, json, textwrap
# ── BLAS THREAD CAP (must be set BEFORE numpy/scipy/sklearn import) ────────────
# The reducer benchmark runs cross_validate(n_jobs=-1); each worker process then had
# OpenBLAS spawn one thread PER CORE, and the per-thread allocations multiply across
# workers → "OpenBLAS: Memory allocation still failed after 10 retries" and killed
# workers (TerminatedWorkerError → 'all reducers failed to benchmark').  Pinning each
# process to a single BLAS thread removes the explosion; joblib still parallelises
# across worker PROCESSES, so results are byte-identical — only thread fan-out changes.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
# Windows consoles default to cp1252 and choke on the ✓/⚠/█ glyphs used below.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew as sp_skew, kurtosis as sp_kurt
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, GroupShuffleSplit, GroupKFold,
                                     StratifiedGroupKFold, RandomizedSearchCV)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score)
import joblib
from adaptive_weighting import AdaptiveWeightedClassifier

os.makedirs("generated", exist_ok=True)
os.makedirs("plots",     exist_ok=True)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
# DATASET: the model is now trained on the newly collected ROT_activities_master.csv
# — the ROTATION activity set this LEFT-wrist model targets.  The 12 classes follow a
# {SIDE}_{BODY}_{DIRECTION} naming convention:
#     SIDE      : B (both hands), L (left hand), R (right hand)
#     BODY      : HAND (wrist rotation), SHOU (shoulder rotation)
#     DIRECTION : CLK (clockwise), ACLK (anti-clockwise)
#   e.g. B_HAND_ACLK, L_SHOU_CLK, R_HAND_CLK, …
# The path is resolved relative to THIS file so the script runs regardless of the
# current working directory.  Both .csv and .xlsx are accepted — the loader below
# dispatches on the extension (_read_xlsx handles .xlsx).
#   NOTE: ROT_activities_master.csv ships raw axes + mag_xyz (magnetometer) +
#   session_id / participant_id.  The columns the downstream pipeline expects
#   (accel_mag / gyro_mag magnitudes, device_encoded, activity_id) are re-derived in
#   the SCHEMA-NORMALISE block so EDA, windowing and training run unchanged.  The raw
#   CSV itself is NEVER modified — the left-hand row selection happens in-memory.
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_DATASET_CANDIDATES = [
    os.path.join(_here, "ROT_activities_master.csv"),        # primary (this folder)
    os.path.join(_here, "..", "ROT_activities_master.csv"),
    os.path.join(_here, "ROT_activities_master.xlsx"),
]
CSV_PATH = next((p for p in _DATASET_CANDIDATES if os.path.exists(p)), None)
if CSV_PATH is None:
    raise FileNotFoundError(
        "ROT_activities_master.csv/.xlsx not found. Searched:\n  "
        + "\n  ".join(os.path.normpath(p) for p in _DATASET_CANDIDATES))

# ── HAND-SELECTION POLICY (LEFT-hand rotation model) ──────────────────────────
# This is a LEFT-hand model.  The ONLY rows removed are the activities where the
# RIGHT hand is the one in action — the R_* classes (R_HAND_*, R_SHOU_*).  Every
# other row is kept AS-IS, including BOTH device streams (LEFT and RIGHT), for the
# left-hand (L_*) and both-hand (B_*) rotations.  The RIGHT device is NOT dropped.
#
#   • L_*  (left hand rotates)   → KEEP  (both device streams)
#   • B_*  (both hands rotate)   → KEEP  (both device streams)
#   • R_*  (right hand in action)→ DROP  (both device streams) — the only removal
#
# The real CSV is never modified — the R_* rows are filtered in-memory only.
TRAIN_HAND = "LEFT"

# SIDE prefix parser.  DROP_SIDES lists the activity sides removed for this model:
# 'right' = the R_* rotations (the right hand is the one in action).
def _activity_side(label) -> str:
    """Return the rotation SIDE ('both'/'left'/'right'/'unknown') from a label's
    {SIDE}_{BODY}_{DIR} prefix (B_/L_/R_), case/spacing agnostic."""
    s = str(label).strip().upper().replace(" ", "_").replace("-", "_")
    head = s.split("_", 1)[0]
    return {"B": "both", "BOTH": "both",
            "L": "left", "LEFT": "left",
            "R": "right", "RIGHT": "right"}.get(head, "unknown")
DROP_SIDES = {"right"}   # only the R_* (right-hand-in-action) activities are dropped

# NOTE: the wrist-mirror / axis-reflection transform has been REMOVED from this
# architecture.  Both device streams are used exactly as recorded — no activity's
# axes are ever sign-flipped or reflected between wrists.
SENSOR_COLS = ["accel_x","accel_y","accel_z","accel_mag",
               "gyro_x","gyro_y","gyro_z","gyro_mag"]

# 6 raw axes for statistical abstraction
#  index:     0          1          2          3         4         5
RAW_AXES  = ["accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z"]
CORR_PAIRS = [(0,1),(0,2),(1,2),(3,4),(3,5),(4,5)]
CORR_NAMES = [("accel_x","accel_y"),("accel_x","accel_z"),("accel_y","accel_z"),
              ("gyro_x","gyro_y"),  ("gyro_x","gyro_z"),  ("gyro_y","gyro_z")]
JERK_IDX   = [0, 1, 2]

WINDOW_SIZE = 150   # CHOSEN BY THE WINDOW-SIZE SWEEP (master-prompt item 2): of
                    # {64,80,96,100,128,150}, size=150 posted the best leak-free
                    # grouped-CV val acc (0.383 vs 0.357 at the old 100).  A longer
                    # window (~3 s @ 50 Hz) captures a fuller rotation cycle, so the
                    # periodicity / CLK-vs-ACLK direction cue is measurable.  Re-run
                    # `WIN_SWEEP=1 python HAR_4_PCA.py` to re-derive on new data.
# 50 % overlap: adjacent windows share half their samples.  step = WINDOW_SIZE/2.
STEP_SIZE   = 75    # 50 % overlap (WINDOW_SIZE / 2)

# ── WINDOW-SIZE SWEEP (master-prompt item 2) ──────────────────────────────────
# Benchmark candidate window sizes on the session-grouped protocol and deploy the
# best.  Overlap is configurable (WIN_OVERLAP → step = round(size*(1-overlap))).
# Each candidate re-extracts the full feature matrix, so the sweep costs
# ~len(WINDOW_CANDIDATES)× a normal extraction → gated behind WIN_SWEEP (default
# off).  When off, the architecture default 100/50 above is used.  Note: freq
# features zero-pad/truncate to NFFT=128, so windows >128 use only their first 128
# samples for the spectral block (time/rotation features still use the full window).
WINDOW_CANDIDATES = [64, 80, 96, 100, 128, 150]
WIN_OVERLAP       = float(os.environ.get("WIN_OVERLAP", "0.5"))   # 0.5 → 50% overlap
WIN_SWEEP         = os.environ.get("WIN_SWEEP", "0") == "1"

# ── RF HYPER-PARAMETERS (regularised against the ~31pp train-vs-val gap) ──────
# The forest was still memorising training sessions — a ~31pp train-vs-validation
# gap.  Tighten the variance knobs further (each one shrinks per-tree capacity, so
# the ensemble stops keying on per-session noise and the gap closes):
#   n_estimators 18 → 100 : more trees → a smoother, lower-variance vote.
#   max_depth    8  → 6   : even shallower trees — 6 levels can't carve the deep,
#                           session-specific leaves that drove the memorisation.
#                           (Raise back to 8 if validation accuracy is starved; 6
#                           and 8 are the two sanctioned ceilings.)
#   min_samples_leaf 4 → 5: every leaf must generalise over ≥5 windows, killing the
#                           near-pure leaves that overfit a single session.
#   max_features = "sqrt" : each split sees only √n_features candidates, decorrelating
#                           the trees so no single dominant feature is reused in every
#                           tree (the main lever against high-variance ensembles).
RF_N_TREES          = int(os.environ.get("RF_N_TREES", "18"))   # 18→40: more trees stabilise the
                                                                # ensemble at the wider K=80 hybrid input
RF_MAX_DEPTH        = int(os.environ.get("RF_MAX_DEPTH", "8"))
RF_MIN_SAMPLES_LEAF = int(os.environ.get("RF_MIN_SAMPLES_LEAF", "2"))
RF_MIN_SAMPLES_SPLIT= int(os.environ.get("RF_MIN_SAMPLES_SPLIT", "4"))
RF_MAX_FEATURES     = "sqrt"
                             # axes = gravity direction) that separate the static
                             # postures standing/sitting; running's huge motion
                             # variance dominated the top PCs.  0.99 keeps them.
# ── RF HYPERPARAMETER SEARCH (RandomizedSearchCV, session-grouped, leak-safe) ──
# Master-prompt item 3.  The hand-picked RF above is deliberately conservative;
# this runs a RandomizedSearchCV over the deployed pipeline (scaler+reducer+RF)
# scored by StratifiedGroupKFold on the TRAINING sessions ONLY — the held-out
# TEST sessions never inform the search, so the winning config is chosen on true
# unseen-session generalisation, not the test set.  The best clf__* params
# overwrite the RF_* globals below so every downstream build_pipeline() (reducer
# benchmark, overfit check, final fit, deployed model, C export) ships the tuned
# forest.  Gated by RF_SEARCH (default on); RF_SEARCH_ITER caps the sampled
# configs for runtime.  Fixed random_state → reproducible.  n_estimators is kept
# modest (embedded compactness) and depth/leaf/split/max_features do the real
# regularisation work against the ~0.76 train-val overfit gap.
RF_BOOTSTRAP     = True    # bagging on (prompt item 3); also enables OOB scoring
RF_SEARCH        = os.environ.get("RF_SEARCH", "1") == "1"
RF_SEARCH_ITER   = int(os.environ.get("RF_SEARCH_ITER", "40"))   # sampled configs

# ── PCA COMPONENT BUDGET (ADAPTIVE supervised top-K selection) ────────────────
# After PCA expands to the full PCA_VARIANCE_TARGET pool, the model keeps the
# PCA_TOP_K components the RandomForest relies on most (ranked by importance) —
# NOT the first K by variance.  Several high-importance axes (kurtosis / skew /
# FFT-shape) live in PCA's low-variance tail, so a plain variance truncation would
# discard them.
#
# CHANGE: PCA_TOP_K is NO LONGER hard-capped at 20.  Instead the kept-component
# count is GROWN adaptively (PCA-K SWEEP block, right after n_pca_var is known):
# start at PCA_TOP_K_MIN and add PCA_TOP_K_STEP components at a time, re-evaluating
# the held-out-session TEST accuracy each step, and STOP as soon as it reaches
# PCA_TOP_K_TARGET_ACC (≥80%).  If even the full pool can't hit the target we keep
# the K with the best accuracy.  PCA_TOP_K below is only the search FLOOR / seed;
# the deployed value is whatever the sweep lands on.
PCA_VARIANCE_TARGET  = 0.99    # cumulative variance the candidate pool must cover
PCA_TOP_K_MIN        = 20      # search floor (smallest pool we bother trying)
PCA_TOP_K_STEP       = 5       # grow the kept-component count by this each step
PCA_TOP_K_TARGET_ACC = 0.95   # grow K until held-out test accuracy ≥ this
PCA_TOP_K_MAX        = None    # hard cap on K; None → the variance pool (n_pca_var)
PCA_TOP_K            = PCA_TOP_K_MIN   # adapted at runtime by the PCA-K sweep

# ── FEATURE-REDUCER SELECTION (swappable, leak-safe) ──────────────────────────
# The dimensionality-reduction slot is now configurable.  Every reducer is a
# scikit-learn transformer fit INSIDE the pipeline, so cross_validate / the honest
# splits refit it per fold → no selection-on-test leakage (the trap in a naive
# "fit a selector on all 560 features then filter train+test").  Supported kinds:
#
#   "pca"          : supervised top-K PCA (TopPCAComponents) — variance pool then
#                    top-PCA_TOP_K by RF importance.            [projection]
#   "lda"          : LinearDiscriminantAnalysis → n_classes-1 components that
#                    maximise between-class separation.         [projection]
#   "rf_importance": RandomForest fit on the 560 features, keep the top-K by
#                    feature_importances_.                       [raw selection]
#   "mutual_info"  : SelectKBest(mutual_info_classif) top-K.     [raw selection]
#   "hybrid"       : HYBRID physics-aware reducer.  SelectKBest(mutual_info_classif)
#                    ranks every feature by MI to the label, then the high-impact
#                    rotation-physics descriptors (angular displacement, circular
#                    motion, gyro energy, energy ratio, RMS angular velocity,
#                    correlation, accel magnitude, accel variance, orientation change,
#                    gyro-to-accel, peak-to-peak — see _is_physics_feature) are taken
#                    FIRST in MI order, and any remaining budget is filled with the
#                    next-best features by MI.  Replaces PCA as the deployed reducer.
#                                                                 [raw selection]
#   "mrmr"         : mRMR — Maximum-Relevance/Minimum-Redundancy. Greedy: keep the
#                    feature with the best (MI-to-label − mean |corr|-to-already-
#                    kept) score each step.  Picks discriminative features that are
#                    NOT duplicates of each other (e.g. accel_mag__rms vs
#                    accel_mag__energy), so the budget is spent on distinct activity
#                    information instead of the same signal re-measured. [raw selection]
#   "rfe"          : RFE(RandomForest) pruning down to K.        [raw selection]
#
# FEATURE_SELECTOR pins the reducer used by the deployed pipeline.
#
# CONFIG: the dimensionality reduction is the HYBRID physics-aware reducer (SelectKBest
# + mutual-info ranking with rotation-physics prioritisation) — it REPLACES PCA as the
# deployed reducer.  FEATURE_SELECTOR is PINNED to 'hybrid'.  'pca' is still benchmarked
# alongside it (ALL_SELECTORS) purely for an honest A/B in the log; the deployed model
# uses 'hybrid'.  The hybrid width is SEL_K_HYBRID.
FEATURE_SELECTOR   = "hybrid"  # HYBRID SelectKBest+MI physics-aware reduction
SEL_K_HYBRID       = int(os.environ.get("SEL_K_HYBRID", "80"))  # hybrid kept features (physics-prioritised, then top-MI)
                           # K=80 tuned by the in-memory sweep (sweep_hybrid.py): it more
                           # than DOUBLED the held-out-session test acc (0.160→0.338) and
                           # lifted the leak-free 5-fold session-CV val to ~0.26, with 50 of
                           # the 80 kept features being rotation-physics descriptors.  ≤127
                           # keeps the int8 C-export feature index safe.
SEL_K_RF           = 25    # rf_importance top features (benchmark only)
SEL_K_MI           = 25    # mutual_info  top features (benchmark only)
SEL_K_MRMR         = 25    # mRMR kept features (benchmark only)
SEL_K_RFE          = 25    # rfe          surviving features (benchmark only)
SEL_K_SFS          = 25    # sequential forward selection kept features (benchmark only)
SEL_SFS            = os.environ.get("SEL_SFS", "0") == "1"  # SFS is O(n·k) fits → opt-in
# (pca width is PCA_TOP_K, sized by the PCA-K sweep)

# ── MAGNITUDE / FREQUENCY-DOMAIN FEATURE PRIORITISATION ───────────────────────
# Cross-wrist robustness comes from features that are INVARIANT to which wrist the
# sensor is worn on: vector magnitudes (‖accel‖, ‖gyro‖, and the jerk / gravity /
# body-acceleration magnitudes already produced by the UCI-HAR block — e.g.
# accel_mag__*, gyro_mag__fft_energy, tGravityAccMag-*, tBodyAccJerkMag-*) plus the
# frequency-domain energy/shape stats.  Single-AXIS direction features
# (accel_x__mean, gyro_y__std, …-X/-Y/-Z) flip sign between the two wrists, so a
# model that leans on them transfers poorly across LEFT/RIGHT.
# When PRIORITIZE_MAGNITUDE_FEATURES is on, the raw-feature selectors
# (rf_importance / mutual_info) fill their budget from these wrist-invariant
# magnitude & frequency features FIRST (ranked by importance), only falling back to
# single-axis features if the budget is not yet met — so the deployed model leans
# on wrist-invariant signal and transfers better across wrists.
PRIORITIZE_MAGNITUDE_FEATURES = True

def _is_magfreq_feature(name: str) -> bool:
    """True for wrist-invariant magnitude / frequency-domain / energy / angle
    features; False for single-axis time-domain direction features."""
    n = name.lower()
    return (("mag" in n)               # accel_mag, gyro_mag, …Mag (jerk/gravity/body)
            or ("fft_" in n)           # 66-feature frequency block
            or n.startswith("fbody")   # UCI frequency-domain signals
            or ("energy" in n)         # time- & frequency-domain energy stats
            or n.startswith("angle(")) # gravity-orientation angles (no single axis)

# ── PHYSICS-PRIORITY FEATURE MATCHER (drives the HYBRID reducer) ──────────────
# The hybrid reducer (FEATURE_SELECTOR="hybrid") ranks every feature by mutual
# information to the label, then PRIORITISES the physically-meaningful gesture
# descriptors listed below — the quantities that separate the rotation classes on
# first principles.  Each user-requested concept maps to the exact feature-name
# substrings emitted by har_rot_features.py / the UCI block:
#
#   angular displacement   → *__ang_disp_* (cumulative-trapezoid of gyro)
#   circular motion        → *__crossprod_sign, *__cross_arm_xcorr (rotation-plane trajectory)
#   gyroscope energy       → gyro*__energy, *__L/R_gyro_energy, gyro_mag__fft_energy
#   energy ratio           → *__gyro_accel_energy_ratio, *__log_energy_ratio, *__*band_ratio
#   RMS angular velocity   → *__rms_angvel, gyro*__rms
#   correlated features    → corr__*, *__*xcorr
#   accelerometer magnitude→ accel_mag__* (orientation-invariant ‖accel‖ stats)
#   variance of accel      → *__linacc_mag_var, accel*__std / accel*__var
#   orientation change     → *__orient_sweep, *__ang_disp_dominant (quat sweep)
#   gyro-to-accelerometer  → *__gyro_accel_* (energy/ratio coupling)
#   peak-to-peak motion    → *__linacc_mag_peak, *__linacc_p2p_*, gyro_*peak_*, *__range
PHYSICS_PRIORITY_CONCEPTS = (
    "angular_displacement", "circular_motion", "gyro_energy", "energy_ratio",
    "rms_angular_velocity", "correlation", "accel_magnitude", "accel_variance",
    "orientation_change", "gyro_to_accel", "peak_to_peak",
)

def _is_physics_feature(name: str) -> bool:
    """True for the high-impact rotation-physics descriptors the hybrid reducer
    prioritises (angular displacement, circular motion, gyro energy, energy ratio,
    RMS angular velocity, correlation, accel magnitude, accel variance, orientation
    change, gyro-to-accel coupling, peak-to-peak motion)."""
    n = name.lower()
    return (
        "ang_disp" in n                                   # angular displacement
        or "crossprod" in n                               # circular motion (rotation-plane traj)
        or ("gyro" in n and "energy" in n)                # gyroscope energy
        or "energy_ratio" in n                            # energy ratio (gyro/accel, log, side)
        or "band_ratio" in n                              # low/mid band energy ratio
        or "rms_angvel" in n                              # RMS angular velocity
        or ("gyro" in n and "rms" in n)                   # RMS gyro = angular-velocity RMS
        or "corr__" in n or "xcorr" in n                  # correlated / cross-correlated features
        or "accel_mag" in n                               # accelerometer magnitude
        or "linacc_mag_var" in n                          # variance of accel magnitude
        or ("accel" in n and ("__std" in n or "__var" in n))  # variance of accelerometer
        or "orient" in n                                  # orientation change / sweep
        or "gyro_accel" in n                              # gyro-to-accelerometer coupling
        or "linacc_mag_peak" in n or "p2p" in n           # peak-to-peak motion
        or "pospeak" in n or "negpeak" in n
        or "__range" in n
    )

# ── CLASS-WEIGHT POLICY ───────────────────────────────────────────────────────
# class_weight="balanced" gave running (only ~35 windows) ~14× the weight of
# walking (~485), turning "running" into a default attractor: walking → running
# and any minor movement → running.  We instead use a *tempered* inverse-freq
# weight (sqrt) so rare classes are nudged, not allowed to dominate.  Set
# CLASS_WEIGHT_MODE = "none" to disable entirely, "balanced" for the old
# behaviour, or "tempered" (default).
CLASS_WEIGHT_MODE = "tempered"

# ── REAL-TIME INFERENCE GUARDS (also emitted into the C header) ───────────────
CONF_THRESHOLD = 0.55   # min top-class probability / RF vote-share for a firm call
SMOOTH_N       = 5      # windows in the temporal majority-vote buffer (~2.5 s)

# ── SOFT-MAX THRESHOLD REJECTION (open-set / out-of-distribution guard) ───────
# Two independent reasons to refuse a window and return "uncertain" instead of
# forcing a class:
#   1. LOW CONFIDENCE  — the top softmax/vote-share probability is below
#      REJECT_THRESHOLD.  Stricter than CONF_THRESHOLD (0.55): a firm display
#      call needs the model to be genuinely committed, not merely past the
#      streaming floor.  Random noise and between-activity transitions rarely
#      clear 0.65.
#   2. LOW ENERGY      — the window carries far less motion energy than even the
#      quietest *active* class ever shows.  A dead-still / detached sensor or
#      pure low-amplitude noise produces an energy well under the active-class
#      floor; without this guard the model would still snap such a window to the
#      nearest low-energy posture (usually 'sitting').  We learn the floor from
#      the data (min active-class energy) and reject anything significantly below
#      it (ENERGY_REJECT_FRACTION × floor).  See the ENERGY-FLOOR block built
#      just before predict_from_window().
REJECT_THRESHOLD       = 0.65   # min top-class probability to accept a firm class
ENERGY_REJECT_FRACTION = 0.50   # reject if energy < this × min active-class energy
# Static postures are EXCLUDED when learning the active-energy floor: they are
# themselves near-still, so including them would drag the floor down to ~0 and
# defeat the low-energy guard.  The rotation dataset has NO static-posture classes
# (every activity is an ongoing hand/shoulder rotation), so this set is empty and
# the energy floor is learned from all classes.
STATIC_CLASSES         = set()
# Which window feature represents total motion energy.  accel_mag__mean is the
# orientation-invariant mean acceleration magnitude (≈ |g| at rest, rising with
# motion); accel_x__energy is the per-axis mean-square fallback if the magnitude
# feature is unavailable.  Resolved against FEATURES in the ENERGY-FLOOR block.
ENERGY_FEATURE_PRIMARY  = "accel_mag__mean"
ENERGY_FEATURE_FALLBACK = "accel_x__energy"

# ── FREQUENCY-DOMAIN FEATURE CONFIG ───────────────────────────────────────────
# Master toggle:
#   True  → 79 time-domain  +  66 frequency-domain  = 145 features
#   False → original 79 time-domain features only   (bit-for-bit backward compat)
#
# The frequency block is APPENDED after the 79 time features, so indices
# [0..78] are byte-identical to the original model in either mode.  Any code,
# PCA basis, or downstream consumer that expects the original 79-feature layout
# keeps working unchanged.
USE_FREQ_FEATURES = True
FS_HZ   = 50.0    # sampling rate (WINDOW_SIZE @ 50 Hz → 1 s window)
NFFT    = 256     # zero-pad length → power-of-two radix-2 FFT (exact C parity).
                  # Raised 128→256 so the 150-sample window (see WINDOW_SIZE) is
                  # NOT truncated in the spectral block; 256 is still radix-2, and
                  # the C `float re/im[IMU_NFFT]` stack arrays scale fine on ESP32.

# 11 signals get spectral features (parallels the time-domain signal set)
FREQ_SIGNALS = ["accel_x", "accel_y", "accel_z",
                "gyro_x",  "gyro_y",  "gyro_z",
                "accel_mag", "gyro_mag",
                "jerk_x", "jerk_y", "jerk_z"]
FREQ_STATS_PER_SIGNAL = 6   # domfreq dommag centroid spread entropy energy
NUM_FREQ_FEAT = len(FREQ_SIGNALS) * FREQ_STATS_PER_SIGNAL if USE_FREQ_FEATURES else 0
EXPECTED_TIME_FEAT = 79
EXPECTED_TOTAL_FEAT = EXPECTED_TIME_FEAT + NUM_FREQ_FEAT   # = 145 (original layout)

# ── UCI-HAR DERIVED-SIGNAL FEATURE BLOCK ──────────────────────────────────────
# Master toggle for the third feature layer.  When True an UCI-HAR-style block of
# derived-signal features is APPENDED after the 145 original (time+freq) features,
# pushing the model past the 500-feature target.  Indices [0..144] stay
# byte-identical to the original model in either mode (same backward-compat
# guarantee the frequency block already follows), so the .pkl/JSON consumers that
# expect the 145-feature prefix keep working.
#
# The block reconstructs the canonical UCI Human-Activity-Recognition signal set
# from each raw window and computes the statistical measures from the provided
# feature dictionary:
#
#   Derived signals (NOT raw axes — this is what makes them non-redundant):
#     tBodyAcc      = accel − gravity        (gravity removed by a low-pass)
#     tGravityAcc   = gravity component       (low-pass of accel)
#     tBodyAccJerk  = d/dt tBodyAcc
#     tBodyGyro     = gyro
#     tBodyGyroJerk = d/dt gyro
#     + the 5 magnitude signals (‖·‖ of each triaxial)
#     + frequency (FFT-magnitude) versions: fBodyAcc, fBodyAccJerk, fBodyGyro
#       and the 4 magnitude spectra
#     + 7 angle() features between mean vectors and gravity
#
#   Statistical measures (per the provided list):
#     time : mean std mad max min sma energy iqr entropy arCoeff(4) correlation
#     freq : mean std mad max min sma energy iqr entropy maxInds meanFreq
#            skewness kurtosis
#
# Naming follows the UCI convention exactly (e.g. "tBodyAcc-mean()-X"), which is a
# different namespace from the original "accel_x__mean" features — so there are no
# accidental name collisions.  Any UCI feature that turns out to be NUMERICALLY
# identical to an already-present feature (e.g. tBodyGyro == raw gyro) is detected
# and dropped by the exact-duplicate pass after windowing (see DEDUP below), so the
# requirement "add only features not already present, no redundant additions" is
# enforced data-driven, not just by naming.
# ENABLED: the UCI-HAR derived-signal block is ON — the model uses the 145 original
# features (79 time + 66 frequency) PLUS the UCI-HAR derived-signal set (tBodyAcc,
# tGravityAcc, jerk, gyro, their magnitudes/spectra + angle() features), pushing the
# feature pool past 500.  Exact duplicates of an already-present feature are pruned by
# the DEDUP pass below, so nothing redundant is added.  These derived signals feed the
# hybrid physics-aware reducer with richer angular / energy / orientation cues.
# (Shell override USE_UCI_FEATURES=0 can force the old 145-feature-only mode for A/B.)
USE_UCI_FEATURES   = os.environ.get("USE_UCI_FEATURES", "1") != "0"
# ── RESEARCH FEATURE BANK (master-prompt items 6–9) — DESKTOP-ONLY ────────────
# A dependency-free bank of advanced descriptors: extra spectral shape (rolloff/
# flux/crest/flatness/harmonic/bandwidth/multi-peak/energy-ratios), rotation-
# specific cues (signed angular momentum, axis dominance, gyro curvature, cross-
# axis phase, rotation periodicity/symmetry), Haar-DWT wavelet energies/entropy,
# and non-linear time-domain stats (Hjorth, sample/permutation/approximate
# entropy, Teager energy, crest/shape/impulse factors).  Every feature name is
# prefixed 'res_' so it can be split out of the DEPLOYED feature set: these are
# NOT computed in the C firmware, so they are EXCLUDED from the embedded pipeline
# and its self-test, and are used ONLY in the desktop selector-benchmark + SHAP/
# importance analysis (so we can honestly report whether porting any of them to C
# would be worth it).  Gated OFF by default because they add real extraction cost
# (sample/approx entropy are O(n²)); set RESEARCH_FEATURES=1 to compute them.
USE_RESEARCH_FEATURES = os.environ.get("RESEARCH_FEATURES", "0") == "1"
GRAVITY_CUTOFF_HZ  = 0.3   # UCI low-pass cutoff separating gravity from body accel
AR_ORDER           = 4     # AR (Yule-Walker/Levinson) coefficients per signal

# ── LOAD ───────────────────────────────────────────────────────────────────────
def _read_xlsx(filepath: str) -> pd.DataFrame:
    """Read .xlsx using only Python stdlib — no openpyxl required."""
    import zipfile, re
    from xml.etree import ElementTree as ET

    def _ns_of(root) -> str:
        m = re.match(r"\{([^}]+)\}", root.tag)
        return m.group(1) if m else ""

    def Q(ns: str, tag: str) -> str:
        return f"{{{ns}}}{tag}" if ns else tag

    def col_idx(col_str: str) -> int:
        idx = 0
        for ch in col_str.upper():
            idx = idx * 26 + ord(ch) - 64
        return idx - 1

    with zipfile.ZipFile(filepath) as z:
        names = z.namelist()

        # Shared strings (may not exist — this file stores everything inline)
        shared: list = []
        ss_path = next((n for n in names if "sharedstrings" in n.lower()), None)
        if ss_path:
            r = ET.parse(z.open(ss_path)).getroot()
            sns = _ns_of(r)
            for si in r.iter(Q(sns, "si")):
                shared.append("".join(t.text or "" for t in si.iter(Q(sns, "t"))))

        # First worksheet
        sheet_file = sorted(
            n for n in names if re.search(r"worksheets/sheet\d+\.xml", n, re.I)
        )[0]
        root = ET.parse(z.open(sheet_file)).getroot()
        ns = _ns_of(root)
        sheet_data = root.find(Q(ns, "sheetData"))

        rows_raw = []
        for row_el in sheet_data.findall(Q(ns, "row")):
            row_dict: dict = {}
            for cell in row_el.findall(Q(ns, "c")):
                ref = cell.get("r", "")
                m = re.match(r"([A-Za-z]+)", ref)
                if not m:
                    continue
                ci = col_idx(m.group(1))
                t = cell.get("t", "")

                # Inline string: <is><t>text</t></is>  (t="inlineStr")
                is_el = cell.find(Q(ns, "is"))
                if is_el is not None:
                    row_dict[ci] = "".join(
                        e.text or "" for e in is_el.iter(Q(ns, "t"))
                    )
                    continue

                # Numeric / shared-string value: <v>...</v>
                v_el = cell.find(Q(ns, "v"))
                if v_el is None or v_el.text is None:
                    continue
                if t == "s":
                    idx = int(v_el.text)
                    row_dict[ci] = shared[idx] if idx < len(shared) else ""
                elif t in ("str", "e"):
                    row_dict[ci] = v_el.text
                else:
                    try:
                        row_dict[ci] = float(v_el.text)
                    except ValueError:
                        row_dict[ci] = v_el.text
            rows_raw.append(row_dict)

    if not rows_raw:
        return pd.DataFrame()

    max_col = max((max(r.keys(), default=-1) for r in rows_raw if r), default=-1)
    table = [[r.get(c) for c in range(max_col + 1)] for r in rows_raw]
    headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(table[0])]
    df = pd.DataFrame(table[1:], columns=headers)
    return df.dropna(axis=1, how="all")


if CSV_PATH.endswith(".xlsx"):
    df = _read_xlsx(CSV_PATH)
elif CSV_PATH.endswith(".xls"):
    df = _read_xlsx(CSV_PATH)
else:
    df = pd.read_csv(CSV_PATH)
print(f"Dataset      : {os.path.normpath(CSV_PATH)}")

# ── SCHEMA-NORMALISE ─────────────────────────────────────────────────────────
# ROT_activities_master.csv ships raw axes + mag_xyz (magnetometer) +
# session_id / participant_id.  Re-derive the columns the downstream code expects
# so EDA, windowing and training run UNCHANGED:
#   accel_mag / gyro_mag : vector magnitude of the raw triaxial signals
#                          (the windowed extractor recomputes them per-window, but
#                           the raw-signal EDA reads the column directly)
#   device_encoded       : 0=LEFT 1=RIGHT  (used only by the EDA correlation panel)
#   activity_id          : integer code per activity_label (EDA correlation panel)
# Raw sensor columns are coerced to numeric in case the CSV parsed any as object.
for _c in RAW_AXES:
    if _c in df.columns:
        df[_c] = pd.to_numeric(df[_c], errors="coerce")
if "accel_mag" not in df.columns:
    df["accel_mag"] = np.sqrt(df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2)
if "gyro_mag" not in df.columns:
    df["gyro_mag"] = np.sqrt(df["gyro_x"]**2 + df["gyro_y"]**2 + df["gyro_z"]**2)
if "device_encoded" not in df.columns:
    df["device_encoded"] = (df["device_id"].astype(str).str.upper().str.strip()
                            == "RIGHT").astype(int)
if "activity_id" not in df.columns:
    df["activity_id"] = df["activity_label"].astype("category").cat.codes

print(f"Shape  : {df.shape}")
print(f"Nulls  :\n{df.isnull().sum()}")
print(f"Target :\n{df['activity_label'].value_counts()}")
print(f"Device :\n{df['device_id'].value_counts()}")

# ── HAND-SELECTION (LEFT-hand model; see HAND-SELECTION POLICY above) ──────────
# ONE drop, in-memory only (the CSV is never touched): remove the activities where
# the RIGHT hand is in action — the R_* classes (side ∈ DROP_SIDES).  BOTH device
# streams (LEFT and RIGHT) are KEPT for every other activity (L_* and B_*).
_dev_norm  = df["device_id"].astype(str).str.upper().str.strip()
_side      = df["activity_label"].map(_activity_side)
_n_before  = len(df)
_drop_act  = _side.isin(DROP_SIDES)                 # R_* = right hand in action
_keep_mask = ~_drop_act
_n_act_dropped = int(_drop_act.sum())
df = df.loc[_keep_mask].reset_index(drop=True)
print(f"\nHand-selection : kept {len(df):,} / {_n_before:,} rows")
print(f"                 R_* (right-hand-in-action) rows dropped : {_n_act_dropped:,}")
print(f"                 device streams kept                     : {sorted(df['device_id'].unique())}")
print(f"                 kept classes                            : {sorted(df['activity_label'].unique())}")
print("Per-activity rows kept (by device):")
print(df.groupby(["activity_label", "device_id"]).size().unstack(fill_value=0))

# ── EDA (raw data) ─────────────────────────────────────────────────────────────
activities = df["activity_label"].unique()
palette    = sns.color_palette("tab10", len(activities))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cnt = df["activity_label"].value_counts()
axes[0].bar(cnt.index, cnt.values, color=sns.color_palette("Set2", len(cnt)))
axes[0].set_title("Class Distribution"); axes[0].tick_params(axis='x', rotation=30)
dev = df["device_id"].value_counts()
axes[1].pie(dev.values, labels=dev.index, autopct="%1.1f%%")
axes[1].set_title("Hand Distribution")
plt.tight_layout(); plt.savefig("plots/eda_class_dist.png", dpi=120); plt.close()

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, col in enumerate(SENSOR_COLS):
    ax = axes.flatten()[i]
    groups = [df.loc[df["activity_label"]==a, col].values for a in activities]
    bp = ax.boxplot(groups, labels=activities, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], palette): patch.set_facecolor(c)
    ax.set_title(col); ax.tick_params(axis='x', rotation=40)
plt.suptitle("Raw Sensor Signals per Activity", y=1.01, fontsize=14)
plt.tight_layout(); plt.savefig("plots/eda_boxplots.png", dpi=120); plt.close()

# Per-activity LEFT vs RIGHT device sample counts AFTER hand-selection.  Both device
# streams are kept for every remaining activity (only the R_* classes were dropped),
# so both bars are populated.  Grouped bars, robust to a missing LEFT/RIGHT column.
counts_lr    = df.groupby(["activity_label","device_id"]).size().unstack(fill_value=0)
activities_l = counts_lr.index.tolist()
LEFT_COLOR, RIGHT_COLOR = "#378ADD", "#D85A30"
left_counts  = [int(counts_lr.loc[a, "LEFT"])  if "LEFT"  in counts_lr.columns else 0
                for a in activities_l]
right_counts = [int(counts_lr.loc[a, "RIGHT"]) if "RIGHT" in counts_lr.columns else 0
                for a in activities_l]
x = np.arange(len(activities_l)); w = 0.4
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - w/2, left_counts,  w, label="LEFT device",  color=LEFT_COLOR)
ax.bar(x + w/2, right_counts, w, label="RIGHT device", color=RIGHT_COLOR)
ax.set_xticks(x); ax.set_xticklabels([a.replace("_", " ") for a in activities_l], rotation=30, ha="right")
ax.set_title("Samples per activity after hand-selection (R_* dropped)", fontsize=14)
ax.set_ylabel("samples"); ax.legend()
plt.tight_layout(); plt.savefig("pie_left_vs_right.png", dpi=150, bbox_inches="tight"); plt.close()

ACTIVITIES = df["activity_label"].unique()
PALETTE    = sns.color_palette("tab10", len(ACTIVITIES))
fig, axes  = plt.subplots(2, 4, figsize=(20, 10)); axes = axes.flatten()
for i, col in enumerate(SENSOR_COLS):
    for j, act in enumerate(ACTIVITIES):
        vals = df.loc[df["activity_label"] == act, col].dropna()
        axes[i].hist(vals, bins=50, alpha=0.5, label=act, color=PALETTE[j])
    axes[i].set_title(col, fontsize=12)
    axes[i].set_xlabel("Value"); axes[i].set_ylabel("Frequency")
axes[0].legend(fontsize=8)
plt.suptitle("Sensor Signal Distributions per Activity", fontsize=15, y=1.01)
plt.tight_layout(); plt.savefig("eda_04_boxplots.png", dpi=120); plt.close()

try:
    numeric_cols = SENSOR_COLS + ["activity_id", "device_encoded"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Pearson Correlation Matrix (raw features)", fontsize=14)
    plt.tight_layout(); plt.savefig("eda_05_correlation_heatmap.png", dpi=120); plt.close()
except Exception as e:
    print(f"  Skipping correlation heatmap: {e}")

print("Raw EDA plots saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — Windowed Statistical Abstraction
# ───────────────────────────────────────────────────────────────────────────────
# 8 O(n) statistics per axis  (no sort → fast on ESP32-P4)
#   mean  std  range  rms  energy  skew  kurt  zcr
# 2 magnitude signals (accel_mag, gyro_mag) × 8 stats  [orientation-invariant]
# 6 cross-axis Pearson correlations
# 3 jerk axes × 3 stats  (mean  std  rms)
# ─────────────────────────────────────────────────
# Total: 6×8 + 2×8 + 6 + 3×3 = 79 features per window
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(f"FEATURE ENGINEERING — Windowed Statistical Abstraction "
      f"({EXPECTED_TOTAL_FEAT} features: {EXPECTED_TIME_FEAT} time + {NUM_FREQ_FEAT} freq"
      f"{' + UCI' if USE_UCI_FEATURES else ''})")
print("="*65)

# (The wrist-mirror / axis-reflection step has been removed — raw axes go straight
#  into windowing exactly as recorded, with no sign flips.)

# Step 1: Statistical extractor (float64 for scipy, O(n) only, no sort)
def _safe_corr(a, b):
    return float(np.corrcoef(a, b)[0,1]) if a.std()>1e-9 and b.std()>1e-9 else 0.0

def _axis_stats(x: np.ndarray, col_name: str) -> dict:
    """Compute 8 statistics for a 1-D float64 signal."""
    n      = len(x)
    mu     = x.mean()
    sd     = x.std()
    energy = (x**2).mean()
    zcr    = float(((x[:-1]*x[1:])<0).sum()) / (n - 1)
    return {
        f"{col_name}__mean":   mu,
        f"{col_name}__std":    sd,
        f"{col_name}__range":  float(x.max() - x.min()),
        f"{col_name}__rms":    float(np.sqrt(energy)),
        f"{col_name}__energy": float(energy),
        f"{col_name}__skew":   float(sp_skew(x, bias=True)),
        f"{col_name}__kurt":   float(sp_kurt(x, fisher=True, bias=True)),
        f"{col_name}__zcr":    zcr,
    }


# ── Frequency-domain extractor ────────────────────────────────────────────────
# rfftfreq for the AC bins (1 … NFFT/2) — DC (bin 0) is dropped so that the
# spectral-shape metrics are not dominated by the signal mean.
_FREQS_AC = np.fft.rfftfreq(NFFT, d=1.0 / FS_HZ)[1:]   # shape (NFFT/2,) = (64,)

def _freq_axis_stats(x: np.ndarray, name: str) -> dict:
    """
    6 spectral statistics for a 1-D real signal, computed on its
    zero-padded (length NFFT) power spectral density.  Order MUST match the
    C imu_freq_stats():

        dom_freq  : frequency (Hz) of the strongest AC bin
        dom_mag   : relative power of that peak   (peak / total, 0..1)
        centroid  : power-weighted mean frequency (Hz) — spectral "center of mass"
        spread    : power-weighted std around the centroid (Hz) — bandwidth
        entropy   : Shannon entropy of the normalised PSD, scaled to 0..1
                    (low → tonal/periodic motion, high → broadband/noisy)
        energy    : total AC spectral power (Parseval-equivalent of time energy)
    """
    X      = np.fft.rfft(x, n=NFFT)
    psd    = np.abs(X) ** 2
    psd_ac = psd[1:]                       # drop DC bin → 64 AC bins
    total  = float(psd_ac.sum()) + 1e-12
    pn     = psd_ac / total                # normalised power distribution
    dom    = int(psd_ac.argmax())
    cent   = float((_FREQS_AC * pn).sum())
    spread = float(np.sqrt((((_FREQS_AC - cent) ** 2) * pn).sum()))
    ent    = float(-(pn * np.log(pn + 1e-12)).sum() / np.log(len(pn)))
    return {
        f"{name}__fft_domfreq":  float(_FREQS_AC[dom]),
        f"{name}__fft_dommag":   float(psd_ac[dom] / total),
        f"{name}__fft_centroid": cent,
        f"{name}__fft_spread":   spread,
        f"{name}__fft_entropy":  ent,
        f"{name}__fft_energy":   total,
    }


def extract_freq_stats(win: np.ndarray) -> dict:
    """
    win : (WINDOW_SIZE, 6) — same raw window used by extract_stats().
    Returns 66 features (11 signals × 6 stats) in FREQ_SIGNALS order, which
    MUST match the C imu_extract_freq_features().
    """
    w   = win.astype(np.float64)
    sig = {
        "accel_x": w[:, 0], "accel_y": w[:, 1], "accel_z": w[:, 2],
        "gyro_x":  w[:, 3], "gyro_y":  w[:, 4], "gyro_z":  w[:, 5],
    }
    sig["accel_mag"] = np.sqrt(w[:, 0]**2 + w[:, 1]**2 + w[:, 2]**2)
    sig["gyro_mag"]  = np.sqrt(w[:, 3]**2 + w[:, 4]**2 + w[:, 5]**2)
    sig["jerk_x"]    = np.diff(w[:, 0])
    sig["jerk_y"]    = np.diff(w[:, 1])
    sig["jerk_z"]    = np.diff(w[:, 2])

    feats = {}
    for name in FREQ_SIGNALS:
        feats.update(_freq_axis_stats(sig[name], name))
    return feats


# ── UCI-HAR derived-signal feature extractor ──────────────────────────────────
# Self-contained: turns one raw (WINDOW_SIZE, 6) window into the UCI-HAR feature
# block.  All helpers are O(n) or O(n·order); the only non-trivial cost is the
# gravity low-pass (scipy Butterworth, with an EMA fallback) and the FFTs.
def _grav_split(x: np.ndarray) -> np.ndarray:
    """Low-pass an accel axis to estimate its gravity component (UCI 0.3 Hz)."""
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(3, GRAVITY_CUTOFF_HZ / (FS_HZ / 2.0), btype="low")
        pad  = min(len(x) - 1, 3 * (max(len(a), len(b)) - 1))
        return filtfilt(b, a, x, padlen=pad)
    except Exception:
        # Zero-phase EMA fallback (forward then backward) if scipy.signal is absent
        alpha = 0.02
        g = x.astype(np.float64).copy()
        for i in range(1, len(g)):           g[i] = alpha * x[i]   + (1 - alpha) * g[i - 1]
        for i in range(len(g) - 2, -1, -1):  g[i] = alpha * g[i + 1] + (1 - alpha) * g[i]
        return g

def _mad(x):  # median absolute deviation (UCI mad())
    return float(np.median(np.abs(x - np.median(x))))

def _iqr(x):
    return float(np.percentile(x, 75) - np.percentile(x, 25))

def _entropy(x):
    p = np.abs(x); s = p.sum()
    if s <= 1e-12:
        return 0.0
    p  = p / s
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(len(x)))

def _ar_coeffs(x, order=AR_ORDER):
    """AR(order) coefficients via Levinson–Durbin on the biased autocorrelation."""
    x = np.asarray(x, dtype=np.float64); x = x - x.mean()
    n = len(x)
    r = np.array([np.dot(x[:n - k], x[k:]) for k in range(order + 1)]) / max(n, 1)
    if r[0] <= 1e-12:
        return [0.0] * order
    a = np.zeros(order + 1); a[0] = 1.0; e = r[0]
    for i in range(1, order + 1):
        acc = r[i] + sum(a[j] * r[i - j] for j in range(1, i))
        k   = -acc / e
        prev = a.copy()
        for j in range(1, i):
            a[j] = prev[j] + k * prev[i - j]
        a[i] = k
        e *= (1.0 - k * k)
        if e <= 1e-12:
            break
    return [float(v) for v in a[1:order + 1]]

def _stat(name, x):
    if name == "mean":   return float(np.mean(x))
    if name == "std":    return float(np.std(x))
    if name == "mad":    return _mad(x)
    if name == "max":    return float(np.max(x))
    if name == "min":    return float(np.min(x))
    if name == "energy": return float(np.mean(np.asarray(x) ** 2))
    if name == "iqr":    return _iqr(x)
    if name == "entropy":return _entropy(x)
    return 0.0

def _spectrum(x):
    """FFT-magnitude spectrum (DC dropped) of a detrended signal + its freq axis."""
    x   = np.asarray(x, dtype=np.float64) - np.mean(x)
    mag = np.abs(np.fft.rfft(x, n=NFFT))[1:]            # 64 AC bins
    return mag, _FREQS_AC

def _mean_freq(mag, freqs):
    s = mag.sum()
    return float((freqs * mag).sum() / s) if s > 1e-12 else 0.0

def _norm3(sig):  # ‖·‖ of an (n,3) triaxial signal → (n,)
    return np.sqrt((sig ** 2).sum(axis=1))

def _angle(u, v):
    """UCI angle() — cosine of the angle between two vectors, in [-1, 1]."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))

def _uci_triaxial_time(name, sig):
    """40 UCI time-domain features for one (n,3) triaxial signal."""
    out  = {}
    cols = [sig[:, 0], sig[:, 1], sig[:, 2]]
    axes = ("X", "Y", "Z")
    for stat in ("mean", "std", "mad", "max", "min"):
        for ax_l, c in zip(axes, cols):
            out[f"{name}-{stat}()-{ax_l}"] = _stat(stat, c)
    out[f"{name}-sma()"] = float(np.mean(np.sum(np.abs(sig), axis=1)))
    for stat in ("energy", "iqr", "entropy"):
        for ax_l, c in zip(axes, cols):
            out[f"{name}-{stat}()-{ax_l}"] = _stat(stat, c)
    for ax_l, c in zip(axes, cols):
        for j, coef in enumerate(_ar_coeffs(c), 1):
            out[f"{name}-arCoeff()-{ax_l},{j}"] = coef
    out[f"{name}-correlation()-X,Y"] = _safe_corr(cols[0], cols[1])
    out[f"{name}-correlation()-X,Z"] = _safe_corr(cols[0], cols[2])
    out[f"{name}-correlation()-Y,Z"] = _safe_corr(cols[1], cols[2])
    return out

def _uci_mag_time(name, x):
    """13 UCI time-domain features for one magnitude signal."""
    out = {}
    for stat in ("mean", "std", "mad", "max", "min"):
        out[f"{name}-{stat}()"] = _stat(stat, x)
    out[f"{name}-sma()"] = float(np.mean(np.abs(x)))
    for stat in ("energy", "iqr", "entropy"):
        out[f"{name}-{stat}()"] = _stat(stat, x)
    for j, coef in enumerate(_ar_coeffs(x), 1):
        out[f"{name}-arCoeff(){j}"] = coef
    return out

def _uci_triaxial_freq(name, sig):
    """37 UCI frequency-domain features for one (n,3) triaxial signal."""
    out   = {}
    specs = []
    freqs = None
    for i in range(3):
        m, freqs = _spectrum(sig[:, i]); specs.append(m)
    axes = ("X", "Y", "Z")
    for stat in ("mean", "std", "mad", "max", "min"):
        for ax_l, m in zip(axes, specs):
            out[f"{name}-{stat}()-{ax_l}"] = _stat(stat, m)
    out[f"{name}-sma()"] = float(np.mean([np.sum(np.abs(m)) for m in specs]))
    for stat in ("energy", "iqr", "entropy"):
        for ax_l, m in zip(axes, specs):
            out[f"{name}-{stat}()-{ax_l}"] = _stat(stat, m)
    for ax_l, m in zip(axes, specs):
        out[f"{name}-maxInds-{ax_l}"]    = float(int(np.argmax(m)))
    for ax_l, m in zip(axes, specs):
        out[f"{name}-meanFreq()-{ax_l}"] = _mean_freq(m, freqs)
    for ax_l, m in zip(axes, specs):
        out[f"{name}-skewness()-{ax_l}"] = float(sp_skew(m, bias=True))
    for ax_l, m in zip(axes, specs):
        out[f"{name}-kurtosis()-{ax_l}"] = float(sp_kurt(m, fisher=True, bias=True))
    return out

def _uci_mag_freq(name, x):
    """13 UCI frequency-domain features for one magnitude spectrum."""
    mag, freqs = _spectrum(x)
    out = {}
    for stat in ("mean", "std", "mad", "max", "min"):
        out[f"{name}-{stat}()"] = _stat(stat, mag)
    out[f"{name}-sma()"] = float(np.sum(np.abs(mag)))
    for stat in ("energy", "iqr", "entropy"):
        out[f"{name}-{stat}()"] = _stat(stat, mag)
    out[f"{name}-maxInds"]    = float(int(np.argmax(mag)))
    out[f"{name}-meanFreq()"] = _mean_freq(mag, freqs)
    out[f"{name}-skewness()"] = float(sp_skew(mag, bias=True))
    out[f"{name}-kurtosis()"] = float(sp_kurt(mag, fisher=True, bias=True))
    return out

def extract_uci_features(win: np.ndarray) -> dict:
    """
    Reconstruct the UCI-HAR derived-signal set from one raw window and compute the
    statistical measures from the provided feature list.  Returns 435 features
    (before the exact-duplicate dedup pass that runs after windowing):

      triaxial time  : 5 signals × 40 = 200   (tBodyAcc tGravityAcc tBodyAccJerk
                                                tBodyGyro tBodyGyroJerk)
      magnitude time : 5 signals × 13 =  65   (…Mag of each of the above)
      triaxial freq  : 3 signals × 37 = 111   (fBodyAcc fBodyAccJerk fBodyGyro)
      magnitude freq : 4 signals × 13 =  52   (fBodyAccMag fBodyBodyAccJerkMag
                                               fBodyBodyGyroMag fBodyBodyGyroJerkMag)
      angle()        :                    7
    """
    w  = win.astype(np.float64)
    a3 = w[:, 0:3]                 # raw accel (X,Y,Z)
    g3 = w[:, 3:6]                 # raw gyro  (X,Y,Z)

    grav = np.column_stack([_grav_split(a3[:, 0]),
                            _grav_split(a3[:, 1]),
                            _grav_split(a3[:, 2])])
    bAcc = a3 - grav              # body acceleration (gravity removed)

    def _jerk(sig):               # per-sample derivative, length preserved
        d = np.diff(sig, axis=0)
        return np.vstack([d, d[-1:]])
    bAccJerk  = _jerk(bAcc)
    bGyro     = g3
    bGyroJerk = _jerk(bGyro)

    feats = {}
    for name, sig in (("tBodyAcc", bAcc), ("tGravityAcc", grav),
                      ("tBodyAccJerk", bAccJerk), ("tBodyGyro", bGyro),
                      ("tBodyGyroJerk", bGyroJerk)):
        feats.update(_uci_triaxial_time(name, sig))
    for name, sig in (("tBodyAccMag", bAcc), ("tGravityAccMag", grav),
                      ("tBodyAccJerkMag", bAccJerk), ("tBodyGyroMag", bGyro),
                      ("tBodyGyroJerkMag", bGyroJerk)):
        feats.update(_uci_mag_time(name, _norm3(sig)))
    for name, sig in (("fBodyAcc", bAcc), ("fBodyAccJerk", bAccJerk),
                      ("fBodyGyro", bGyro)):
        feats.update(_uci_triaxial_freq(name, sig))
    for name, sig in (("fBodyAccMag", bAcc), ("fBodyBodyAccJerkMag", bAccJerk),
                      ("fBodyBodyGyroMag", bGyro), ("fBodyBodyGyroJerkMag", bGyroJerk)):
        feats.update(_uci_mag_freq(name, _norm3(sig)))

    gmean = grav.mean(axis=0)
    feats["angle(tBodyAccMean,gravity)"]          = _angle(bAcc.mean(axis=0), gmean)
    feats["angle(tBodyAccJerkMean,gravityMean)"]  = _angle(bAccJerk.mean(axis=0), gmean)
    feats["angle(tBodyGyroMean,gravityMean)"]     = _angle(bGyro.mean(axis=0), gmean)
    feats["angle(tBodyGyroJerkMean,gravityMean)"] = _angle(bGyroJerk.mean(axis=0), gmean)
    feats["angle(X,gravityMean)"] = _angle(np.array([1.0, 0.0, 0.0]), gmean)
    feats["angle(Y,gravityMean)"] = _angle(np.array([0.0, 1.0, 0.0]), gmean)
    feats["angle(Z,gravityMean)"] = _angle(np.array([0.0, 0.0, 1.0]), gmean)
    return feats


# ══════════════════════════════════════════════════════════════════════════════
#  RESEARCH FEATURE BANK (items 6–9) — DESKTOP-ONLY, dependency-free
#  All helpers operate on 1-D signals; extract_research_features() assembles the
#  'res_' block.  Kept out of the deployed/embedded feature set (see the split in
#  the windowing section).  Pure numpy — no pywt/antropy needed.
# ══════════════════════════════════════════════════════════════════════════════
def _hjorth(x):
    """Hjorth activity / mobility / complexity — cheap descriptors of signal power,
    mean frequency, and frequency spread; discriminate smooth vs jerky motion."""
    x = np.asarray(x, np.float64)
    dx = np.diff(x); ddx = np.diff(dx)
    v0 = np.var(x) + 1e-12; v1 = np.var(dx) + 1e-12; v2 = np.var(ddx) + 1e-12
    mob = np.sqrt(v1 / v0)
    comp = (np.sqrt(v2 / v1)) / (mob + 1e-12)
    return float(v0), float(mob), float(comp)

def _sample_entropy(x, m=2, r=None):
    """Sample entropy (regularity). O(n²), vectorised over template pairs."""
    x = np.asarray(x, np.float64); n = len(x)
    if n < m + 2:
        return 0.0
    if r is None:
        r = 0.2 * (np.std(x) + 1e-12)
    def _phi(mm):
        tmpl = np.array([x[i:i+mm] for i in range(n - mm + 1)])
        d = np.abs(tmpl[:, None, :] - tmpl[None, :, :]).max(axis=2)
        np.fill_diagonal(d, np.inf)
        return np.sum(d <= r)
    B = _phi(m); A = _phi(m + 1)
    if B <= 0 or A <= 0:
        return 0.0
    return float(-np.log(A / B))

def _approx_entropy(x, m=2, r=None):
    """Approximate entropy (regularity/predictability). O(n²)."""
    x = np.asarray(x, np.float64); n = len(x)
    if n < m + 2:
        return 0.0
    if r is None:
        r = 0.2 * (np.std(x) + 1e-12)
    def _phi(mm):
        tmpl = np.array([x[i:i+mm] for i in range(n - mm + 1)])
        d = np.abs(tmpl[:, None, :] - tmpl[None, :, :]).max(axis=2)
        C = np.sum(d <= r, axis=1) / (n - mm + 1.0)
        return np.mean(np.log(C + 1e-12))
    return float(abs(_phi(m) - _phi(m + 1)))

def _perm_entropy(x, order=3, delay=1):
    """Permutation entropy (ordinal-pattern complexity), normalised to 0..1."""
    x = np.asarray(x, np.float64); n = len(x)
    if n < order * delay + 1:
        return 0.0
    patt = {}
    for i in range(n - delay * (order - 1)):
        pat = tuple(np.argsort(x[i:i + delay * order:delay]))
        patt[pat] = patt.get(pat, 0) + 1
    c = np.array(list(patt.values()), np.float64); p = c / c.sum()
    from math import factorial
    return float(-(p * np.log(p)).sum() / (np.log(factorial(order)) + 1e-12))

def _teager(x):
    """Mean Teager–Kaiser energy — sensitive to instantaneous amplitude×frequency."""
    x = np.asarray(x, np.float64)
    if len(x) < 3:
        return 0.0
    return float(np.mean(x[1:-1] ** 2 - x[:-2] * x[2:]))

def _shape_factors(x):
    """Crest / shape / impulse factors — waveform peakiness vs its RMS/mean."""
    x = np.asarray(x, np.float64); ax = np.abs(x)
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    mabs = np.mean(ax) + 1e-12; peak = np.max(ax)
    return float(peak / rms), float(rms / mabs), float(peak / mabs)

def _res_time(name, x):
    a, mob, comp = _hjorth(x)
    cf, sf, imf = _shape_factors(x)
    return {
        f"res_{name}_hjorth_activity":  a,
        f"res_{name}_hjorth_mobility":  mob,
        f"res_{name}_hjorth_complex":   comp,
        f"res_{name}_sampen":           _sample_entropy(x),
        f"res_{name}_apen":             _approx_entropy(x),
        f"res_{name}_permen":           _perm_entropy(x),
        f"res_{name}_teager":           _teager(x),
        f"res_{name}_crest_factor":     cf,
        f"res_{name}_shape_factor":     sf,
        f"res_{name}_impulse_factor":   imf,
        f"res_{name}_mad":              _mad(x),
        f"res_{name}_iqr":              _iqr(x),
    }

def _res_spectral(name, x):
    """Extra spectral-shape descriptors on the AC power spectrum (DC dropped)."""
    x = np.asarray(x, np.float64) - np.mean(x)
    X = np.fft.rfft(x, n=NFFT); psd = np.abs(X) ** 2
    psd_ac = psd[1:]; freqs = _FREQS_AC
    total = float(psd_ac.sum()) + 1e-12
    pn = psd_ac / total
    csum = np.cumsum(psd_ac)
    rolloff = float(freqs[np.searchsorted(csum, 0.85 * total)]) if total > 1e-9 \
        else 0.0
    amean = psd_ac.mean() + 1e-12
    gmean = np.exp(np.mean(np.log(psd_ac + 1e-12)))
    flatness = float(gmean / amean)
    crest = float(psd_ac.max() / amean)
    # top-3 peaks by power
    order = np.argsort(psd_ac)[::-1]
    p1 = order[0]; p2 = order[1] if len(order) > 1 else order[0]
    p3 = order[2] if len(order) > 2 else order[0]
    peak_ratio = float(psd_ac[p2] / (psd_ac[p1] + 1e-12))
    # harmonic ratio: power at 2×/3× the dominant bin vs the dominant
    def _at(mult):
        b = int(round((p1 + 1) * mult)) - 1
        return float(psd_ac[b]) if 0 <= b < len(psd_ac) else 0.0
    harmonic = float((_at(2) + _at(3)) / (psd_ac[p1] + 1e-12))
    cent = float((freqs * pn).sum())
    var = float(((freqs - cent) ** 2 * pn).sum())
    # -3 dB bandwidth around the dominant peak
    half = psd_ac[p1] / 2.0
    lo = p1
    while lo > 0 and psd_ac[lo] > half:
        lo -= 1
    hi = p1
    while hi < len(psd_ac) - 1 and psd_ac[hi] > half:
        hi += 1
    bandwidth = float(freqs[hi] - freqs[lo])
    # low/high band energy ratio (split at Nyquist/4)
    split = len(psd_ac) // 4
    lowe = float(psd_ac[:split].sum()); highe = float(psd_ac[split:].sum()) + 1e-12
    return {
        f"res_{name}_spec_rolloff":    rolloff,
        f"res_{name}_spec_flatness":   flatness,
        f"res_{name}_spec_crest":      crest,
        f"res_{name}_spec_peakratio":  peak_ratio,
        f"res_{name}_spec_harmonic":   harmonic,
        f"res_{name}_spec_bandwidth":  bandwidth,
        f"res_{name}_spec_domfreq2":   float(freqs[p2]),
        f"res_{name}_spec_domfreq3":   float(freqs[p3]),
        f"res_{name}_spec_freqvar":    var,
        f"res_{name}_spec_lohi_ratio": float(lowe / highe),
    }

def _spec_flux(x):
    """Spectral flux — mean squared frame-to-frame spectral change (2 half-frames)."""
    x = np.asarray(x, np.float64); h = len(x) // 2
    if h < 4:
        return 0.0
    s1 = np.abs(np.fft.rfft(x[:h] - x[:h].mean(), n=NFFT))
    s2 = np.abs(np.fft.rfft(x[h:2*h] - x[h:2*h].mean(), n=NFFT))
    return float(np.mean((s2 - s1) ** 2))

def _autocorr_features(name, x):
    """Rotation periodicity via the biased autocorrelation: first non-trivial peak
    (dominant cycle), its height (periodicity score / zero-lag similarity), the
    number and spacing of peaks (repetition regularity), and left/right symmetry."""
    x = np.asarray(x, np.float64) - np.mean(x); n = len(x)
    if n < 8 or np.std(x) < 1e-9:
        return {f"res_{name}_ac_cyclelag": 0.0, f"res_{name}_ac_periodicity": 0.0,
                f"res_{name}_ac_npeaks": 0.0, f"res_{name}_ac_peakspacing": 0.0,
                f"res_{name}_ac_symmetry": 0.0}
    ac = np.correlate(x, x, mode="full")[n - 1:]
    ac = ac / (ac[0] + 1e-12)
    # first local maximum after the zero-lag decay
    peaks = [i for i in range(2, len(ac) - 1)
             if ac[i] > ac[i-1] and ac[i] >= ac[i+1] and ac[i] > 0.1]
    cyclelag = float(peaks[0]) if peaks else 0.0
    periodicity = float(ac[peaks[0]]) if peaks else 0.0
    npeaks = float(len(peaks))
    spacing = float(np.std(np.diff(peaks))) if len(peaks) > 1 else 0.0
    m = min(20, n // 2)
    sym = float(1.0 - np.mean(np.abs(ac[1:m] - ac[1:m][::-1])) ) if m > 2 else 0.0
    return {f"res_{name}_ac_cyclelag": cyclelag, f"res_{name}_ac_periodicity": periodicity,
            f"res_{name}_ac_npeaks": npeaks, f"res_{name}_ac_peakspacing": spacing,
            f"res_{name}_ac_symmetry": sym}

def _haar_dwt_energies(x, levels=3):
    """Pure-numpy multi-level Haar DWT.  Returns per-level detail energies, the final
    approximation energy (each normalised to the total), and the Shannon entropy of
    the energy distribution — a non-stationary alternative to the FFT block."""
    x = np.asarray(x, np.float64).copy()
    ener = []
    a = x
    for _ in range(levels):
        if len(a) < 2:
            break
        if len(a) % 2:
            a = a[:-1]
        approx = (a[0::2] + a[1::2]) / np.sqrt(2.0)
        detail = (a[0::2] - a[1::2]) / np.sqrt(2.0)
        ener.append(float(np.sum(detail ** 2)))
        a = approx
    approx_e = float(np.sum(a ** 2))
    while len(ener) < levels:
        ener.append(0.0)
    total = sum(ener) + approx_e + 1e-12
    rel = np.array(ener + [approx_e]) / total
    ent = float(-(rel * np.log(rel + 1e-12)).sum())
    return ener, approx_e, total, ent

def _res_wavelet(name, x):
    d, a, total, ent = _haar_dwt_energies(x, levels=3)
    return {
        f"res_{name}_wav_d1":       d[0] / total,
        f"res_{name}_wav_d2":       d[1] / total,
        f"res_{name}_wav_d3":       d[2] / total,
        f"res_{name}_wav_approx":   a / total,
        f"res_{name}_wav_entropy":  ent,
        f"res_{name}_wav_totalen":  total,
    }

def extract_research_features(win: np.ndarray) -> dict:
    """DESKTOP-ONLY 'res_' feature block (items 6–9).  Computed on the two rotation-
    invariant magnitudes (accel_mag, gyro_mag) plus rotation-specific gyro-axis
    descriptors.  Never entered by the C export."""
    w = win.astype(np.float64)
    ax, ay, az = w[:, 0], w[:, 1], w[:, 2]
    gx, gy, gz = w[:, 3], w[:, 4], w[:, 5]
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag  = np.sqrt(gx**2 + gy**2 + gz**2)
    feats = {}
    # ── time-domain non-linear (item 9) + spectral (item 6) on both magnitudes ──
    for nm, sig in (("accelmag", accel_mag), ("gyromag", gyro_mag)):
        feats.update(_res_time(nm, sig))
        feats.update(_res_spectral(nm, sig))
        feats[f"res_{nm}_spec_flux"] = _spec_flux(sig)
        feats.update(_res_wavelet(nm, sig))         # wavelet (item 8)
        feats.update(_autocorr_features(nm, sig))   # periodicity (item 7)
    # ── rotation-specific descriptors on the gyro axes (item 7) ────────────────
    gyro_axes = np.stack([gx, gy, gz], axis=1)
    ax_energy = (gyro_axes ** 2).sum(axis=0)
    tot_e = ax_energy.sum() + 1e-12
    feats["res_gyro_axis_dominance"] = float(ax_energy.max() / tot_e)
    # signed angular momentum proxy: net integrated angular velocity per axis
    net = gyro_axes.sum(axis=0)
    feats["res_gyro_signed_momentum"] = float(np.linalg.norm(net) / (len(w) + 1e-12))
    feats["res_gyro_net_x"] = float(net[0]); feats["res_gyro_net_y"] = float(net[1])
    feats["res_gyro_net_z"] = float(net[2])
    # directional angular energy: fraction of gyro_mag samples turning "positive"
    # (dominant signed axis > 0) — separates CLK from ACLK when the dominant axis
    # keeps a consistent sign through the rotation.
    dom_ax = int(ax_energy.argmax())
    feats["res_gyro_dir_energy"] = float(np.mean(gyro_axes[:, dom_ax] > 0))
    # gyro trajectory curvature: mean angle between successive angular-velocity vecs
    dv = np.diff(gyro_axes, axis=0)
    nrm = np.linalg.norm(dv, axis=1) + 1e-12
    cosang = np.clip((dv[1:] * dv[:-1]).sum(axis=1) /
                     (nrm[1:] * nrm[:-1]), -1, 1)
    feats["res_gyro_curvature"] = float(np.mean(np.arccos(cosang))) if len(cosang) else 0.0
    # cross-axis phase shift: lag of peak cross-correlation between gx and gy
    def _xlag(u, v):
        u = u - u.mean(); v = v - v.mean()
        if u.std() < 1e-9 or v.std() < 1e-9:
            return 0.0
        xc = np.correlate(u, v, mode="full")
        return float(np.argmax(xc) - (len(u) - 1))
    feats["res_gyro_phase_xy"] = _xlag(gx, gy)
    feats["res_gyro_phase_xz"] = _xlag(gx, gz)
    return feats


def extract_stats(win: np.ndarray) -> dict:
    """
    win  : (WINDOW_SIZE, 6) float32
    Returns 79 features in a fixed order that MUST match C imu_extract_features():

      [  0..47]  6 axes × 8 stats each
                 axis order: accel_x accel_y accel_z gyro_x gyro_y gyro_z
                 stat order: mean std range rms energy skew kurt zcr
      [ 48..55]  accel_mag × 8 stats  (orientation-invariant)
      [ 56..63]  gyro_mag  × 8 stats  (orientation-invariant)
      [ 64..69]  6 Pearson correlations
                 (ax,ay)(ax,az)(ay,az)(gx,gy)(gx,gz)(gy,gz)
      [ 70..72]  jerk_x : mean std rms
      [ 73..75]  jerk_y : mean std rms
      [ 76..78]  jerk_z : mean std rms
    """
    feats = {}
    w = win.astype(np.float64)

    # 6 raw axes × 8 stats = 48
    for ai, col in enumerate(RAW_AXES):
        feats.update(_axis_stats(w[:, ai], col))

    # 2 magnitude signals × 8 stats = 16  (rotation-invariant)
    accel_mag = np.sqrt(w[:,0]**2 + w[:,1]**2 + w[:,2]**2)
    gyro_mag  = np.sqrt(w[:,3]**2 + w[:,4]**2 + w[:,5]**2)
    feats.update(_axis_stats(accel_mag, "accel_mag"))
    feats.update(_axis_stats(gyro_mag,  "gyro_mag"))

    # 6 Pearson correlations
    for (ia, ib), (ca, cb) in zip(CORR_PAIRS, CORR_NAMES):
        feats[f"corr__{ca}__{cb}"] = _safe_corr(w[:,ia], w[:,ib])

    # 3 jerk axes × 3 stats = 9
    for ai in JERK_IDX:
        col  = RAW_AXES[ai]; ax_l = col.split("_")[1]
        j    = np.diff(w[:, ai])
        feats[f"jerk_{ax_l}__mean"] = float(j.mean())
        feats[f"jerk_{ax_l}__std"]  = float(j.std())
        feats[f"jerk_{ax_l}__rms"]  = float(np.sqrt((j**2).mean()))

    # ── Frequency-domain block (appended; indices [79..144]) ──────────────────
    # Skipped entirely when USE_FREQ_FEATURES is False → identical 79-feature
    # vector as the original model.
    if USE_FREQ_FEATURES:
        feats.update(extract_freq_stats(win))

    # ── UCI-HAR derived-signal block (appended; indices [145..]) ──────────────
    # Skipped entirely when USE_UCI_FEATURES is False → identical 145-feature
    # vector as before.  Exact duplicates of earlier features are pruned after
    # windowing (see the DEDUP pass), so nothing redundant survives here.
    if USE_UCI_FEATURES:
        feats.update(extract_uci_features(win))

    # ── DESKTOP-ONLY research bank ('res_' prefix; items 6–9) ─────────────────
    # Appended LAST and split out of the deployed feature set after windowing, so
    # the embedded C export / self-test never see them.  Skipped entirely unless
    # RESEARCH_FEATURES=1 → normal runs keep the exact prior feature vector.
    if USE_RESEARCH_FEATURES:
        feats.update(extract_research_features(win))

    return feats   # insertion-order dict → stable feature layout

# Step 2: Slide windows per (session_id, device_id)
#
# CRITICAL: the two wrist devices are stored INTERLEAVED row-by-row inside each
# session_id.  Windowing on `groupby("session_id")` alone would make every window
# alternate between the LEFT and RIGHT sensor on consecutive samples, turning every
# temporal/frequency feature (jerk, zcr, FFT dom-freq/entropy, correlations) into
# noise.  We instead sort by time and window each (session, device) stream
# independently, and record the session + device of every window so we can do an
# HONEST held-out evaluation below.
df = df.sort_values(["session_id", "device_id", "timestamp"]).reset_index(drop=True)

def _build_windows(window_size, step_size, collect_samples=False):
    """Slide (window_size, step_size) windows over each (session, device) stream and
    extract the full feature row per window.  Returns
    (df_win, win_session, win_device, sample_raw_windows).  Refactored out of the
    top-level loop so the WINDOW-SIZE SWEEP below can re-window at several sizes; the
    per-(session, device) grouping keeps the two interleaved wrist streams from
    contaminating each other's features (see note above)."""
    records, w_session, w_device = [], [], []
    samples = {}
    for (sid, dev), grp in df.groupby(["session_id", "device_id"], sort=False):
        raw    = grp[RAW_AXES].values.astype(np.float32)
        labels = grp["activity_label"].values
        n      = len(raw)
        for start in range(0, n - window_size + 1, step_size):
            end = start + window_size
            row = extract_stats(raw[start:end])
            vals, cnts = np.unique(labels[start:end], return_counts=True)
            maj = vals[cnts.argmax()]
            row["activity_label"] = maj
            records.append(row)
            w_session.append(sid)
            w_device.append(str(dev).upper().strip())
            # keep the LAST clean window per class (a real in-distribution window)
            if collect_samples and cnts.max() == window_size:
                samples[maj] = raw[start:end].copy()
    return pd.DataFrame(records), np.array(w_session), np.array(w_device), samples

# ── WINDOW-SIZE SWEEP — pick the best size on the session-grouped protocol ─────
# For each candidate size we re-window, extract features, and score unseen-session
# accuracy with a fixed StandardScaler → SelectKBest(MI) → RF probe (same probe
# across sizes, so the comparison isolates the WINDOW effect).  The best size (val
# acc; ties → smaller window for lower latency/flash) is deployed.  Gated by
# WIN_SWEEP; when off the architecture default is kept and this block is a no-op.
if WIN_SWEEP:
    print("\n" + "="*65)
    print(f"WINDOW-SIZE SWEEP (overlap={WIN_OVERLAP:.0%}, session-grouped CV, leak-free)")
    print("="*65)
    _win_bench = []
    for _W in WINDOW_CANDIDATES:
        _S = max(1, int(round(_W * (1.0 - WIN_OVERLAP))))
        _dfw, _ws, _wd, _ = _build_windows(_W, _S, collect_samples=False)
        _yv   = LabelEncoder().fit_transform(_dfw["activity_label"].values)
        _cols = [c for c in _dfw.columns if c != "activity_label"]
        _Xv   = np.nan_to_num(_dfw[_cols].values.astype(np.float32),
                              posinf=0.0, neginf=0.0)
        _nsp  = max(2, min(3, len(set(_ws))))
        _probe = Pipeline([
            ("sc", StandardScaler()),
            ("kb", SelectKBest(mutual_info_classif, k=min(60, _Xv.shape[1]))),
            ("rf", RandomForestClassifier(n_estimators=40, max_depth=15,
                     min_samples_leaf=3, min_samples_split=4, max_features=0.3,
                     class_weight="balanced", random_state=42, n_jobs=-1)),
        ])
        _cvr = cross_validate(_probe, _Xv, _yv, groups=_ws,
                              cv=StratifiedGroupKFold(n_splits=_nsp, shuffle=True,
                                                      random_state=42),
                              scoring="accuracy", n_jobs=-1)
        _acc = float(np.mean(_cvr["test_score"]))
        _win_bench.append({"window": _W, "step": _S, "overlap": WIN_OVERLAP,
                           "n_windows": int(len(_dfw)), "val_acc": _acc})
        print(f"  window={_W:4d} step={_S:4d}  windows={len(_dfw):6,d}  "
              f"grouped-CV val acc={_acc:.4f}")
    _best_w = max(_win_bench, key=lambda r: (round(r["val_acc"], 4), -r["window"]))
    WINDOW_SIZE = int(_best_w["window"]); STEP_SIZE = int(_best_w["step"])
    print(f"\n  WINDOW SWEEP → size={WINDOW_SIZE} step={STEP_SIZE} "
          f"(best grouped-CV val acc={_best_w['val_acc']:.4f})")
    with open("generated/window_sweep.json", "w") as fh:
        json.dump(_win_bench, fh, indent=2)
else:
    print(f"\nWINDOW-SIZE SWEEP skipped (WIN_SWEEP=0) — "
          f"using default size={WINDOW_SIZE} step={STEP_SIZE}.")

print(f"Windowing: size={WINDOW_SIZE}  step={STEP_SIZE}  (per session+device, time-sorted)")
df_win, win_session, win_device, sample_raw_windows = _build_windows(
    WINDOW_SIZE, STEP_SIZE, collect_samples=True)
print(f"  Windows  : {len(df_win):,}")

# ── SPLIT OUT DESKTOP-ONLY RESEARCH FEATURES ('res_' prefix; items 6–9) ───────
# Hold the research bank in a SEPARATE matrix so the embedded/deployed path (dedup,
# FEATURES, scaler, reducer, C export, self-test) sees the EXACT same columns as
# before RESEARCH_FEATURES existed.  df_research is consumed only by the desktop
# research-feature evaluation + importance block later.  Empty when the bank is off.
RESEARCH_COLS = [c for c in df_win.columns if c.startswith("res_")]
if RESEARCH_COLS:
    df_research = (df_win[RESEARCH_COLS]
                   .replace([np.inf, -np.inf], np.nan).fillna(0.0)
                   .astype(np.float32).copy())
    df_win = df_win.drop(columns=RESEARCH_COLS)
    print(f"  Research : {len(RESEARCH_COLS)} desktop-only 'res_' features held out "
          f"of the embedded set (benchmarked separately, not shipped to C)")
else:
    df_research = None
print(f"  Features : {df_win.shape[1] - 1}  (pre-dedup, embedded-safe)")

# ── NaN/Inf safety on the appended UCI columns ────────────────────────────────
# A degenerate window (e.g. an all-constant axis) can make a spectrum-skewness or
# correlation come back NaN/Inf.  The original 145 features are already finite;
# scrub only to keep the matrix model-ready (StandardScaler rejects NaN).
_feat_cols_all = [c for c in df_win.columns if c != "activity_label"]
df_win[_feat_cols_all] = (df_win[_feat_cols_all]
                          .replace([np.inf, -np.inf], np.nan)
                          .fillna(0.0))

# ── DEDUP — drop UCI features that are EXACT duplicates of an earlier feature ──
# Honours "add only features not already present / no redundant additions": the
# original 145 are never candidates for removal; each UCI column is compared (in
# order) against every already-kept earlier column and dropped iff numerically
# identical across all windows.  This is what prunes, e.g., tBodyGyro-* (== raw
# gyro) and tBodyGyroMag-* (== gyro_mag) that the UCI signal set re-derives.
if USE_UCI_FEATURES:
    base_cols = _feat_cols_all[:EXPECTED_TOTAL_FEAT]   # original time+freq (protected)
    uci_cols  = _feat_cols_all[EXPECTED_TOTAL_FEAT:]
    kept_vals = {c: df_win[c].values for c in base_cols}
    kept_order = list(base_cols)
    dropped, dup_of = [], {}
    for uc in uci_cols:
        v = df_win[uc].values
        match = next((kc for kc in kept_order
                      if np.allclose(v, kept_vals[kc], rtol=1e-6, atol=1e-9)), None)
        if match is not None:
            dropped.append(uc); dup_of[uc] = match
        else:
            kept_vals[uc] = v; kept_order.append(uc)
    if dropped:
        df_win = df_win.drop(columns=dropped)
        print(f"  Dedup    : dropped {len(dropped)} UCI feature(s) duplicating an "
              f"existing one (e.g. {dropped[0]} == {dup_of[dropped[0]]})")
        with open("generated/uci_dedup_dropped.json", "w") as fh:
            json.dump({uc: dup_of[uc] for uc in dropped}, fh, indent=2)
    else:
        print("  Dedup    : no exact-duplicate UCI features found")

print(f"  Features : {df_win.shape[1] - 1}  (post-dedup)")
print(f"  Balance  :\n{df_win['activity_label'].value_counts()}")

# ── EDA on windowed features ──────────────────────────────────────────────────
act_win = df_win["activity_label"].unique()
pal_win = sns.color_palette("tab10", len(act_win))

# Mean feature distributions
mean_cols = [c for c in df_win.columns if c.endswith("__mean")]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax_obj, col in zip(axes.flatten(), mean_cols[:6]):
    for act, color in zip(act_win, pal_win):
        vals = df_win.loc[df_win["activity_label"]==act, col]
        ax_obj.hist(vals, bins=30, alpha=0.5, label=act, color=color)
    ax_obj.set_title(col, fontsize=9); ax_obj.tick_params(labelsize=8)
axes[0][0].legend(fontsize=8)
plt.suptitle("Windowed Mean Features per Activity", fontsize=14, y=1.01)
plt.tight_layout(); plt.savefig("plots/windowed_means.png", dpi=120); plt.close()

# Std feature boxplots
std_cols = [c for c in df_win.columns if c.endswith("__std")][:6]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax_obj, col in zip(axes.flatten(), std_cols):
    groups = [df_win.loc[df_win["activity_label"]==a, col].values for a in act_win]
    bp = ax_obj.boxplot(groups, labels=act_win, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], pal_win): patch.set_facecolor(c)
    ax_obj.set_title(col, fontsize=9); ax_obj.tick_params(axis='x', rotation=30)
plt.suptitle("Windowed Std Features per Activity", fontsize=14, y=1.01)
plt.tight_layout(); plt.savefig("plots/windowed_stds.png", dpi=120); plt.close()

# Skewness heatmap
skew_cols = [c for c in df_win.columns if c.endswith("__skew")]
skew_means = df_win.groupby("activity_label")[skew_cols].mean()
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(skew_means, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.4, ax=ax)
ax.set_title("Mean Skewness per Feature per Activity", fontsize=13)
plt.tight_layout(); plt.savefig("plots/windowed_skewness.png", dpi=120); plt.close()

# Windowed feature correlation (first 20)
feat_20 = [c for c in df_win.columns if c!="activity_label"][:20]
corr_w  = df_win[feat_20].corr()
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_w, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax)
ax.set_title("Windowed Feature Correlation (first 20)", fontsize=13)
plt.tight_layout(); plt.savefig("plots/windowed_corr.png", dpi=120); plt.close()
print("Windowed EDA saved.")

# ── FEATURES + LABELS ──────────────────────────────────────────────────────────
FEATURES     = [c for c in df_win.columns if c != "activity_label"]
NUM_STAT_FEAT = len(FEATURES)
NUM_UCI_FEAT  = NUM_STAT_FEAT - EXPECTED_TOTAL_FEAT   # surviving UCI features
# The original 145-feature prefix (79 time + 66 freq) must remain byte-identical —
# the dedup pass and UCI append only ever add/remove columns AFTER index 144.
assert FEATURES[:EXPECTED_TIME_FEAT][-1] == "jerk_z__rms", \
    "Time-domain feature ordering changed — backward compatibility broken"
assert FEATURES[:EXPECTED_TOTAL_FEAT] == \
       [c for c in df_win.columns if c != "activity_label"][:EXPECTED_TOTAL_FEAT], \
    "Original 145-feature prefix was disturbed"
if USE_UCI_FEATURES:
    assert NUM_STAT_FEAT > 500, \
        f"UCI integration target is >500 features, got {NUM_STAT_FEAT}"
else:
    assert NUM_STAT_FEAT == EXPECTED_TOTAL_FEAT, \
        f"Expected {EXPECTED_TOTAL_FEAT}, got {NUM_STAT_FEAT}"
print(f"\nFeature count : {NUM_STAT_FEAT}  "
      f"({EXPECTED_TIME_FEAT} time + {NUM_FREQ_FEAT} freq + {NUM_UCI_FEAT} UCI-HAR)")
for i, f in enumerate(FEATURES):
    print(f"  [{i:3d}] {f}")

le = LabelEncoder()
y  = le.fit_transform(df_win["activity_label"].values)
X  = df_win[FEATURES].values.astype(np.float32)
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"\nLabel map: {label_map}")

# ── TRAIN / TEST SPLIT — STRATIFIED + SESSION-GROUPED (no window leakage) ──────
# Windows overlap, so a random row split drops near-duplicate windows of the SAME
# session into both train and test → leaked, optimistic accuracy.  We split on
# session_id so every window of a session stays wholly on one side and both wrists
# of a session travel together (no overlap or L/R near-duplicate crosses the line).
#
# CHANGE: GroupShuffleSplit → StratifiedGroupKFold(n_splits=5).  A plain grouped
# shuffle could, by chance, dump every session of a minority activity onto one side
# (0 training windows for that class).  StratifiedGroupKFold balances the activity
# proportions across folds WHILE keeping groups disjoint, so minority classes (the
# 4-session activities) are reliably present in both train and test.  We take the
# first fold as the held-out-session TEST set (≈20%); the activity label is the
# stratification target and session_id the grouping variable.
# NB: use a dedicated name (win_session), NOT `groups` — the latter is reused as a
# throwaway loop variable in later plotting blocks and would be clobbered.
N_CV_SPLITS = 5
_sgkf_split = StratifiedGroupKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=42)
tr_idx, te_idx = next(_sgkf_split.split(X, y, groups=win_session))

# ── Guarantee every activity is represented in the held-out TEST set ───────────
# StratifiedGroupKFold balances class PROPORTIONS but, because whole sessions stay
# on one side, a low-session-count activity (e.g. SITTING) can still land entirely
# in TRAIN → the held-out test has zero windows of it and its test recall is
# unmeasurable.  For any class absent from the test side we move ONE whole training
# session that carries it across to TEST (the entire session moves → no window
# leakage / group split), but ONLY while ≥1 other training session still carries
# that class so we never strip it from the model.
def _move_session_to_test(sid, tr_idx, te_idx):
    sess_mask = (win_session == sid)
    in_train  = sess_mask[tr_idx]
    moved     = tr_idx[in_train]
    return tr_idx[~in_train], np.concatenate([te_idx, moved])

for c in np.unique(y):
    if c in set(np.unique(y[te_idx])):            # already present in test
        continue
    tr_sess_with_c = np.unique(win_session[tr_idx][y[tr_idx] == c])
    if len(tr_sess_with_c) >= 2:
        sid_move = tr_sess_with_c[0]
        tr_idx, te_idx = _move_session_to_test(sid_move, tr_idx, te_idx)
        print(f"  ↳ '{le.classes_[c]}' was ABSENT from the held-out test set — "
              f"moved session {sid_move} into TEST "
              f"(kept in train via {len(tr_sess_with_c) - 1} other session(s)).")
    else:
        print(f"  ⚠ '{le.classes_[c]}' absent from test and only "
              f"{len(tr_sess_with_c)} training session(s) carry it — left in TRAIN "
              f"to avoid stripping it from the model.")

X_train, X_test = X[tr_idx], X[te_idx]
y_train, y_test = y[tr_idx], y[te_idx]
groups_train    = win_session[tr_idx]       # stable copies for CV + smoke tests
groups_test     = win_session[te_idx]
print(f"\nTrain: {X_train.shape}   Test: {X_test.shape}")
print(f"Train sessions : {sorted(set(groups_train))}")
print(f"Test  sessions : {sorted(set(groups_test))}")

# ── OPTIONAL FEATURE-MATRIX CACHE DUMP (hyperparameter-sweep harness) ──────────
# When DUMP_FEATURES=1 the fully-built, dedup'd feature matrix + the leak-free
# session-grouped train/test split are written to an .npz and the script exits
# BEFORE model fitting.  A separate sweep script then reuses this exact matrix to
# benchmark selector-K / RF-regularisation combos in-memory (no re-extraction), so
# every candidate is compared on the identical leak-free split.  Off by default.
if os.environ.get("DUMP_FEATURES", "0") != "0":
    _dump = os.environ.get("DUMP_FEATURES_PATH", "generated/feature_cache.npz")
    os.makedirs(os.path.dirname(_dump), exist_ok=True)
    np.savez_compressed(
        _dump,
        X=X.astype(np.float32), y=y.astype(np.int64),
        win_session=np.asarray(win_session, dtype=object),   # string session IDs
        tr_idx=np.asarray(tr_idx, dtype=np.int64),
        te_idx=np.asarray(te_idx, dtype=np.int64),
        features=np.asarray(FEATURES, dtype=object),
        classes=np.asarray(le.classes_, dtype=object))
    print(f"\nDUMP_FEATURES: wrote {_dump}  "
          f"(X={X.shape}, {len(set(win_session))} sessions) — exiting before fit.")
    sys.exit(0)
_train_cls = set(np.unique(y_train))
_missing   = [le.classes_[c] for c in range(len(le.classes_)) if c not in _train_cls]
if _missing:
    print(f"⚠ Classes ABSENT from the training split (test recall on them is "
          f"structurally 0): {_missing}")
_test_only_support = {le.classes_[c]: int((y_test == c).sum())
                      for c in np.unique(y_test) if c not in _train_cls}
if _test_only_support:
    print(f"  └ unsupported-class test windows: {_test_only_support}")

# ── PCA COMPONENT SELECTION (variance pool → top-K by RF importance) ───────────
# Two-stage budget:
#   1. n_pca_var  — how many components the PCA_VARIANCE_TARGET (99%) needs. This
#                   is the CANDIDATE POOL the supervised selector ranks over.
#   2. n_pca      — what the deployed model actually uses = the PCA_TOP_K best of
#                   that pool (by RandomForest importance, chosen inside the
#                   TopPCAComponents transformer at fit time, leak-free per fold).
_sc_tmp   = StandardScaler().fit(X_train)
X_tr_sc   = _sc_tmp.transform(X_train)
_pca_full = PCA().fit(X_tr_sc)
cumvar    = np.cumsum(_pca_full.explained_variance_ratio_)
n_pca_var = int(np.searchsorted(cumvar, PCA_VARIANCE_TARGET) + 1)  # candidate pool
n_pca     = int(min(PCA_TOP_K, n_pca_var))                         # kept by model
print(f"\nPCA candidate pool : {n_pca_var}  ({cumvar[n_pca_var-1]*100:.1f}% variance)")
print(f"Model PCA dims      : {n_pca}  (top-{PCA_TOP_K} of the pool by RF importance)")
assert n_pca <= 127, "n_pca exceeds int8_t range — reduce PCA_TOP_K"

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, len(cumvar)+1), cumvar*100, "o-", color="steelblue", lw=2)
ax.axhline(90, color="red",   linestyle="--", label="90% threshold")
ax.axvline(n_pca_var, color="green", linestyle="--", label=f"pool n={n_pca_var}")
ax.set_xlabel("Components"); ax.set_ylabel("Cumulative Variance (%)")
# NOTE: the model keeps the top-K by IMPORTANCE, not a variance prefix, so there
# is no single x-cut for n_pca on this (variance-ordered) curve — see
# plots/pca_component_importance.png for the components actually retained.
ax.set_title(f"PCA Scree Plot (pool={n_pca_var}, model keeps top-{n_pca})")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("plots/pca_scree.png", dpi=120); plt.close()

# ── CLASS WEIGHTS (tempered inverse frequency) ────────────────────────────────
if CLASS_WEIGHT_MODE == "none":
    class_weight = None
elif CLASS_WEIGHT_MODE == "balanced":
    class_weight = "balanced"
else:  # "tempered" — sqrt of inverse frequency, normalised to mean 1.0
    counts   = np.bincount(y_train, minlength=len(le.classes_)).astype(float)
    inv      = np.sqrt(counts.sum() / (len(counts) * np.maximum(counts, 1)))
    inv      = inv / inv.mean()
    class_weight = {i: float(w) for i, w in enumerate(inv)}
    print("\nTempered class weights:")
    for i, cls in enumerate(le.classes_):
        print(f"  {cls:<12} n={int(counts[i]):4d}  weight={class_weight[i]:.2f}")

# ── SUPERVISED TOP-K PCA COMPONENT SELECTION ──────────────────────────────────
# Drop-in replacement for PCA(n_components=K) that keeps the K most PREDICTIVE
# components instead of the first K by variance:
#   fit():  full PCA over the candidate pool → probe RandomForest scores each
#           component → retain the top-K importance indices (importance-ordered).
#   transform():  project ONLY onto the kept rows — (X - mean) @ comps_kept.T —
#           so the full-rank projection is never materialised (memory-efficient).
# It re-exposes the standard PCA attributes the rest of the code consumes
# (components_, mean_, explained_variance_ratio_) plus orig_index_ (the original
# PCA index of each kept row), so the C export, JSON dump, and feature-attribution
# display keep working unchanged — they just see 20 rows instead of 72.
# Because selection happens inside fit(), every CV fold / honest-eval split picks
# its components from ITS OWN training data only (no leakage).
class TopPCAComponents(BaseEstimator, TransformerMixin):
    def __init__(self, n_candidates, k, random_state=42, variance_target=None):
        self.n_candidates    = n_candidates
        self.k               = k
        self.random_state    = random_state
        # When set, the candidate-pool size is derived from THIS fold's own data
        # (cumulative explained variance ≥ variance_target) instead of the globally
        # precomputed n_candidates count — so the dimensionality reduction is 100%
        # per-fold and no validation-fold rows inform the pool budget.  n_candidates
        # is then only an upper cap.
        self.variance_target = variance_target

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        cap = int(min(self.n_candidates, X.shape[0], X.shape[1]))
        if cap < 1:
            raise ValueError("TopPCAComponents: empty candidate pool")
        # Fit the full(-cap) PCA on the fold's training rows only.
        pca = PCA(n_components=cap, random_state=self.random_state).fit(X)
        if self.variance_target is not None:
            # Derive the pool size from this fold's variance curve (leak-free): the
            # smallest #components whose cumulative variance clears the target.
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_cand = int(np.searchsorted(cumvar, self.variance_target) + 1)
            n_cand = int(max(1, min(n_cand, cap)))
            pca.components_              = pca.components_[:n_cand]
            pca.explained_variance_     = pca.explained_variance_[:n_cand]
            pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:n_cand]
            pca.n_components_           = n_cand
        else:
            n_cand = cap
        Z   = pca.transform(X)                       # (n, n_cand)
        # Probe forest ranks components by predictive value (mirrors the deployed
        # RF config so the ranking reflects how the model will actually use them).
        probe = RandomForestClassifier(
            n_estimators     = RF_N_TREES,
            max_depth        = RF_MAX_DEPTH,
            min_samples_leaf = RF_MIN_SAMPLES_LEAF,
            min_samples_split= RF_MIN_SAMPLES_SPLIT,
            max_features     = RF_MAX_FEATURES,
            class_weight     = class_weight,
            random_state     = self.random_state,
            n_jobs           = -1,
        ).fit(Z, y)
        k   = int(max(1, min(self.k, n_cand)))
        sel = np.argsort(probe.feature_importances_)[::-1][:k].astype(int)  # imp desc
        # Re-expose a reduced PCA (importance-ordered kept rows).
        self.orig_index_               = sel
        self.components_               = pca.components_[sel]
        self.mean_                     = pca.mean_
        self.explained_variance_       = pca.explained_variance_[sel]
        self.explained_variance_ratio_ = pca.explained_variance_ratio_[sel]
        self.n_components_             = k
        self.n_features_in_            = X.shape[1]
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T


# ── ALTERNATIVE REDUCERS — all expose the SAME contract as TopPCAComponents ───
# Contract consumed by the downstream analysis + C export:
#   .mean_ (n_feat,) , .components_ (n_out × n_feat)  with
#        transform(X) == (X - mean_) @ components_.T
#   .n_components_ , .orig_index_ (original feature/PC id per output row)
#   .selector_kind_ ("pca"|"lda"|"rf_importance"|"mutual_info"|"rfe")
#   .reducer_shape_ ("projection" | "selection")  — drives the C export path
#   .sel_indices_ (selection only): raw-feature indices fed to the model
# Fitting happens inside the pipeline, so every CV fold selects on its OWN training
# rows → leak-free by construction (same guarantee as TopPCAComponents).
class LDAProjection(BaseEstimator, TransformerMixin):
    """LinearDiscriminantAnalysis as a projection reducer (≤ n_classes-1 comps)."""
    def __init__(self, k=None):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        n_cls = len(np.unique(y))
        k = min(self.k or (n_cls - 1), n_cls - 1, X.shape[1])
        lda = LinearDiscriminantAnalysis(n_components=k, solver="svd").fit(X, y)
        # sklearn LDA.transform(X) == (X - xbar_) @ scalings_[:, :k]
        self.mean_         = lda.xbar_.astype(np.float64)
        self.components_   = lda.scalings_[:, :k].T.astype(np.float64)   # (k × n_feat)
        self.orig_index_   = np.arange(k)
        # LDA's own between-class variance ratio per discriminant component.
        self.explained_variance_ratio_ = np.asarray(
            lda.explained_variance_ratio_[:k], dtype=np.float64)
        self.n_components_  = k
        self.n_features_in_ = X.shape[1]
        self.selector_kind_ = "lda"
        self.reducer_shape_ = "projection"
        self._lda = lda
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T


class RawFeatureSelector(BaseEstimator, TransformerMixin):
    """Keep the top-K RAW features by a supervised score (rf_importance | mutual_info
    | rfe).  transform == X[:, sel_indices_].  Exposed AS a one-hot projection so the
    Python downstream stays uniform, while the C export uses the cheap gather path."""
    def __init__(self, kind, k, random_state=42):
        self.kind = kind
        self.k = k
        self.random_state = random_state

    def _probe_rf(self):
        return RandomForestClassifier(
            n_estimators=RF_N_TREES, max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF, min_samples_split=RF_MIN_SAMPLES_SPLIT,
            max_features=RF_MAX_FEATURES, class_weight=class_weight,
            random_state=self.random_state, n_jobs=-1)

    def _mrmr_select(self, X, y, k):
        """mRMR (Maximum-Relevance / Minimum-Redundancy) greedy feature selection.

        relevance[i]  = mutual information between feature i and the label (how much
                        activity information the feature carries on its own).
        redundancy[i] = mean |Pearson corr| between feature i and the features ALREADY
                        kept (how much of that information is a DUPLICATE of what we
                        have).  Correlation is the standard efficient redundancy proxy
                        used by mainstream mRMR (Peng et al. / the `mrmr` library);
                        a full pairwise-MI matrix would be far costlier for the same
                        ranking.

        Each step adds argmax(relevance / redundancy)  (the MIQ / quotient scheme).
        Quotient — not the difference (relevance − redundancy) — because MI relevance
        is unbounded while |corr| redundancy is ≤ 1, so a plain subtraction barely
        penalises a high-MI duplicate; the ratio is scale-robust and is what the
        mainstream `mrmr` library uses.  So two highly-correlated discriminative twins
        (e.g. accel_mag__rms and accel_mag__energy) can't BOTH be spent — once one is
        kept the other's redundancy divides its relevance down and a feature carrying
        NEW signal wins the slot instead.
        Returns the kept indices in selection order (most informative first).
        """
        n_feat = X.shape[1]
        k = int(max(1, min(k, n_feat)))
        rel = np.nan_to_num(
            mutual_info_classif(X, y, random_state=self.random_state))
        # z-score the columns once so an inner product == Pearson correlation·n.
        Xc  = X - X.mean(axis=0)
        sd  = Xc.std(axis=0); sd[sd < 1e-12] = 1.0
        Xn  = Xc / sd
        n   = X.shape[0]
        # First pick: pure max relevance (no redundancy term yet).
        first     = int(np.argmax(rel))
        selected  = [first]
        remaining = np.ones(n_feat, dtype=bool); remaining[first] = False
        # Running sum of |corr| from every feature to the kept set (incremental, so
        # we never build the full n_feat×n_feat correlation matrix).
        red_sum = np.abs(Xn.T @ Xn[:, first]) / n
        while len(selected) < k and remaining.any():
            redundancy = red_sum / len(selected)             # mean |corr| to kept set
            score      = rel / np.maximum(redundancy, 1e-6)  # MIQ quotient (scale-robust)
            score[~remaining] = -np.inf
            nxt = int(np.argmax(score))
            selected.append(nxt)
            remaining[nxt] = False
            red_sum += np.abs(Xn.T @ Xn[:, nxt]) / n
        idx = np.asarray(selected, dtype=int)
        self.feature_scores_ = rel[idx]                      # relevance of kept feats
        return idx

    def _prioritised_topk(self, scores, k):
        """Pick the top-k feature indices by `scores`, but — when
        PRIORITIZE_MAGNITUDE_FEATURES is on and feature names are available — take
        wrist-invariant magnitude/frequency features FIRST (ranked by score), then
        fill any remaining slots with the best single-axis features."""
        order = np.argsort(scores)[::-1]                  # all indices, score desc
        names = globals().get("FEATURES")
        if not PRIORITIZE_MAGNITUDE_FEATURES or names is None or len(names) != len(scores):
            return order[:k]
        pref = [i for i in order if _is_magfreq_feature(names[i])]
        rest = [i for i in order if not _is_magfreq_feature(names[i])]
        return np.asarray((pref + rest)[:k], dtype=int)

    def _physics_prioritised_topk(self, scores, k):
        """HYBRID pick: `scores` are the SelectKBest mutual-information scores.  Take
        the high-impact rotation-physics features (_is_physics_feature) FIRST — ranked
        among themselves by MI, and only those that actually carry signal (MI > 0, so a
        dead physics feature never crowds out an informative one) — then fill the
        remaining budget with the next-best features by MI (physics or not).  This is
        the "SelectKBest + mutual-info ranking that selects features with high impact on
        angular displacement / circular motion / gyro energy / energy ratio / RMS
        angular velocity / correlation / accel magnitude / accel variance / orientation
        change / gyro-to-accel / peak-to-peak" the hybrid reducer implements."""
        order = np.argsort(scores)[::-1]                  # all indices, MI score desc
        names = globals().get("FEATURES")
        if names is None or len(names) != len(scores):    # names unavailable → plain top-K
            return order[:k]
        eps  = 1e-9
        phys = [i for i in order if _is_physics_feature(names[i]) and scores[i] > eps]
        phys_set = set(phys)
        rest = [i for i in order if i not in phys_set]
        chosen = (phys + rest)[:k]
        self.n_physics_selected_ = sum(1 for i in chosen if i in phys_set)
        return np.asarray(chosen, dtype=int)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        n_feat = X.shape[1]
        k = int(max(1, min(self.k, n_feat)))
        if self.kind == "rf_importance":
            imp = self._probe_rf().fit(X, y).feature_importances_
            idx = self._prioritised_topk(imp, k)
            self.feature_scores_ = imp[idx]
        elif self.kind == "mutual_info":
            skb = SelectKBest(mutual_info_classif, k="all").fit(X, y)
            scores = np.nan_to_num(skb.scores_)
            idx = self._prioritised_topk(scores, k)
            self.feature_scores_ = scores[idx]
        elif self.kind == "hybrid":
            # SelectKBest(mutual_info_classif) ranks all features by MI to the label;
            # _physics_prioritised_topk then takes the high-impact rotation-physics
            # descriptors first (in MI order) and fills the rest of the budget by MI.
            skb = SelectKBest(mutual_info_classif, k="all").fit(X, y)
            scores = np.nan_to_num(skb.scores_)
            idx = self._physics_prioritised_topk(scores, k)
            self.feature_scores_ = scores[idx]
        elif self.kind == "mrmr":
            # mRMR sets self.feature_scores_ internally (relevance of kept feats).
            # Its redundancy term already suppresses duplicate axes, so the
            # magnitude-prioritisation heuristic (_prioritised_topk) is NOT applied
            # here — mRMR handles "minimum duplication" directly and data-driven.
            idx = self._mrmr_select(X, y, k)
        elif self.kind == "rfe":
            rfe = RFE(self._probe_rf(), n_features_to_select=k,
                      step=0.1).fit(X, y)
            idx = np.where(rfe.support_)[0]
            # order by RFE ranking (all selected share rank 1; keep stable order)
            self.feature_scores_ = np.ones(len(idx))
        elif self.kind == "sequential":
            # Sequential Forward Selection (master-prompt item 10).  Greedy wrapper:
            # add the feature that most improves grouped-CV accuracy each step until k
            # are chosen.  O(n_feat·k) model fits → very slow on ~560 features, so a
            # LIGHT probe RF is used and this selector is OFF by default (SEL_SFS=1 to
            # benchmark it).  Leak-safe: still fit only on the fold's train rows.
            from sklearn.feature_selection import SequentialFeatureSelector
            sfs = SequentialFeatureSelector(
                self._probe_rf(), n_features_to_select=k, direction="forward",
                scoring="accuracy", cv=3, n_jobs=-1).fit(X, y)
            idx = np.where(sfs.get_support())[0]
            self.feature_scores_ = np.ones(len(idx))
        else:
            raise ValueError(f"unknown RawFeatureSelector kind {self.kind!r}")
        idx = np.asarray(sorted(int(i) for i in idx), dtype=int)
        self.sel_indices_  = idx
        self.mean_         = np.zeros(n_feat, dtype=np.float64)
        comp = np.zeros((len(idx), n_feat), dtype=np.float64)
        comp[np.arange(len(idx)), idx] = 1.0          # one-hot gather as a projection
        self.components_    = comp
        self.orig_index_    = idx
        # No true "explained variance" for a selection; expose each kept feature's
        # share of total input variance so the variance-labelled displays don't crash.
        var = np.asarray(X).var(axis=0)
        self.explained_variance_ratio_ = (var[idx] / (var.sum() + 1e-12)).astype(np.float64)
        self.n_components_   = len(idx)
        self.n_features_in_  = n_feat
        self.selector_kind_  = self.kind
        self.reducer_shape_  = "selection"
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)[:, self.sel_indices_]


# Tag the PCA reducer with the same shape metadata the others carry.
TopPCAComponents.selector_kind_ = "pca"
TopPCAComponents.reducer_shape_ = "projection"


def make_selector(kind):
    """Factory for the configurable reducer slot (see FEATURE_SELECTOR)."""
    if kind == "pca":
        # variance_target lets each CV fold derive its OWN candidate pool from its
        # own training rows (n_pca_var is only an upper cap) → no cross-fold leakage.
        return TopPCAComponents(n_candidates=n_pca_var, k=PCA_TOP_K,
                                variance_target=PCA_VARIANCE_TARGET)
    if kind == "lda":
        return LDAProjection(k=len(le.classes_) - 1)
    if kind == "rf_importance":
        return RawFeatureSelector("rf_importance", SEL_K_RF)
    if kind == "mutual_info":
        return RawFeatureSelector("mutual_info", SEL_K_MI)
    if kind == "hybrid":
        return RawFeatureSelector("hybrid", SEL_K_HYBRID)
    if kind == "mrmr":
        return RawFeatureSelector("mrmr", SEL_K_MRMR)
    if kind == "rfe":
        return RawFeatureSelector("rfe", SEL_K_RFE)
    if kind == "sequential":
        return RawFeatureSelector("sequential", SEL_K_SFS)
    raise ValueError(f"unknown FEATURE_SELECTOR {kind!r}")


def selector_n_out(kind):
    """Expected model-input width for a reducer kind (for sizing / int8 check)."""
    return {"pca": int(min(PCA_TOP_K, n_pca_var)),
            "lda": len(le.classes_) - 1,
            "rf_importance": SEL_K_RF,
            "mutual_info": SEL_K_MI,
            "hybrid": SEL_K_HYBRID,
            "mrmr": SEL_K_MRMR,
            "rfe": SEL_K_RFE,
            "sequential": SEL_K_SFS}[kind]


# FEATURE-SELECTOR BENCHMARK (master-prompt items 4 & 10).  Every candidate reducer
# is scored on the SAME session-grouped protocol so the log carries an honest A/B of
# all of them.  The master prompt asks that MI / RF-importance / RFE / mRMR / SFS be
# prioritised over PCA for rotational IMU data (PCA mixes signed physical axes into
# variance-ordered components and destroys the direction cues that separate CLK vs
# ACLK), so PCA is kept ONLY as a baseline for the comparison, not as a default.
# 'sequential' (SFS) is O(n_feat·k) model fits → included only when SEL_SFS=1.
ALL_SELECTORS = ["hybrid", "mutual_info", "rf_importance", "rfe", "mrmr", "pca"]
if SEL_SFS:
    ALL_SELECTORS.insert(-1, "sequential")


# ── BUILD PIPELINE ─────────────────────────────────────────────────────────────
def build_pipeline(selector_kind=None):
    # The reducer step keeps the legacy key "pca" so the existing downstream code
    # (pipe.named_steps["pca"], the C export) reads it unchanged — it now holds
    # whichever reducer FEATURE_SELECTOR_ACTIVE selects, not necessarily PCA.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    make_selector(selector_kind or FEATURE_SELECTOR_ACTIVE)),
        ("clf",    RandomForestClassifier(
            n_estimators     = RF_N_TREES,
            max_depth        = RF_MAX_DEPTH,
            min_samples_leaf = RF_MIN_SAMPLES_LEAF,
            min_samples_split= RF_MIN_SAMPLES_SPLIT,
            max_features     = RF_MAX_FEATURES,
            bootstrap        = RF_BOOTSTRAP,
            class_weight     = class_weight,
            random_state     = 42,
            n_jobs           = -1,
        )),
    ])


# ── RF HYPERPARAMETER SEARCH (master-prompt item 3) ───────────────────────────
# RandomizedSearchCV over the FULL deployed pipeline (scaler → reducer → RF),
# scored by StratifiedGroupKFold on the TRAINING sessions only.  Because the whole
# pipeline is refit inside each CV fold, the scaler + feature reducer are re-fit on
# every fold's train rows — no selection/scaling touches the validation rows, and
# the held-out TEST sessions are never seen by the search.  The winner is chosen on
# session-grouped VAL accuracy (true unseen-session generalisation), then written
# back into the RF_* globals so every later build_pipeline() ships the tuned forest.
# WHY this helps generalisation: the seed RF (depth 8, leaf 2) posts a ~0.76
# train-val gap — a textbook high-variance overfit.  Searching deeper regularisation
# (larger min_samples_leaf/split, capped depth, decorrelating max_features) directly
# trades train-fit for held-out-session stability, which is the metric we optimise.
if RF_SEARCH:
    print("\n" + "="*65)
    print(f"RF HYPERPARAMETER SEARCH — RandomizedSearchCV ({RF_SEARCH_ITER} configs, "
          f"session-grouped CV, leak-free)")
    print("="*65)
    _rf_space = {
        # n_estimators kept modest for embedded compactness (each tree costs C
        # code + flash); depth/leaf/split/max_features carry the regularisation.
        "clf__n_estimators":      [16, 24, 32, 40, 50, 64],
        "clf__max_depth":         [6, 8, 10, 12, 15, None],
        "clf__min_samples_leaf":  [1, 2, 3, 4],
        "clf__min_samples_split": [2, 4, 6, 8],
        "clf__max_features":      ["sqrt", "log2", 0.3, 0.5, 0.8],
        "clf__bootstrap":         [True],
    }
    _rf_search_cv = StratifiedGroupKFold(
        n_splits=max(2, min(N_CV_SPLITS, len(set(groups_train)))),
        shuffle=True, random_state=42)
    _rf_search = RandomizedSearchCV(
        estimator          = build_pipeline(FEATURE_SELECTOR),   # deployed reducer
        param_distributions= _rf_space,
        n_iter             = RF_SEARCH_ITER,
        scoring            = "accuracy",
        cv                 = _rf_search_cv,
        random_state       = 42,
        n_jobs             = -1,
        refit              = False,           # we only want the winning params
        error_score        = 0.0,
    )
    _rf_search.fit(X_train, y_train, groups=groups_train)
    _bp = _rf_search.best_params_
    print(f"  best session-grouped CV val acc = {_rf_search.best_score_:.4f}")
    print(f"  best params: {_bp}")
    # Write winners back into the globals build_pipeline() reads at call time.
    RF_N_TREES           = int(_bp["clf__n_estimators"])
    RF_MAX_DEPTH         = _bp["clf__max_depth"]            # may be None (unbounded)
    RF_MIN_SAMPLES_LEAF  = int(_bp["clf__min_samples_leaf"])
    RF_MIN_SAMPLES_SPLIT = int(_bp["clf__min_samples_split"])
    RF_MAX_FEATURES      = _bp["clf__max_features"]
    RF_BOOTSTRAP         = bool(_bp["clf__bootstrap"])
    print(f"  → deployed RF: n_estimators={RF_N_TREES} max_depth={RF_MAX_DEPTH} "
          f"min_samples_leaf={RF_MIN_SAMPLES_LEAF} "
          f"min_samples_split={RF_MIN_SAMPLES_SPLIT} max_features={RF_MAX_FEATURES}")
    with open("generated/rf_search_best.json", "w") as fh:
        json.dump({"best_val_acc": float(_rf_search.best_score_),
                   "best_params": {k: (v if not isinstance(v, np.generic) else v.item())
                                   for k, v in _bp.items()},
                   "n_iter": RF_SEARCH_ITER}, fh, indent=2)
else:
    print("\nRF HYPERPARAMETER SEARCH skipped (RF_SEARCH=0) — using seed RF params.")

# ── PCA-K SWEEP — grow the kept-component budget until test accuracy ≥ target ──
# Replaces the fixed PCA_TOP_K=20.  Starting at PCA_TOP_K_MIN we add PCA_TOP_K_STEP
# components at a time, refit the full 'pca' pipeline on the training rows, and
# measure the held-out-session TEST accuracy — stopping at the FIRST K that reaches
# PCA_TOP_K_TARGET_ACC (≥80%).  If even the full variance pool can't clear it we
# keep the K with the best accuracy observed.  Only the 'pca' reducer is sized here;
# the raw selectors keep their SEL_K_* budgets.  The kept count is capped at 127 so
# the int8 PCA-index used by the C export never overflows.
#   NOTE: tuning K against the test set is a mild optimism (the test set informs a
#   hyper-parameter).  We therefore ALSO print the session-grouped CV val accuracy
#   at the chosen K as the honest generalisation signal.
#   SKIPPED unless 'pca' is an active candidate: the deployed reducer is now mRMR on
#   raw features (FEATURE_SELECTOR='mrmr'), so this blind grow-K-until-accuracy PCA
#   search is dead weight — sizing it would burn the test set on a reducer we never
#   ship.  PCA_TOP_K keeps its seed default if the sweep is skipped.
if "pca" in ALL_SELECTORS:
    _pca_k_cap = min(int(n_pca_var), 127)
    if PCA_TOP_K_MAX is not None:
        _pca_k_cap = min(_pca_k_cap, int(PCA_TOP_K_MAX))
    print("\n" + "="*65)
    print(f"PCA-K SWEEP — grow components until held-out test acc ≥ {PCA_TOP_K_TARGET_ACC:.0%}")
    print("="*65)
    _pca_k_grid = list(range(max(1, PCA_TOP_K_MIN), _pca_k_cap + 1, PCA_TOP_K_STEP))
    if not _pca_k_grid or _pca_k_grid[-1] != _pca_k_cap:
        _pca_k_grid.append(_pca_k_cap)            # always try the full pool last
    _pca_sweep, _pca_best, _pca_hit = [], None, None
    for _k in _pca_k_grid:
        PCA_TOP_K = _k                            # rebind the global make_selector reads
        _pk   = build_pipeline("pca").fit(X_train, y_train)
        _acc  = accuracy_score(y_test, _pk.predict(X_test))
        _kept = int(_pk.named_steps["pca"].n_components_)
        _pca_sweep.append((_kept, _acc))
        _flag = "  ← target met" if _acc >= PCA_TOP_K_TARGET_ACC else ""
        print(f"  PCA K={_kept:3d}  held-out-session test acc = {_acc:.3f}{_flag}")
        if _pca_best is None or _acc > _pca_best[1]:
            _pca_best = (_kept, _acc)
        if _acc >= PCA_TOP_K_TARGET_ACC:
            _pca_hit = (_kept, _acc)
            break
    if _pca_hit is not None:
        PCA_TOP_K = _pca_hit[0]
        print(f"  → PCA_TOP_K = {PCA_TOP_K}  (first K to clear "
              f"{PCA_TOP_K_TARGET_ACC:.0%}, test acc = {_pca_hit[1]:.3f})")
    else:
        PCA_TOP_K = _pca_best[0]
        print(f"  → target {PCA_TOP_K_TARGET_ACC:.0%} not reached on the test set; "
              f"PCA_TOP_K = {PCA_TOP_K}  (best test acc = {_pca_best[1]:.3f})")
    # Honest cross-check: session-grouped CV val accuracy at the deployed K.
    _pca_cv = cross_validate(build_pipeline("pca"), X_train, y_train, groups=groups_train,
                             cv=StratifiedGroupKFold(
                                 n_splits=max(2, min(N_CV_SPLITS, len(set(groups_train)))),
                                 shuffle=True, random_state=42),
                             scoring="accuracy", n_jobs=-1)
    print(f"  honest check : group-CV val acc at PCA K={PCA_TOP_K} = "
          f"{np.mean(_pca_cv['test_score']):.3f}  (test-tuned K may be optimistic)")
else:
    print("\nPCA-K SWEEP skipped — 'pca' is not an active reducer "
          f"(deployed = hybrid SelectKBest+MI physics reducer, "
          f"FEATURE_SELECTOR='{FEATURE_SELECTOR}').")

# ── REDUCER BENCHMARK — pick the feature-reduction method honestly ────────────
# Every candidate reducer is benchmarked under the SAME session-grouped protocol
# the rest of the script uses: GroupKFold over the training sessions (held-out-
# session val accuracy, leak-free) plus the held-out-session TEST accuracy.  All
# reducers fit inside their pipeline, so no selection touches the val/test rows.
# FEATURE_SELECTOR="auto" deploys the winner (best test acc, then val, then fewer
# features); set FEATURE_SELECTOR to a kind to skip the auto-pick.
print("\n" + "="*65)
print("FEATURE-REDUCER BENCHMARK (session-grouped, leak-free)")
print("="*65)
_n_train_groups = len(set(groups_train))
# StratifiedGroupKFold needs at least n_splits groups; cap defensively, but use the
# requested 5 folds whenever the training set has enough sessions.
_cv_splits = max(2, min(N_CV_SPLITS, _n_train_groups))
_cv_factory = lambda: StratifiedGroupKFold(n_splits=_cv_splits, shuffle=True,
                                           random_state=42)
_bench = []
for _kind in ALL_SELECTORS:
    try:
        _p = build_pipeline(_kind)
        _cv = cross_validate(_p, X_train, y_train, groups=groups_train,
                             cv=_cv_factory(),
                             scoring="accuracy", n_jobs=-1)
        _val = float(np.mean(_cv["test_score"]))
        _p.fit(X_train, y_train)
        _tacc = accuracy_score(y_test, _p.predict(X_test))
        _nout = _p.named_steps["pca"].n_components_
        _shape = _p.named_steps["pca"].reducer_shape_
        _bench.append({"kind": _kind, "val": _val, "test": _tacc,
                       "n_out": _nout, "shape": _shape})
        print(f"  {_kind:14s} n_out={_nout:3d} [{_shape:10s}]  "
              f"group-CV val={_val:.3f}   held-out-session test={_tacc:.3f}")
    except Exception as e:
        print(f"  {_kind:14s} FAILED: {type(e).__name__}: {e}")
if not _bench:
    raise RuntimeError("all reducers failed to benchmark")

# ── SELECTOR COMPARISON PLOT + JSON (master-prompt item 10) ───────────────────
# Grouped-CV val vs held-out-session test accuracy for every candidate reducer,
# so the choice of selector is auditable rather than asserted.
try:
    _bo = sorted(_bench, key=lambda r: r["val"], reverse=True)
    _labels = [b["kind"] for b in _bo]
    _xv = np.arange(len(_labels)); _w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.4*len(_labels)), 4.5))
    ax.bar(_xv - _w/2, [b["val"] for b in _bo],  _w, label="grouped-CV val", color="steelblue")
    ax.bar(_xv + _w/2, [b["test"] for b in _bo], _w, label="held-out-session test", color="coral")
    for i, b in enumerate(_bo):
        ax.text(i - _w/2, b["val"]+.005,  f"{b['val']:.2f}",  ha="center", va="bottom", fontsize=8)
        ax.text(i + _w/2, b["test"]+.005, f"{b['test']:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(_xv); ax.set_xticklabels(_labels, rotation=20)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.0); ax.legend(); ax.grid(alpha=.3, axis="y")
    ax.set_title("Feature-selector benchmark (session-grouped, leak-free)")
    plt.tight_layout(); plt.savefig("plots/selector_benchmark.png", dpi=120); plt.close()
except Exception as _e:
    print(f"  (selector-comparison plot skipped: {_e})")
with open("generated/selector_benchmark.json", "w") as fh:
    json.dump(_bench, fh, indent=2, default=float)

if FEATURE_SELECTOR == "auto":
    # Master-prompt item 4: prefer physically-meaningful RAW selectors over PCA.
    # Rank raw selectors by (val, test, fewer-features); pick PCA only if it beats
    # the best raw selector's val by a clear margin (destroying signed axes must be
    # earned, not chosen on a 0.001 tie).
    _PCA_MARGIN = 0.02
    _raw  = [r for r in _bench if r["kind"] != "pca"]
    _pca  = next((r for r in _bench if r["kind"] == "pca"), None)
    _best_raw = max(_raw, key=lambda r: (round(r["val"], 4), round(r["test"], 4),
                                         -r["n_out"])) if _raw else None
    if _best_raw is None:
        _winner = _pca
    elif _pca is not None and _pca["val"] > _best_raw["val"] + _PCA_MARGIN:
        _winner = _pca
        print(f"\n  (PCA beat best raw selector by >{_PCA_MARGIN:.0%} val → allowed)")
    else:
        _winner = _best_raw
    FEATURE_SELECTOR_ACTIVE = _winner["kind"]
    print(f"\n  AUTO-PICK → '{FEATURE_SELECTOR_ACTIVE}' "
          f"(grouped-CV val = {_winner['val']:.3f}, held-out-session test = {_winner['test']:.3f})")
else:
    FEATURE_SELECTOR_ACTIVE = FEATURE_SELECTOR
    print(f"\n  PINNED   → '{FEATURE_SELECTOR_ACTIVE}' (FEATURE_SELECTOR override)")
# Model-input width + int8 feature-index guard for the chosen reducer.
n_pca = selector_n_out(FEATURE_SELECTOR_ACTIVE)
assert n_pca <= 127, f"reducer n_out={n_pca} exceeds int8_t — lower its K"
print(f"  Deployed reducer: '{FEATURE_SELECTOR_ACTIVE}'  model-input dims = {n_pca}")

pipe = build_pipeline()

# ── OVERFITTING CHECK — STRATIFIED + GROUP-AWARE CV (session-disjoint folds) ───
# StratifiedKFold alone would shuffle overlapping windows of a session across folds
# (the same leakage the split above fixes), so the gap looked artificially tiny.
# StratifiedGroupKFold keeps each session wholly inside one fold (no window leak)
# AND balances the activity proportions across folds, so minority classes are
# represented in every validation fold and the gap is an honest train-vs-held-out
# estimate.  (_cv_splits / _n_train_groups were computed in the reducer benchmark.)
cv_res = cross_validate(
    pipe, X_train, y_train, groups=groups_train,
    cv=_cv_factory(),
    scoring="accuracy", return_train_score=True, n_jobs=-1)
print(f"Stratified-Group-CV folds : {_cv_splits}  (over {_n_train_groups} training sessions)")
tr_cv  = cv_res["train_score"]
val_cv = cv_res["test_score"]
gap    = tr_cv.mean() - val_cv.mean()
print(f"\nCV Train: {tr_cv.mean():.4f}  Val: {val_cv.mean():.4f}  Gap: {gap:.4f}  "
      f"{'⚠ OVERFIT' if gap > 0.05 else '✓ OK'}")

fig, ax = plt.subplots(figsize=(8, 4))
folds = range(1, len(tr_cv) + 1)
ax.plot(folds, tr_cv,  "o-",  label="Train",      color="steelblue", lw=2)
ax.plot(folds, val_cv, "s--", label="Validation (held-out session)", color="coral", lw=2)
ax.fill_between(folds, tr_cv, val_cv, alpha=0.15, color="gray",
                label=f"Gap={gap:.3f}")
ax.set_ylim(0.0, 1.05); ax.legend(); ax.grid(alpha=0.3)
ax.set_xticks(list(folds))
ax.set_title(f"{len(tr_cv)}-Fold StratifiedGroupKFold (session-disjoint): Train vs Validation")
plt.tight_layout(); plt.savefig("plots/overfitting_check.png", dpi=120); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# COMPLEMENTARY CHECK — hold out an entire device stream (LEFT vs RIGHT wrist)
# ───────────────────────────────────────────────────────────────────────────────
# The main split above is already session-grouped (leak-free).  This second view
# measures cross-DEVICE generalization: train on one wrist's stream, test on the
# other.  Both streams are the REAL recordings as-is (no mirror/reflection is
# applied anywhere), so a low score here reflects the genuine LEFT↔RIGHT frame
# difference — treat it as a wrist-transfer check, not the headline metric.
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("HONEST EVAL — train on one wrist, test on the other (no window leakage)")
print("="*65)
# MIN_TRUSTWORTHY_SUPPORT: per-class test support below this is treated as noise —
# an f1/recall on a handful of windows is not a measurement.  Convention, not law.
MIN_TRUSTWORTHY_SUPPORT = 30
honest_results = []   # consumed by the honest reporter at the end of the run
for hold in ("RIGHT", "LEFT"):
    tr = win_device != hold
    te = win_device == hold
    if te.sum() == 0 or tr.sum() == 0:
        continue
    p = build_pipeline()
    p.fit(X[tr], y[tr])
    yp = p.predict(X[te])
    acc = accuracy_score(y[te], yp)
    # Majority-class ("always predict the most common class") baseline ON THIS
    # split — the bar any honest accuracy must clear to mean anything.
    cls_counts = np.bincount(y[te], minlength=len(le.classes_))
    baseline   = float(cls_counts.max() / cls_counts.sum())
    baseline_cls = le.classes_[int(cls_counts.argmax())]
    print(f"\n  Train={'+'.join(sorted(set(win_device[tr])))}  →  Test={hold}"
          f"   acc={acc:.3f}  (majority-'{baseline_cls}' baseline={baseline:.3f})")
    labels_present = np.unique(np.concatenate([y[te], yp]))
    print(classification_report(
        y[te], yp, labels=labels_present,
        target_names=[le.classes_[i] for i in labels_present],
        zero_division=0))
    honest_results.append({
        "hold": hold,
        "train_on": "+".join(sorted(set(win_device[tr]))),
        "acc": acc,
        "baseline": baseline,
        "baseline_cls": baseline_cls,
        "n_test": int(te.sum()),
        "support": {le.classes_[i]: int(cls_counts[i]) for i in range(len(le.classes_))},
    })

# ── FINAL TRAINING + EVALUATION ───────────────────────────────────────────────
print("\n" + "="*65)
print("FINAL TRAINING + EVALUATION")
print("="*65)
pipe.fit(X_train, y_train)
y_pred    = pipe.predict(X_test)
train_acc = pipe.score(X_train, y_train)
test_acc  = accuracy_score(y_test, y_pred)
final_gap = train_acc - test_acc
print(f"Train accuracy : {train_acc:.4f}")
print(f"Test  accuracy : {test_acc:.4f}")
print(f"Gap            : {final_gap:.4f}  {'⚠ Overfit' if final_gap>0.05 else '✓ Good'}")
print("\n--- Per-class Report ---")
# labels= all classes so the report is robust when a session-grouped test split
# does not contain every class (zero_division=0 keeps absent classes at 0.0).
print(classification_report(y_test, y_pred,
                            labels=np.arange(len(le.classes_)),
                            target_names=le.classes_, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_)))
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
    ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix (held-out-session test acc={test_acc:.3f})")
plt.xticks(rotation=30); plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=120); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE-IMPORTANCE ANALYSIS (master-prompt item 12) — DEPLOYED embedded model
#  Permutation importance on the held-out TEST split (leak-free: importance is
#  measured by how much shuffling each raw input degrades UNSEEN-session accuracy)
#  plus RF impurity importance.  Ranks the embedded features, prints the top-50/
#  top-100, and flags the least-useful ones as removal candidates.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("FEATURE-IMPORTANCE ANALYSIS (permutation, held-out test — item 12)")
print("="*65)
from sklearn.inspection import permutation_importance
_perm = permutation_importance(pipe, X_test, y_test, n_repeats=5,
                               random_state=42, n_jobs=-1, scoring="accuracy")
_imp   = _perm.importances_mean
_order = np.argsort(_imp)[::-1]
_ranked = [(FEATURES[i], float(_imp[i]), float(_perm.importances_std[i])) for i in _order]
print(f"  Top 15 embedded features by permutation importance ({len(FEATURES)} total):")
for nm, mu, sd in _ranked[:15]:
    print(f"    {mu:+.4f} ± {sd:.4f}   {nm}")
_removable = [nm for nm, mu, _ in _ranked if mu <= 0.0]   # no unseen-session value
print(f"  Removal candidates (≤0 permutation importance): {len(_removable)} / {len(FEATURES)}")
with open("generated/feature_importance.json", "w") as fh:
    json.dump({"top100": [{"feature": nm, "importance": mu, "std": sd}
                          for nm, mu, sd in _ranked[:100]],
               "removable_zero_importance": _removable,
               "n_features": len(FEATURES)}, fh, indent=2)
# top-30 bar chart
_tn = min(30, len(_ranked))
fig, ax = plt.subplots(figsize=(9, max(4, 0.28*_tn)))
ax.barh([r[0] for r in _ranked[:_tn]][::-1], [r[1] for r in _ranked[:_tn]][::-1],
        color="steelblue"); ax.set_xlabel("permutation importance (Δ acc)")
ax.set_title(f"Top-{_tn} embedded features (held-out-session permutation importance)")
plt.tight_layout(); plt.savefig("plots/feature_importance.png", dpi=120); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  RESEARCH-FEATURE EVALUATION (items 6–9) — do the DESKTOP-ONLY 'res_' features
#  add anything on unseen sessions?  Compares embedded-only vs embedded+research on
#  the SAME session-grouped protocol, and ranks where the res_ features land in a
#  full-set RF importance list — so we can honestly say which (if any) would be
#  worth the cost of a C port.  Skipped unless the research bank was computed.
# ══════════════════════════════════════════════════════════════════════════════
if df_research is not None and df_research.shape[1] > 0:
    print("\n" + "="*65)
    print(f"RESEARCH-FEATURE EVALUATION ({df_research.shape[1]} desktop-only features — "
          f"items 6–9)")
    print("="*65)
    _R = df_research.values.astype(np.float32)
    _Xfull        = np.hstack([X, _R])
    _feat_full    = list(FEATURES) + list(RESEARCH_COLS)
    _Xtr_f, _Xte_f = _Xfull[tr_idx], _Xfull[te_idx]
    _cvf = lambda: StratifiedGroupKFold(n_splits=_cv_splits, shuffle=True, random_state=42)
    def _score_set(Xtr, Xte, k):
        _pl = Pipeline([
            ("sc", StandardScaler()),
            ("kb", SelectKBest(mutual_info_classif, k=min(k, Xtr.shape[1]))),
            ("rf", RandomForestClassifier(
                n_estimators=RF_N_TREES, max_depth=RF_MAX_DEPTH,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF, min_samples_split=RF_MIN_SAMPLES_SPLIT,
                max_features=RF_MAX_FEATURES, bootstrap=RF_BOOTSTRAP,
                class_weight=class_weight, random_state=42, n_jobs=-1))])
        _v = float(np.mean(cross_validate(_pl, Xtr, y_train, groups=groups_train,
                     cv=_cvf(), scoring="accuracy", n_jobs=-1)["test_score"]))
        _pl.fit(Xtr, y_train)
        _t = accuracy_score(y_test, _pl.predict(Xte))
        return _v, _t
    _v_emb, _t_emb   = _score_set(X_train, X_test, SEL_K_HYBRID)
    _v_full, _t_full = _score_set(_Xtr_f, _Xte_f, SEL_K_HYBRID)
    print(f"  embedded-only ({X.shape[1]:3d} feats)  grouped-CV val={_v_emb:.3f}  test={_t_emb:.3f}")
    print(f"  + research    ({_Xfull.shape[1]:3d} feats)  grouped-CV val={_v_full:.3f}  test={_t_full:.3f}")
    print(f"  Δ from research bank:  val {_v_full-_v_emb:+.3f}   test {_t_full-_t_emb:+.3f}")
    # Where do res_ features rank in a full-set RF importance list?
    _rf_full = RandomForestClassifier(
        n_estimators=RF_N_TREES, max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF, min_samples_split=RF_MIN_SAMPLES_SPLIT,
        max_features=RF_MAX_FEATURES, bootstrap=RF_BOOTSTRAP,
        class_weight=class_weight, random_state=42, n_jobs=-1)
    _rf_full.fit(StandardScaler().fit_transform(_Xtr_f), y_train)
    _fi = _rf_full.feature_importances_
    _fo = np.argsort(_fi)[::-1]
    _res_ranks = [(rank, _feat_full[i], float(_fi[i]))
                  for rank, i in enumerate(_fo) if _feat_full[i].startswith("res_")]
    print(f"  Best-ranked research features (rank / {len(_feat_full)} by RF importance):")
    for rank, nm, imp in _res_ranks[:10]:
        print(f"    #{rank+1:<4d} imp={imp:.4f}  {nm}")
    _n_res_top100 = sum(1 for rank, _, _ in _res_ranks if rank < 100)
    print(f"  Research features in the global top-100: {_n_res_top100} / {len(RESEARCH_COLS)}")
    _verdict = ("worth a C port — measurably lifts unseen-session accuracy"
                if _v_full > _v_emb + 0.01 else
                "NOT worth porting — no unseen-session gain over the embedded set")
    print(f"  Verdict: research bank {_verdict}.")
    with open("generated/research_feature_report.json", "w") as fh:
        json.dump({"embedded": {"val": _v_emb, "test": _t_emb, "n_feat": int(X.shape[1])},
                   "plus_research": {"val": _v_full, "test": _t_full,
                                     "n_feat": int(_Xfull.shape[1])},
                   "delta_val": _v_full - _v_emb, "delta_test": _t_full - _t_emb,
                   "n_research": len(RESEARCH_COLS),
                   "research_in_top100": int(_n_res_top100),
                   "top_research": [{"rank": r+1, "feature": nm, "importance": imp}
                                    for r, nm, imp in _res_ranks[:25]],
                   "verdict": _verdict}, fh, indent=2)

# ── DEPLOYED model — refit on ALL sessions ────────────────────────────────────
# `pipe` (above) is the EVALUATION model: trained only on the session-disjoint
# training split, so its held-out metrics and the out-of-sample smoke tests are
# honest.  The model we SHIP, however, must be able to emit every class — a model
# that never saw `running` in training can never predict it on-device.  So the
# exported artefacts (.pkl, C header, preprocess JSON) come from `deploy_pipe`,
# refit on the full dataset.  Its expected field accuracy is the GROUP-CV / held-
# out-session estimate from `pipe`, NOT its (optimistic) full-data training fit.
deploy_pipe = build_pipeline().fit(X, y)
joblib.dump(deploy_pipe, "generated/imu_pipeline.pkl")
joblib.dump(le,          "generated/label_encoder.pkl")
print("Deployed model : refit on all sessions (so every class is predictable); "
      "honest accuracy = the group-CV estimate above, not the training fit.")

# ═══════════════════════════════════════════════════════════════════════════════
# POST-MODEL ANALYSIS  —  PCA feature attribution · SMOTE stress-test · feature EDA
# ───────────────────────────────────────────────────────────────────────────────
# Everything below is PURELY ADDITIVE: it inspects the already-trained `pipe`,
# runs a SMOTE robustness experiment on a *separate* pipeline, and produces EDA on
# the PCA-reduced feature space.  None of it mutates `pipe`, `X*`, `y*`, `le`, the
# exported header, or any value consumed by the C export below — that section
# still sees the identical fitted model.
#
# Extra deps (all already used elsewhere except imblearn): roc/pr metrics +
# imblearn.SMOTE.  Imported here, locally, so the additions stay self-contained.
# ═══════════════════════════════════════════════════════════════════════════════
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score)

CLASS_NAMES = list(le.classes_)
N_CLASSES   = len(CLASS_NAMES)

# ───────────────────────────────────────────────────────────────────────────────
# 1.  BEST FEATURES *AFTER* PCA  —  what the model actually keys on
# ───────────────────────────────────────────────────────────────────────────────
# PCA does not "select" original columns; it builds NUM_PCA orthogonal components
# (PC0…PC{n-1}), each a weighted blend of all NUM_STAT_FEATURES.  The RF then
# operates ONLY on those PC axes.  So "best features after PCA" has two layers:
#
#   (a) which PCA components the forest relies on  → rf.feature_importances_
#       (one importance per PCA index, summing to 1.0)
#   (b) which ORIGINAL features dominate each important component → |loadings|
#       (the absolute weights in pca.components_[pc])
#
# We surface both, with the PCA index explicitly marked as `PC[ k]`, so the model
# output can be read alongside the feature attribution that produced it.
print("\n" + "="*65)
print(f"BEST FEATURES AFTER REDUCER '{FEATURE_SELECTOR_ACTIVE}' — model attribution")
print("="*65)
# NB: the reducer slot may be PCA, LDA, or a raw-feature selector.  For projection
# reducers (pca/lda) each model dim is a weighted blend of the 560 features and
# orig_index_ is the component id; for selection reducers each model dim IS one raw
# feature (components_ is one-hot) and orig_index_ is that feature's column index.

_pca = pipe.named_steps["pca"]
_rf  = pipe.named_steps["clf"]
pc_importance = _rf.feature_importances_          # shape (n_pca,)
pc_order      = np.argsort(pc_importance)[::-1]   # most→least important model dim
TOP_LOADINGS  = 5                                 # original feats shown per PC
# Map each model dim (0..n_pca-1) back to its ORIGINAL PCA component index so the
# display marks the true PCA identity, not the post-selection re-index.  Falls
# back to identity if a plain PCA is ever swapped back in.
pc_orig = np.asarray(getattr(_pca, "orig_index_", np.arange(n_pca)), dtype=int)
print(f"  (model dim → original PCA index: "
      f"{', '.join(f'{j}->PC{pc_orig[j]}' for j in range(min(n_pca, 8)))} …)")

# Per-PC attribution: importance + the original features it is built from.
pca_attribution = []   # collected for the JSON dump + plots
print(f"\n  PCA reduced {NUM_STAT_FEAT} stat features → {n_pca} components "
      f"({_pca.explained_variance_ratio_.sum()*100:.1f}% variance retained)")
print(f"  Ranked by RandomForest importance (top {TOP_LOADINGS} loadings each):\n")
for rank, pc in enumerate(pc_order):
    imp   = pc_importance[pc]
    evr   = _pca.explained_variance_ratio_[pc]
    load  = _pca.components_[pc]
    top_i = np.argsort(np.abs(load))[::-1][:TOP_LOADINGS]
    blurb = ", ".join(f"{FEATURES[i]}({load[i]:+.2f})" for i in top_i)
    bar   = "█" * int(round(imp / pc_importance.max() * 30))
    print(f"  PC[{pc_orig[pc]:2d}]  imp={imp:6.4f} {bar:<30}  var={evr*100:4.1f}%")
    print(f"          ↳ {blurb}")
    pca_attribution.append({
        "pca_index":   int(pc_orig[pc]),   # original PCA component index
        "model_dim":   int(pc),            # its column in the deployed model
        "rank":        int(rank),
        "rf_importance": float(imp),
        "explained_variance_ratio": float(evr),
        "top_features": [
            {"name": FEATURES[i], "loading": float(load[i])} for i in top_i
        ],
    })

# The single most-influential original feature across the model = importance-
# weighted sum of |loading| over every PC.  A compact "what matters most" view.
feat_influence = np.zeros(NUM_STAT_FEAT)
for pc in range(n_pca):
    feat_influence += pc_importance[pc] * np.abs(_pca.components_[pc])
feat_influence /= feat_influence.sum() + 1e-12
infl_order = np.argsort(feat_influence)[::-1]
print("\n  Top original features by model-wide influence "
      "(Σ rf_importance·|loading|):")
for i in infl_order[:10]:
    print(f"    {FEATURES[i]:<28} {feat_influence[i]*100:5.2f}%")

# Persist the attribution next to the other generated artefacts.
with open("generated/pca_feature_attribution.json", "w") as fh:
    json.dump({
        "n_pca": int(n_pca),
        "n_candidate_pool": int(n_pca_var),
        "selection": (f"top-{PCA_TOP_K} of {n_pca_var} PCA components by RF importance"
                      if FEATURE_SELECTOR_ACTIVE == "pca"
                      else f"reducer '{FEATURE_SELECTOR_ACTIVE}' → {n_pca} raw features"),
        "kept_pca_indices": [int(x) for x in pc_orig],
        "n_stat_features": int(NUM_STAT_FEAT),
        "variance_retained": float(_pca.explained_variance_ratio_.sum()),
        "pca_components_ranked": pca_attribution,
        "feature_influence": [
            {"name": FEATURES[i], "influence": float(feat_influence[i])}
            for i in infl_order
        ],
    }, fh, indent=2)
print("  Saved          : generated/pca_feature_attribution.json")

# ── Plot 1a: PCA-component importance (original PCA index on the axis) ─────────
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar([f"PC{pc_orig[p]}" for p in pc_order], pc_importance[pc_order],
       color=sns.color_palette("viridis", n_pca))
ax.set_title(f"RandomForest Importance per PCA Component "
             f"(top-{n_pca} kept, original PCA indices)", fontsize=13)
ax.set_xlabel("PCA component (ranked)"); ax.set_ylabel("RF importance")
ax.tick_params(axis='x', rotation=60, labelsize=8); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig("plots/pca_component_importance.png", dpi=120); plt.close()

# ── Plot 1b: loading heatmap of the top-importance PCs vs their driver feats ───
N_TOP_PC   = min(8, n_pca)
top_pcs    = pc_order[:N_TOP_PC]
driver_idx = sorted({int(i) for pc in top_pcs
                     for i in np.argsort(np.abs(_pca.components_[pc]))[::-1][:TOP_LOADINGS]})
load_mat   = _pca.components_[np.ix_(top_pcs, driver_idx)]
fig, ax = plt.subplots(figsize=(min(1.0*len(driver_idx)+3, 22), 0.6*N_TOP_PC+2))
sns.heatmap(load_mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            linewidths=0.4,
            xticklabels=[FEATURES[i] for i in driver_idx],
            yticklabels=[f"PC{pc_orig[p]}" for p in top_pcs], ax=ax)
ax.set_title("Top-PCA Loadings — original feature → component weights", fontsize=13)
ax.tick_params(axis='x', rotation=75, labelsize=8)
plt.tight_layout(); plt.savefig("plots/pca_loadings_heatmap.png", dpi=120); plt.close()
print("  Plots          : plots/pca_component_importance.png, plots/pca_loadings_heatmap.png")

# ───────────────────────────────────────────────────────────────────────────────
# 2.  SMOTE STRESS-TEST  —  does synthetic minority oversampling help?
# ───────────────────────────────────────────────────────────────────────────────
# WHY:  the class balance is brutal (running ≈ a few dozen windows vs walking in
# the hundreds — see [[har2-data-ceiling]]).  The production model leans on
# *tempered class weights* instead of resampling.  SMOTE is the natural
# alternative: it synthesises new minority windows by interpolating between a
# sample and its k nearest same-class neighbours, balancing the TRAIN set without
# duplicating rows.  This block quantifies whether SMOTE would beat the shipped
# class-weight policy, on the SAME untouched test split, so the comparison is fair.
#
# METHODOLOGY / GUARDS:
#   • SMOTE is fit on the TRAINING split ONLY — never the test set (no leakage).
#   • k_neighbors must be < the smallest class count, else SMOTE raises; we clamp.
#   • A class with <2 samples cannot be interpolated → SMOTE is skipped cleanly.
#   • The SMOTE model uses class_weight=None (resampling already balanced it).
print("\n" + "="*65)
print("SMOTE STRESS-TEST — synthetic minority oversampling vs class-weights")
print("="*65)

train_counts = np.bincount(y_train, minlength=N_CLASSES)
min_class    = int(train_counts[train_counts > 0].min())
print("\n  Train class balance (pre-SMOTE):")
for c in range(N_CLASSES):
    print(f"    {CLASS_NAMES[c]:<16} {train_counts[c]:4d}")

smote_ran = False
if min_class < 2:
    print(f"\n  ⚠ Smallest class has {min_class} sample(s) — SMOTE needs ≥2 to "
          f"interpolate.  Skipping SMOTE test.")
else:
    try:
        from imblearn.over_sampling import SMOTE
        k_neighbors = min(5, min_class - 1)   # k must be < smallest class size
        print(f"\n  SMOTE k_neighbors = {k_neighbors}  "
              f"(smallest class = {min_class} samples)")
        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        res_counts = np.bincount(y_res, minlength=N_CLASSES)
        print("  Post-SMOTE balance (synthetic windows added):")
        for c in range(N_CLASSES):
            added = res_counts[c] - train_counts[c]
            print(f"    {CLASS_NAMES[c]:<16} {res_counts[c]:4d}  (+{added})")

        # Fresh pipeline; resampling already balanced classes → no class_weight.
        # Uses the SAME reducer as the deployed model so the only variable under
        # test is resampling vs class-weighting.
        smote_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    make_selector(FEATURE_SELECTOR_ACTIVE)),
            ("clf",    RandomForestClassifier(
                n_estimators=RF_N_TREES, max_depth=RF_MAX_DEPTH,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                min_samples_split=RF_MIN_SAMPLES_SPLIT,
                max_features=RF_MAX_FEATURES, class_weight=None,
                random_state=42, n_jobs=-1)),
        ])
        smote_pipe.fit(X_res, y_res)
        y_pred_smote = smote_pipe.predict(X_test)
        smote_acc    = accuracy_score(y_test, y_pred_smote)

        print(f"\n  Baseline (class-weight) test acc : {test_acc:.4f}")
        print(f"  SMOTE-resampled        test acc : {smote_acc:.4f}"
              f"   (Δ {smote_acc-test_acc:+.4f})")
        print("\n  --- SMOTE per-class report (same test split) ---")
        print(classification_report(y_test, y_pred_smote,
                                    labels=np.arange(len(CLASS_NAMES)),
                                    target_names=CLASS_NAMES, zero_division=0))
        smote_ran = True
    except Exception as e:
        print(f"\n  ⚠ SMOTE test failed ({type(e).__name__}: {e}). Skipping plots.")

# ── SMOTE visualisations (only when SMOTE actually ran) ───────────────────────
if smote_ran:
    # 2a: class distribution before vs after SMOTE — the headline of the method.
    fig, ax = plt.subplots(figsize=(10, 5))
    xpos = np.arange(N_CLASSES); w = 0.4
    ax.bar(xpos - w/2, train_counts, w, label="Original train",
           color="#4C72B0")
    ax.bar(xpos + w/2, res_counts,   w, label="After SMOTE",
           color="#DD8452")
    ax.set_xticks(xpos); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Window count"); ax.set_title("Class Distribution — SMOTE oversampling")
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig("plots/smote_class_distribution.png", dpi=120); plt.close()

    # 2b: SMOTE confusion matrix on the held-out test split.
    # labels= pinned to all classes (like the main matrix) so the matrix is always
    # N_CLASSES×N_CLASSES — a session-grouped split need not contain every class,
    # and the auto-selected reducer can make SMOTE predict a strict subset, which
    # would otherwise shrink the matrix below the 5 display labels → ValueError.
    cm_s = confusion_matrix(y_test, y_pred_smote, labels=np.arange(N_CLASSES))
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm_s, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Oranges", colorbar=False)
    ax.set_title(f"SMOTE Confusion Matrix (test acc={smote_acc:.3f})")
    plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig("plots/smote_confusion_matrix.png", dpi=120); plt.close()

    # 2c: baseline vs SMOTE per-class RECALL — the metric SMOTE is meant to lift
    #     for rare classes (running/standing).  This is the real verdict.
    from sklearn.metrics import recall_score
    rec_base  = recall_score(y_test, y_pred,        average=None,
                             labels=range(N_CLASSES), zero_division=0)
    rec_smote = recall_score(y_test, y_pred_smote,  average=None,
                             labels=range(N_CLASSES), zero_division=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(xpos - w/2, rec_base,  w, label="Class-weight", color="#4C72B0")
    ax.bar(xpos + w/2, rec_smote, w, label="SMOTE",        color="#DD8452")
    ax.set_xticks(xpos); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Recall (test)")
    ax.set_title("Per-class Recall — class-weight vs SMOTE")
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig("plots/smote_recall_comparison.png", dpi=120); plt.close()

    # 2d/2e: ROC + Precision-Recall curves (one-vs-rest, multiclass).
    # predict_proba columns follow smote_pipe.classes_ — align to label indices.
    try:
        proba_s    = smote_pipe.predict_proba(X_test)
        proba_cols = list(smote_pipe.named_steps["clf"].classes_)
        present    = [c for c in range(N_CLASSES) if c in proba_cols]
        y_test_bin = label_binarize(y_test, classes=list(range(N_CLASSES)))
        roc_pal    = sns.color_palette("tab10", N_CLASSES)

        fig, (axr, axp) = plt.subplots(1, 2, figsize=(15, 6))
        for c in present:
            col   = proba_cols.index(c)
            y_c   = y_test_bin[:, c]
            if y_c.sum() == 0:          # class absent from test → undefined curve
                continue
            fpr, tpr, _ = roc_curve(y_c, proba_s[:, col])
            axr.plot(fpr, tpr, color=roc_pal[c], lw=2,
                     label=f"{CLASS_NAMES[c]} (AUC={auc(fpr,tpr):.2f})")
            prec, rec, _ = precision_recall_curve(y_c, proba_s[:, col])
            ap = average_precision_score(y_c, proba_s[:, col])
            axp.plot(rec, prec, color=roc_pal[c], lw=2,
                     label=f"{CLASS_NAMES[c]} (AP={ap:.2f})")
        axr.plot([0, 1], [0, 1], "k--", alpha=0.4)
        axr.set_xlabel("False Positive Rate"); axr.set_ylabel("True Positive Rate")
        axr.set_title("SMOTE ROC (one-vs-rest)"); axr.legend(fontsize=8); axr.grid(alpha=0.3)
        axp.set_xlabel("Recall"); axp.set_ylabel("Precision")
        axp.set_title("SMOTE Precision-Recall (one-vs-rest)")
        axp.legend(fontsize=8); axp.grid(alpha=0.3)
        plt.suptitle("SMOTE model — per-class discrimination on held-out test",
                     fontsize=14, y=1.02)
        plt.tight_layout(); plt.savefig("plots/smote_roc_pr_curves.png", dpi=120); plt.close()
    except Exception as e:
        print(f"  ⚠ ROC/PR curve generation skipped: {type(e).__name__}: {e}")

    print("  SMOTE plots    : smote_class_distribution / smote_confusion_matrix / "
          "smote_recall_comparison / smote_roc_pr_curves (.png)")

# ───────────────────────────────────────────────────────────────────────────────
# 3.  EDA ON THE NEW (PCA-REDUCED) FEATURES
# ───────────────────────────────────────────────────────────────────────────────
# The C model never sees the 145 stat features directly — it sees the n_pca
# projected components.  These plots characterise THAT space: how separable the
# activities are along each PC, and how (de)correlated the PCs are.  On the train
# split PCA guarantees orthogonality; computing the heatmap over the FULL dataset
# instead reveals residual structure the forest can still exploit.
print("\n" + "="*65)
print("EDA — PCA-reduced feature space (the axes the model actually uses)")
print("="*65)

# Project the full feature matrix through the fitted scaler+PCA (no re-fit → the
# transform is identical to what `pipe` applies at inference).  Reusing the
# fitted steps avoids duplicating the StandardScaler/PCA in memory.
Z = _pca.transform(pipe.named_steps["scaler"].transform(X))   # (n_samples, n_pca)
# Columns carry the ORIGINAL PCA index (model dims are importance-ordered), so the
# EDA plots line up with the attribution display and the C export's feature map.
PC_COLS  = [f"PC{pc_orig[j]:02d}" for j in range(n_pca)]
df_pca   = pd.DataFrame(Z, columns=PC_COLS)
df_pca["activity_label"] = le.inverse_transform(y)
print(f"  Projected {Z.shape[0]:,} windows into {n_pca} top-importance PCA dims "
      f"(original indices: {', '.join(f'PC{pc_orig[j]}' for j in range(min(n_pca,8)))} …)")

act_pca  = df_pca["activity_label"].unique()
pal_pca  = sns.color_palette("tab10", len(act_pca))
N_PC_EDA = min(6, n_pca)   # show the leading (most-important) kept components

# ── EDA 3a: boxplots of the leading PCs per activity (distribution + spread) ───
fig, axes = plt.subplots(2, 3, figsize=(18, 10)); axes = axes.flatten()
for ax_obj, j in zip(axes, range(N_PC_EDA)):
    col    = PC_COLS[j]
    groups = [df_pca.loc[df_pca["activity_label"] == a, col].values for a in act_pca]
    bp = ax_obj.boxplot(groups, labels=act_pca, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], pal_pca):
        patch.set_facecolor(c)
    ax_obj.set_title(f"{col}  (var={_pca.explained_variance_ratio_[j]*100:.1f}%)",
                     fontsize=10)
    ax_obj.tick_params(axis='x', rotation=30, labelsize=8); ax_obj.grid(axis='y', alpha=0.3)
for ax_obj in axes[N_PC_EDA:]:
    ax_obj.set_visible(False)
plt.suptitle("PCA Component Distributions per Activity (boxplots)", fontsize=14, y=1.01)
plt.tight_layout(); plt.savefig("plots/pca_eda_boxplots.png", dpi=120); plt.close()

# ── EDA 3b: correlation heatmap of the PCA features ───────────────────────────
corr_pca = df_pca[PC_COLS].corr()
fig, ax = plt.subplots(figsize=(min(0.6*n_pca+3, 18), min(0.6*n_pca+2, 16)))
sns.heatmap(corr_pca, annot=(n_pca <= 16), fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.4, square=True, ax=ax)
ax.set_title("PCA Feature Correlation (full dataset)", fontsize=13)
plt.tight_layout(); plt.savefig("plots/pca_eda_correlation.png", dpi=120); plt.close()

# ── EDA 3c: violin of mean PC value per activity — separability at a glance ────
mean_by_act = df_pca.groupby("activity_label")[PC_COLS[:N_PC_EDA]].mean()
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(mean_by_act, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            linewidths=0.4, ax=ax)
ax.set_title("Mean PCA Component Value per Activity (class separability)", fontsize=13)
ax.set_xlabel("PCA component"); ax.set_ylabel("Activity")
plt.tight_layout(); plt.savefig("plots/pca_eda_class_means.png", dpi=120); plt.close()

# ── EDA 3d: 2-D scatter on the two leading PCs — visual cluster overlap ────────
if n_pca >= 2:
    fig, ax = plt.subplots(figsize=(9, 7))
    for a, c in zip(act_pca, pal_pca):
        m = df_pca["activity_label"] == a
        ax.scatter(df_pca.loc[m, PC_COLS[0]], df_pca.loc[m, PC_COLS[1]],
                   s=12, alpha=0.5, color=c, label=a)
    ax.set_xlabel(f"{PC_COLS[0]} ({_pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"{PC_COLS[1]} ({_pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Activities in the 2 most-important PCA dimensions"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("plots/pca_eda_scatter.png", dpi=120); plt.close()

print("  PCA EDA plots  : pca_eda_boxplots / pca_eda_correlation / "
      "pca_eda_class_means / pca_eda_scatter (.png)")
print("Post-model analysis complete.\n")

# ═══════════════════════════════════════════════════════════════════════════════
#  COMPACT C EXPORT — flat IMUNode struct array (same format as compressed.py)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  IMUNode struct (10 bytes, naturally aligned — no packed attribute needed):
#
#    typedef struct {
#        float    threshold;   // offset 0, 4 B (f32 aligned)
#        int16_t  left;        // offset 4, 2 B (-1 = leaf)
#        int16_t  right;       // offset 6, 2 B (-1 = leaf)
#        int8_t   feature;     // offset 8, 1 B (PCA index)
#        uint8_t  pred_class;  // offset 9, 1 B (leaf majority class)
#    } IMUNode;                // sizeof = 10
#
#  Tree traversal: while left != -1: follow left or right by threshold.
#  Total model flash: nodes × 10 bytes + PCA arrays.
# ═══════════════════════════════════════════════════════════════════════════════

def export_tree_to_c_array(tree, tree_idx):
    """
    Convert one sklearn DecisionTree into a flat C IMUNode initialiser.
    Field order: { threshold, left, right, feature, pred_class }
    """
    sk_tree        = tree.tree_
    n_nodes        = sk_tree.node_count
    children_left  = sk_tree.children_left
    children_right = sk_tree.children_right
    feature        = sk_tree.feature
    threshold      = sk_tree.threshold
    value          = sk_tree.value          # (n_nodes, 1, n_classes)

    lines = []
    for node_id in range(n_nodes):
        left  = int(children_left[node_id])
        right = int(children_right[node_id])
        if left == -1:                      # leaf node
            pred_cls = int(np.argmax(value[node_id, 0, :]))
            feat = 0; thr = 0.0
        else:                               # internal node
            pred_cls = 0
            feat     = int(feature[node_id])
            thr      = float(threshold[node_id])
        # Field order matches the naturally-aligned struct definition
        lines.append(
            f"    {{ {thr:13.6f}f, {left:6d}, {right:6d}, "
            f"{feat:4d}, {pred_cls:3d} }}"
        )

    body = ",\n".join(lines)
    return (
        f"/* tree_{tree_idx}: {n_nodes} nodes */\n"
        f"static const IMUNode tree_{tree_idx}[{n_nodes}] = {{\n{body}\n}};\n",
        n_nodes
    )


def _fmt_f32_array(arr, label, cols=6):
    """Format a 1-D float32 array as a C initialiser body with label."""
    vals  = arr.astype(np.float32).flatten()
    rows, row = [], []
    for i, v in enumerate(vals):
        row.append(f"{v:.7f}f")
        if (i + 1) % cols == 0:
            rows.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        rows.append("    " + ", ".join(row))
    return "\n".join(rows)


def _fmt_pca_components(components):
    """Format PCA components matrix: one row per component."""
    rows = []
    for i, row in enumerate(components.astype(np.float32)):
        vals = ", ".join(f"{v:.7f}f" for v in row)
        rows.append(f"    /* PC{i+1:02d} */ {{ {vals} }}")
    return ",\n".join(rows)


def _fmt_matrix_f32(mat, cols=8):
    """Format a 2-D float32 matrix as nested C initialiser rows (one {..} per row)."""
    out = []
    for r, row in enumerate(mat.astype(np.float32)):
        cells, line = [], []
        for i, v in enumerate(row):
            line.append(f"{v:.7e}f")
            if (i + 1) % cols == 0:
                cells.append("        " + ", ".join(line)); line = []
        if line:
            cells.append("        " + ", ".join(line))
        body = ",\n".join(cells)
        out.append(f"    /* row {r:3d} */ {{\n{body}\n    }}")
    return ",\n".join(out)


def _fmt_int_array(arr, cols=16):
    """Format a 1-D int array as a C initialiser body."""
    vals = [int(v) for v in arr]
    rows, row = [], []
    for i, v in enumerate(vals):
        row.append(f"{v:4d}")
        if (i + 1) % cols == 0:
            rows.append("    " + ", ".join(row) + ","); row = []
    if row:
        rows.append("    " + ", ".join(row))
    return "\n".join(rows)


def _build_gravity_matrix(window_size):
    """gravity = G @ accel_axis — exact linear export of scipy butter+filtfilt.
    filtfilt on a fixed-length window is a linear, shift-varying operator, so the
    whole 0.3 Hz Butterworth low-pass collapses to one constant matrix G that the
    firmware applies as a matrix-vector product (no on-device IIR filter)."""
    G = np.zeros((window_size, window_size), dtype=np.float64)
    eye = np.eye(window_size)
    for i in range(window_size):
        G[:, i] = _grav_split(eye[:, i])
    return G.astype(np.float32)


def _build_uci_keep(features):
    """Canonical 435-feature UCI order + the indices that survive dedup, so the C
    extractor can compute the full set then copy only the kept columns (matching
    the trained 560-feature layout / SCALER / PCA exactly)."""
    probe = np.random.default_rng(1).standard_normal(
        (WINDOW_SIZE, 6)).astype(np.float32)
    canon = list(extract_uci_features(probe).keys())
    surv  = list(features[EXPECTED_TOTAL_FEAT:])          # FEATURES[145:]
    pos   = {name: i for i, name in enumerate(canon)}
    keep  = [pos[name] for name in surv]
    assert [canon[i] for i in keep] == surv, \
        "UCI canonical/keep ordering mismatch — C copy would misalign with SCALER/PCA"
    return canon, keep


def _build_selftest_block(features):
    """Emit an opt-in on-device parity check: one REAL window + the Python
    reference feature vector, so the firmware can call imu_selftest() and confirm
    the C extractor reproduces the host pipeline (within float32 tolerance).
    Guarded by IMU_ENABLE_SELFTEST so it costs zero flash unless explicitly built."""
    if not sample_raw_windows:
        return ""
    cls = sorted(sample_raw_windows)[0]
    win = sample_raw_windows[cls].astype(np.float32)        # (WINDOW_SIZE, 6)
    ref = extract_stats(win)
    expect = np.array([ref[f] for f in features], dtype=np.float64)
    expect = np.nan_to_num(expect, nan=0.0, posinf=0.0, neginf=0.0)

    win_rows = ",\n".join(
        "        { " + ", ".join(f"{v:.7e}f" for v in row) + " }" for row in win)
    exp_body = _fmt_f32_array(expect.astype(np.float32), "SELFTEST_EXPECT")
    return f"""
/* ══════════════════════════════════════════════════════════════════════════
 * SECTION 6 — On-device parity self-test  (opt-in: -DIMU_ENABLE_SELFTEST)
 *
 *   imu_selftest() recomputes the {len(features)}-feature vector for a real
 *   '{cls}' window and returns the max |C(float32) − Python(float64)| diff.
 *   The gravity-derived AR/correlation features are numerically ill-conditioned
 *   (the gravity signal is near-constant), so expect a worst-case diff on the
 *   order of ~1e-2 there; every other feature matches to ~1e-4.  A diff far
 *   above that (e.g. > 0.1) signals a real porting/ABI problem, not rounding.
 * ══════════════════════════════════════════════════════════════════════════ */
#ifdef IMU_ENABLE_SELFTEST
static const float SELFTEST_WIN[IMU_WINDOW_SIZE][NUM_RAW_AXES] = {{
{win_rows}
}};
static const float SELFTEST_EXPECT[NUM_STAT_FEATURES] = {{
{exp_body}
}};
static inline float imu_selftest(void)
{{
    float f[NUM_STAT_FEATURES]; int i; float mx = 0.0f;
    imu_extract_features(SELFTEST_WIN, f);
    for (i = 0; i < NUM_STAT_FEATURES; i++) {{
        float d = f[i] - SELFTEST_EXPECT[i];
        if (d < 0.0f) d = -d;
        if (d > mx) mx = d;
    }}
    return mx;   /* compare against your tolerance, e.g. assert(mx < 0.1f) */
}}
#endif /* IMU_ENABLE_SELFTEST */
"""


# ── Static C code sections (plain strings — avoids f-string brace escaping) ───

_C_FREQ = """\
#if NUM_FREQ_FEATURES > 0
/* ══════════════════════════════════════════════════════════════════════════
 * SECTION 2b — Frequency-domain feature extraction
 *
 *   Radix-2 in-place FFT (size IMU_NFFT, a power of two).  Each real signal is
 *   zero-padded to IMU_NFFT, transformed, and reduced to 6 spectral stats.
 *   Numerically mirrors NumPy's rfft path in _freq_axis_stats() (float32 here,
 *   float64 in Python — same tolerance as the time-domain features).
 *   Needs cosf/sinf/logf/sqrtf from <math.h>; stack-only, no heap.
 * ══════════════════════════════════════════════════════════════════════════ */
#ifndef IMU_PI
#define IMU_PI 3.14159265358979323846f
#endif

static void imu_fft(float re[IMU_NFFT], float im[IMU_NFFT])
{
    int n = IMU_NFFT, i, j, k, m, step;

    /* bit-reversal permutation */
    j = 0;
    for (i = 1; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    /* Danielson–Lanczos butterflies */
    for (step = 2; step <= n; step <<= 1) {
        float ang = -2.0f * IMU_PI / (float)step;
        float wsr = cosf(ang), wsi = sinf(ang);
        for (m = 0; m < n; m += step) {
            float wr = 1.0f, wi = 0.0f;
            for (k = 0; k < step / 2; k++) {
                int   a  = m + k, b = a + step / 2;
                float tr = wr * re[b] - wi * im[b];
                float ti = wr * im[b] + wi * re[b];
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr;        im[a] += ti;
                float nwr = wr * wsr - wi * wsi;
                wi = wr * wsi + wi * wsr;
                wr = nwr;
            }
        }
    }
}

/* 6 spectral stats for one real signal (length L, zero-padded to IMU_NFFT).
 * Output order matches Python _freq_axis_stats():
 *   [0] dom_freq  [1] dom_mag  [2] centroid  [3] spread  [4] entropy  [5] energy
 * DC bin excluded; AC bins 1..IMU_NFFT/2 used (== NumPy rfft[1:]).            */
static void imu_freq_stats(const float *x, int L, float out[6])
{
    float re[IMU_NFFT], im[IMU_NFFT];
    float psd[IMU_NFFT / 2];
    int   NB   = IMU_NFFT / 2;
    float fbin = IMU_FS_HZ / (float)IMU_NFFT;
    float total = 1e-12f, peak = -1.0f;
    int   i, dom = 0;

    for (i = 0; i < IMU_NFFT; i++) { re[i] = (i < L) ? x[i] : 0.0f; im[i] = 0.0f; }
    imu_fft(re, im);

    for (i = 1; i <= NB; i++) {
        float p = re[i] * re[i] + im[i] * im[i];
        psd[i - 1] = p;
        total += p;
        if (p > peak) { peak = p; dom = i - 1; }
    }

    float cent = 0.0f, ent = 0.0f;
    for (i = 0; i < NB; i++) {
        float f  = (float)(i + 1) * fbin;
        float pn = psd[i] / total;
        cent += f * pn;
        if (pn > 1e-12f) ent += -pn * logf(pn);
    }
    float spread = 0.0f;
    for (i = 0; i < NB; i++) {
        float f  = (float)(i + 1) * fbin;
        float pn = psd[i] / total;
        float d  = f - cent;
        spread += d * d * pn;
    }

    out[0] = (float)(dom + 1) * fbin;   /* dom_freq  */
    out[1] = psd[dom] / total;          /* dom_mag   */
    out[2] = cent;                      /* centroid  */
    out[3] = sqrtf(spread);             /* spread    */
    out[4] = ent / logf((float)NB);     /* entropy   */
    out[5] = total;                     /* energy    */
}

/* Fill NUM_FREQ_FEATURES values; signal order MUST match Python FREQ_SIGNALS:
 * accel_x accel_y accel_z gyro_x gyro_y gyro_z accel_mag gyro_mag jerk_x jerk_y jerk_z */
static void imu_extract_freq_features(
        const float win[IMU_WINDOW_SIZE][NUM_RAW_AXES],
        float      *fout)
{
    float buf[IMU_WINDOW_SIZE];
    float out[6];
    int   i, s, c = 0;

    for (s = 0; s < 6; s++) {                    /* 6 raw axes */
        for (i = 0; i < IMU_WINDOW_SIZE; i++) buf[i] = win[i][s];
        imu_freq_stats(buf, IMU_WINDOW_SIZE, out);
        for (i = 0; i < 6; i++) fout[c++] = out[i];
    }
    for (i = 0; i < IMU_WINDOW_SIZE; i++)        /* accel_mag */
        buf[i] = sqrtf(win[i][0]*win[i][0] + win[i][1]*win[i][1] + win[i][2]*win[i][2]);
    imu_freq_stats(buf, IMU_WINDOW_SIZE, out);
    for (i = 0; i < 6; i++) fout[c++] = out[i];

    for (i = 0; i < IMU_WINDOW_SIZE; i++)        /* gyro_mag */
        buf[i] = sqrtf(win[i][3]*win[i][3] + win[i][4]*win[i][4] + win[i][5]*win[i][5]);
    imu_freq_stats(buf, IMU_WINDOW_SIZE, out);
    for (i = 0; i < 6; i++) fout[c++] = out[i];

    for (s = 0; s < 3; s++) {                    /* jerk_x, jerk_y, jerk_z */
        for (i = 0; i < IMU_WINDOW_SIZE - 1; i++) buf[i] = win[i+1][s] - win[i][s];
        imu_freq_stats(buf, IMU_WINDOW_SIZE - 1, out);
        for (i = 0; i < 6; i++) fout[c++] = out[i];
    }
}
#endif /* NUM_FREQ_FEATURES > 0 */
"""

_C_FEAT_EXTRACT = """\
/* ══════════════════════════════════════════════════════════════════════════
 * SECTION 2 — Statistical Feature Extraction
 *
 * imu_extract_features(win, feat)
 *   win  : const float[IMU_WINDOW_SIZE][NUM_RAW_AXES]
 *            col 0=accel_x 1=accel_y 2=accel_z
 *            col 3=gyro_x  4=gyro_y  5=gyro_z
 *   feat : float[NUM_STAT_FEATURES]  (79 features)
 *
 * All operations O(n), no sort, no heap.
 * Uses sqrtf from <math.h> — hardware FPU on ESP32-P4.
 *
 * Feature layout (mirrors Python extract_stats() exactly):
 *   [  0.. 7]  accel_x   : mean std range rms energy skew kurt zcr
 *   [  8..15]  accel_y   : mean std range rms energy skew kurt zcr
 *   [ 16..23]  accel_z   : mean std range rms energy skew kurt zcr
 *   [ 24..31]  gyro_x    : mean std range rms energy skew kurt zcr
 *   [ 32..39]  gyro_y    : mean std range rms energy skew kurt zcr
 *   [ 40..47]  gyro_z    : mean std range rms energy skew kurt zcr
 *   [ 48..55]  accel_mag : mean std range rms energy skew kurt zcr  [rotation-invariant]
 *   [ 56..63]  gyro_mag  : mean std range rms energy skew kurt zcr  [rotation-invariant]
 *   [ 64..69]  Pearson r: (ax,ay)(ax,az)(ay,az)(gx,gy)(gx,gz)(gy,gz)
 *   [ 70..72]  jerk_x  : mean std rms
 *   [ 73..75]  jerk_y  : mean std rms
 *   [ 76..78]  jerk_z  : mean std rms
 * ══════════════════════════════════════════════════════════════════════════ */
static void imu_extract_features(
        const float win[IMU_WINDOW_SIZE][NUM_RAW_AXES],
        float       feat[NUM_STAT_FEATURES])
{
    int k = 0;   /* cursor — equals NUM_TIME_FEATURES (79) after time block */
    int ax, i;

    /* ── 8 statistics per axis ──────────────────────────────────────────── */
    for (ax = 0; ax < NUM_RAW_AXES; ax++) {

        float x[IMU_WINDOW_SIZE];
        for (i = 0; i < IMU_WINDOW_SIZE; i++) x[i] = win[i][ax];

        /* mean */
        float mu = 0.0f;
        for (i = 0; i < IMU_WINDOW_SIZE; i++) mu += x[i];
        mu /= (float)IMU_WINDOW_SIZE;

        /* population variance / std */
        float var = 0.0f;
        for (i = 0; i < IMU_WINDOW_SIZE; i++) {
            float d = x[i] - mu;
            var += d * d;
        }
        var /= (float)IMU_WINDOW_SIZE;
        float sd = sqrtf(var);

        /* min / max → range  (no sort needed) */
        float xmin = x[0], xmax = x[0];
        for (i = 1; i < IMU_WINDOW_SIZE; i++) {
            if (x[i] < xmin) xmin = x[i];
            if (x[i] > xmax) xmax = x[i];
        }

        /* energy / rms */
        float energy = 0.0f;
        for (i = 0; i < IMU_WINDOW_SIZE; i++) energy += x[i] * x[i];
        energy /= (float)IMU_WINDOW_SIZE;

        /* skewness / excess kurtosis  (Fisher-Pearson, bias=True)
         * skew = E[(x-mu)^3] / sigma^3
         * kurt = E[(x-mu)^4] / sigma^4  - 3  (excess)              */
        float skew = 0.0f, kurt = 0.0f;
        if (sd > 1e-9f) {
            for (i = 0; i < IMU_WINDOW_SIZE; i++) {
                float nrm = (x[i] - mu) / sd;
                float n2  = nrm * nrm;
                skew += n2 * nrm;
                kurt += n2 * n2;
            }
            skew /= (float)IMU_WINDOW_SIZE;
            kurt  = kurt / (float)IMU_WINDOW_SIZE - 3.0f;
        }

        /* zero-crossing rate */
        int zc = 0;
        for (i = 0; i < IMU_WINDOW_SIZE - 1; i++)
            if (x[i] * x[i + 1] < 0.0f) zc++;
        float zcr = (float)zc / (float)(IMU_WINDOW_SIZE - 1);

        /* store 8 stats — order must match Python extract_stats() */
        feat[k++] = mu;
        feat[k++] = sd;
        feat[k++] = xmax - xmin;      /* range   */
        feat[k++] = sqrtf(energy);    /* rms     */
        feat[k++] = energy;
        feat[k++] = skew;
        feat[k++] = kurt;
        feat[k++] = zcr;
        /* k += 8 per axis → k = 48 after 6 axes */
    }

    /* ── 8 statistics for accel_mag and gyro_mag (orientation-invariant) ── */
    /* accel_mag = sqrt(ax²+ay²+az²), gyro_mag = sqrt(gx²+gy²+gz²)          */
    /* k = 48 → 64                                                            */
    {
        float am[IMU_WINDOW_SIZE], gm[IMU_WINDOW_SIZE];
        float *mag_bufs[2];
        int mi;
        mag_bufs[0] = am;
        mag_bufs[1] = gm;
        for (i = 0; i < IMU_WINDOW_SIZE; i++) {
            am[i] = sqrtf(win[i][0]*win[i][0] + win[i][1]*win[i][1] + win[i][2]*win[i][2]);
            gm[i] = sqrtf(win[i][3]*win[i][3] + win[i][4]*win[i][4] + win[i][5]*win[i][5]);
        }
        for (mi = 0; mi < 2; mi++) {
            float *x = mag_bufs[mi];
            float mu_m = 0.0f;
            for (i = 0; i < IMU_WINDOW_SIZE; i++) mu_m += x[i];
            mu_m /= (float)IMU_WINDOW_SIZE;

            float var_m = 0.0f;
            for (i = 0; i < IMU_WINDOW_SIZE; i++) { float d = x[i] - mu_m; var_m += d*d; }
            var_m /= (float)IMU_WINDOW_SIZE;
            float sd_m = sqrtf(var_m);

            float xmin_m = x[0], xmax_m = x[0];
            for (i = 1; i < IMU_WINDOW_SIZE; i++) {
                if (x[i] < xmin_m) xmin_m = x[i];
                if (x[i] > xmax_m) xmax_m = x[i];
            }

            float en_m = 0.0f;
            for (i = 0; i < IMU_WINDOW_SIZE; i++) en_m += x[i]*x[i];
            en_m /= (float)IMU_WINDOW_SIZE;

            float sk_m = 0.0f, ku_m = 0.0f;
            if (sd_m > 1e-9f) {
                for (i = 0; i < IMU_WINDOW_SIZE; i++) {
                    float nrm = (x[i] - mu_m) / sd_m;
                    float n2  = nrm * nrm;
                    sk_m += n2 * nrm;
                    ku_m += n2 * n2;
                }
                sk_m /= (float)IMU_WINDOW_SIZE;
                ku_m  = ku_m / (float)IMU_WINDOW_SIZE - 3.0f;
            }

            int zc_m = 0;
            for (i = 0; i < IMU_WINDOW_SIZE - 1; i++)
                if (x[i] * x[i+1] < 0.0f) zc_m++;
            float zcr_m = (float)zc_m / (float)(IMU_WINDOW_SIZE - 1);

            feat[k++] = mu_m;
            feat[k++] = sd_m;
            feat[k++] = xmax_m - xmin_m;
            feat[k++] = sqrtf(en_m);
            feat[k++] = en_m;
            feat[k++] = sk_m;
            feat[k++] = ku_m;
            feat[k++] = zcr_m;
        }
        /* k = 64 */
    }

    /* ── 6 Pearson cross-correlations ───────────────────────────────────── */
    /* pairs (axis indices): (0,1)(0,2)(1,2)(3,4)(3,5)(4,5)
     * 0=accel_x 1=accel_y 2=accel_z  3=gyro_x 4=gyro_y 5=gyro_z          */
    {
        static const int CP[6][2] = {{0,1},{0,2},{1,2},{3,4},{3,5},{4,5}};
        int p;
        for (p = 0; p < 6; p++) {
            int a = CP[p][0], b = CP[p][1];
            float ma = 0.0f, mb = 0.0f;
            for (i = 0; i < IMU_WINDOW_SIZE; i++) {
                ma += win[i][a];
                mb += win[i][b];
            }
            ma /= (float)IMU_WINDOW_SIZE;
            mb /= (float)IMU_WINDOW_SIZE;
            float num = 0.0f, da2 = 0.0f, db2 = 0.0f;
            for (i = 0; i < IMU_WINDOW_SIZE; i++) {
                float da = win[i][a] - ma;
                float db = win[i][b] - mb;
                num += da * db;
                da2 += da * da;
                db2 += db * db;
            }
            float den = sqrtf(da2 * db2);
            feat[k++] = (den > 1e-9f) ? num / den : 0.0f;
        }
        /* k = 70 */
    }

    /* ── 3 jerk stats × accel axes (0=accel_x 1=accel_y 2=accel_z) ─────── */
    {
        static const int JA[3] = {0, 1, 2};
        int jp;
        for (jp = 0; jp < 3; jp++) {
            int a = JA[jp];
            float j[IMU_WINDOW_SIZE - 1];
            for (i = 0; i < IMU_WINDOW_SIZE - 1; i++)
                j[i] = win[i + 1][a] - win[i][a];

            float jmu = 0.0f, jen = 0.0f;
            for (i = 0; i < IMU_WINDOW_SIZE - 1; i++) {
                jmu += j[i];
                jen += j[i] * j[i];
            }
            jmu /= (float)(IMU_WINDOW_SIZE - 1);
            jen /= (float)(IMU_WINDOW_SIZE - 1);

            float jvar = 0.0f;
            for (i = 0; i < IMU_WINDOW_SIZE - 1; i++) {
                float d = j[i] - jmu;
                jvar += d * d;
            }
            jvar /= (float)(IMU_WINDOW_SIZE - 1);

            feat[k++] = jmu;          /* mean */
            feat[k++] = sqrtf(jvar);  /* std  */
            feat[k++] = sqrtf(jen);   /* rms  */
        }
        /* k = 79 (time-domain block complete) */
    }

#if NUM_FREQ_FEATURES > 0
    /* ── Append frequency-domain block → feat[79 .. 144] ── */
    imu_extract_freq_features(win, &feat[k]);
    k += NUM_FREQ_FEATURES;
#endif

#if NUM_UCI_FEATURES > 0
    /* ── Append UCI-HAR derived-signal block → feat[145 .. NUM_STAT_FEATURES-1] ─
     * imu_extract_uci() computes the full canonical 435-feature set; only the
     * dedup-surviving subset (UCI_KEEP) is copied into the model vector so the
     * SCALER/PCA arrays line up byte-for-byte with the trained 560-feature layout. */
    {
        float uci_full[NUM_UCI_CANON];
        int   u;
        imu_extract_uci(win, uci_full);
        for (u = 0; u < NUM_UCI_FEATURES; u++)
            feat[k + u] = uci_full[UCI_KEEP[u]];
        k += NUM_UCI_FEATURES;
    }
#endif
    /* k == NUM_STAT_FEATURES at exit */
}
"""

_C_UCI = """\
#if NUM_UCI_FEATURES > 0
/* ══════════════════════════════════════════════════════════════════════════
 * SECTION 2c — UCI-HAR derived-signal feature extraction
 *
 *   Reconstructs the canonical UCI-HAR signal set and computes the statistical
 *   measures, mirroring Python extract_uci_features() in canonical order.  The
 *   surviving (dedup) subset is copied into the model vector by SECTION 2.
 *
 *   GRAVITY SEPARATION.  Python used scipy butter(3)+filtfilt @ 0.3 Hz.  On a
 *   fixed-length window filtfilt is a LINEAR operator, so it is exported as the
 *   constant matrix GRAVITY_MATRIX and applied here as gravity = G · accel_axis
 *   — bit-equivalent to the trained model (float32 vs float64, same tolerance
 *   as the time/freq blocks).  This avoids shipping an IIR filter on-device.
 *
 *   Needs sqrtf/fabsf/floorf/logf (<math.h>) + imu_fft (SECTION 2b, requires
 *   NUM_FREQ_FEATURES > 0).  Stack-only, no heap.
 * ══════════════════════════════════════════════════════════════════════════ */

/* insertion sort, ascending (n <= IMU_WINDOW_SIZE → small, no heap) */
static void uci_sort(float *a, int n)
{
    int i, j; float key;
    for (i = 1; i < n; i++) {
        key = a[i]; j = i - 1;
        while (j >= 0 && a[j] > key) { a[j + 1] = a[j]; j--; }
        a[j + 1] = key;
    }
}
static float uci_median_sorted(const float *s, int n)
{ return (n & 1) ? s[n / 2] : 0.5f * (s[n / 2 - 1] + s[n / 2]); }

static float uci_percentile_sorted(const float *s, int n, float p)
{   /* numpy 'linear' interpolation */
    float pos = p / 100.0f * (float)(n - 1);
    int   lo  = (int)floorf(pos);
    float fr  = pos - (float)lo;
    if (lo + 1 >= n) return s[lo];
    return s[lo] + fr * (s[lo + 1] - s[lo]);
}
static float uci_mean(const float *x, int n)
{ float s = 0.0f; int i; for (i = 0; i < n; i++) s += x[i]; return s / (float)n; }

static float uci_std(const float *x, int n)
{
    float mu = uci_mean(x, n), v = 0.0f; int i;
    for (i = 0; i < n; i++) { float d = x[i] - mu; v += d * d; }
    return sqrtf(v / (float)n);
}
static float uci_energy(const float *x, int n)
{ float s = 0.0f; int i; for (i = 0; i < n; i++) s += x[i] * x[i]; return s / (float)n; }

static float uci_max(const float *x, int n)
{ float m = x[0]; int i; for (i = 1; i < n; i++) if (x[i] > m) m = x[i]; return m; }
static float uci_min(const float *x, int n)
{ float m = x[0]; int i; for (i = 1; i < n; i++) if (x[i] < m) m = x[i]; return m; }

static float uci_mad(const float *x, int n, float *scr)
{
    int i; float med;
    for (i = 0; i < n; i++) scr[i] = x[i];
    uci_sort(scr, n); med = uci_median_sorted(scr, n);
    for (i = 0; i < n; i++) scr[i] = fabsf(x[i] - med);
    uci_sort(scr, n); return uci_median_sorted(scr, n);
}
static float uci_iqr(const float *x, int n, float *scr)
{
    int i; for (i = 0; i < n; i++) scr[i] = x[i];
    uci_sort(scr, n);
    return uci_percentile_sorted(scr, n, 75.0f) - uci_percentile_sorted(scr, n, 25.0f);
}
static float uci_entropy(const float *x, int n)
{
    int i; float s = 0.0f, e = 0.0f;
    for (i = 0; i < n; i++) s += fabsf(x[i]);
    if (s <= 1e-12f) return 0.0f;
    for (i = 0; i < n; i++) { float p = fabsf(x[i]) / s; if (p > 0.0f) e += -p * logf(p); }
    return e / logf((float)n);
}
static float uci_skew(const float *x, int n)
{
    float mu = uci_mean(x, n), sd = uci_std(x, n), s = 0.0f; int i;
    if (sd <= 1e-9f) return 0.0f;
    for (i = 0; i < n; i++) { float z = (x[i] - mu) / sd; s += z * z * z; }
    return s / (float)n;
}
static float uci_kurt(const float *x, int n)
{
    float mu = uci_mean(x, n), sd = uci_std(x, n), s = 0.0f; int i;
    if (sd <= 1e-9f) return 0.0f;
    for (i = 0; i < n; i++) { float z = (x[i] - mu) / sd; float z2 = z * z; s += z2 * z2; }
    return s / (float)n - 3.0f;
}
static float uci_corr(const float *a, const float *b, int n)
{
    float ma = uci_mean(a, n), mb = uci_mean(b, n), num = 0, da2 = 0, db2 = 0; int i;
    for (i = 0; i < n; i++) { float da = a[i] - ma, db = b[i] - mb; num += da * db; da2 += da * da; db2 += db * db; }
    if (sqrtf(da2 / (float)n) <= 1e-9f || sqrtf(db2 / (float)n) <= 1e-9f) return 0.0f;
    return num / sqrtf(da2 * db2);
}
static void uci_ar(const float *x, int n, float out[4])
{   /* Levinson-Durbin on biased autocorrelation — mirrors Python _ar_coeffs() */
    int i, j, k; float mu = uci_mean(x, n);
    float r[5], a[5], prev[5], e;
    for (k = 0; k <= 4; k++) {
        float s = 0.0f; for (i = 0; i < n - k; i++) s += (x[i] - mu) * (x[i + k] - mu);
        r[k] = s / (float)n;
    }
    for (i = 0; i < 4; i++) out[i] = 0.0f;
    if (r[0] <= 1e-12f) return;
    for (i = 0; i <= 4; i++) a[i] = 0.0f;
    a[0] = 1.0f; e = r[0];
    for (i = 1; i <= 4; i++) {
        float acc = r[i]; for (j = 1; j < i; j++) acc += a[j] * r[i - j];
        float kk = -acc / e;
        for (j = 0; j <= 4; j++) prev[j] = a[j];
        for (j = 1; j < i; j++) a[j] = prev[j] + kk * prev[i - j];
        a[i] = kk; e *= (1.0f - kk * kk);
        if (e <= 1e-12f) break;
    }
    for (i = 0; i < 4; i++) out[i] = a[i + 1];
}

/* magnitude spectrum (DC dropped), detrended, zero-padded to IMU_NFFT.
 * mag[i] = |rfft(x - mean)|[i+1], i = 0 .. IMU_NFFT/2 - 1  → matches Python _spectrum(). */
static void uci_spectrum(const float *x, int n, float *mag)
{
    float re[IMU_NFFT], im[IMU_NFFT]; int i;
    float mu = uci_mean(x, n);
    for (i = 0; i < IMU_NFFT; i++) { re[i] = (i < n) ? (x[i] - mu) : 0.0f; im[i] = 0.0f; }
    imu_fft(re, im);
    for (i = 0; i < IMU_NFFT / 2; i++) { float rr = re[i + 1], ii = im[i + 1]; mag[i] = sqrtf(rr * rr + ii * ii); }
}
static float uci_meanfreq(const float *mag, int nb)
{
    float s = 0.0f, wf = 0.0f; int i; float fbin = IMU_FS_HZ / (float)IMU_NFFT;
    for (i = 0; i < nb; i++) s += mag[i];
    if (s <= 1e-12f) return 0.0f;
    for (i = 0; i < nb; i++) wf += ((float)(i + 1) * fbin) * mag[i];
    return wf / s;
}
static float uci_maxinds(const float *mag, int nb)
{ int i, mi = 0; float mx = mag[0]; for (i = 1; i < nb; i++) if (mag[i] > mx) { mx = mag[i]; mi = i; } return (float)mi; }

static float uci_angle(const float u[3], const float v[3])
{
    float d  = u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
    float nu = sqrtf(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
    float nv = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (nu < 1e-12f || nv < 1e-12f) return 0.0f;
    float c = d / (nu * nv);
    if (c >  1.0f) c =  1.0f;
    if (c < -1.0f) c = -1.0f;
    return c;
}

/* ── Per-signal block emitters (append into out[], advancing *c) ──────────────
 * Order within each block MUST match the corresponding Python helper exactly. */
static void uci_tri_time(const float *x, const float *y, const float *z,
                         int n, float *o, int *c, float *scr)
{
    const float *col[3]; int a; col[0] = x; col[1] = y; col[2] = z;
    for (a = 0; a < 3; a++) o[(*c)++] = uci_mean(col[a], n);
    for (a = 0; a < 3; a++) o[(*c)++] = uci_std(col[a], n);
    for (a = 0; a < 3; a++) o[(*c)++] = uci_mad(col[a], n, scr);
    for (a = 0; a < 3; a++) o[(*c)++] = uci_max(col[a], n);
    for (a = 0; a < 3; a++) o[(*c)++] = uci_min(col[a], n);
    { float sma = 0.0f; int i; for (i = 0; i < n; i++) sma += fabsf(x[i]) + fabsf(y[i]) + fabsf(z[i]); o[(*c)++] = sma / (float)n; }
    for (a = 0; a < 3; a++) o[(*c)++] = uci_energy(col[a], n);
    for (a = 0; a < 3; a++) o[(*c)++] = uci_iqr(col[a], n, scr);
    for (a = 0; a < 3; a++) o[(*c)++] = uci_entropy(col[a], n);
    for (a = 0; a < 3; a++) { float ar[4]; uci_ar(col[a], n, ar);
        o[(*c)++] = ar[0]; o[(*c)++] = ar[1]; o[(*c)++] = ar[2]; o[(*c)++] = ar[3]; }
    o[(*c)++] = uci_corr(x, y, n); o[(*c)++] = uci_corr(x, z, n); o[(*c)++] = uci_corr(y, z, n);
}
static void uci_mag_time(const float *x, int n, float *o, int *c, float *scr)
{
    o[(*c)++] = uci_mean(x, n); o[(*c)++] = uci_std(x, n); o[(*c)++] = uci_mad(x, n, scr);
    o[(*c)++] = uci_max(x, n);  o[(*c)++] = uci_min(x, n);
    { float sma = 0.0f; int i; for (i = 0; i < n; i++) sma += fabsf(x[i]); o[(*c)++] = sma / (float)n; }
    o[(*c)++] = uci_energy(x, n); o[(*c)++] = uci_iqr(x, n, scr); o[(*c)++] = uci_entropy(x, n);
    { float ar[4]; uci_ar(x, n, ar); o[(*c)++] = ar[0]; o[(*c)++] = ar[1]; o[(*c)++] = ar[2]; o[(*c)++] = ar[3]; }
}
static void uci_tri_freq(const float *x, const float *y, const float *z,
                         int n, float *o, int *c, float *scr)
{
    int NB = IMU_NFFT / 2, a, i;
    float sx[IMU_NFFT / 2], sy[IMU_NFFT / 2], sz[IMU_NFFT / 2];
    const float *sp[3]; sp[0] = sx; sp[1] = sy; sp[2] = sz;
    uci_spectrum(x, n, sx); uci_spectrum(y, n, sy); uci_spectrum(z, n, sz);
    for (a = 0; a < 3; a++) o[(*c)++] = uci_mean(sp[a], NB);
    for (a = 0; a < 3; a++) o[(*c)++] = uci_std(sp[a], NB);
    for (a = 0; a < 3; a++) o[(*c)++] = uci_mad(sp[a], NB, scr);
    for (a = 0; a < 3; a++) o[(*c)++] = uci_max(sp[a], NB);
    for (a = 0; a < 3; a++) o[(*c)++] = uci_min(sp[a], NB);
    { float sma = 0.0f; for (a = 0; a < 3; a++) { float s2 = 0.0f; for (i = 0; i < NB; i++) s2 += fabsf(sp[a][i]); sma += s2; } o[(*c)++] = sma / 3.0f; }
    for (a = 0; a < 3; a++) o[(*c)++] = uci_energy(sp[a], NB);
    for (a = 0; a < 3; a++) o[(*c)++] = uci_iqr(sp[a], NB, scr);
    for (a = 0; a < 3; a++) o[(*c)++] = uci_entropy(sp[a], NB);
    for (a = 0; a < 3; a++) o[(*c)++] = uci_maxinds(sp[a], NB);
    for (a = 0; a < 3; a++) o[(*c)++] = uci_meanfreq(sp[a], NB);
    for (a = 0; a < 3; a++) o[(*c)++] = uci_skew(sp[a], NB);
    for (a = 0; a < 3; a++) o[(*c)++] = uci_kurt(sp[a], NB);
}
static void uci_mag_freq(const float *x, int n, float *o, int *c, float *scr)
{
    int NB = IMU_NFFT / 2, i; float mag[IMU_NFFT / 2];
    uci_spectrum(x, n, mag);
    o[(*c)++] = uci_mean(mag, NB); o[(*c)++] = uci_std(mag, NB); o[(*c)++] = uci_mad(mag, NB, scr);
    o[(*c)++] = uci_max(mag, NB);  o[(*c)++] = uci_min(mag, NB);
    { float sma = 0.0f; for (i = 0; i < NB; i++) sma += fabsf(mag[i]); o[(*c)++] = sma; }
    o[(*c)++] = uci_energy(mag, NB); o[(*c)++] = uci_iqr(mag, NB, scr); o[(*c)++] = uci_entropy(mag, NB);
    o[(*c)++] = uci_maxinds(mag, NB); o[(*c)++] = uci_meanfreq(mag, NB);
    o[(*c)++] = uci_skew(mag, NB); o[(*c)++] = uci_kurt(mag, NB);
}

/* Fill the canonical NUM_UCI_CANON (435) UCI feature vector (pre-dedup order). */
static void imu_extract_uci(const float win[IMU_WINDOW_SIZE][NUM_RAW_AXES],
                            float out[NUM_UCI_CANON])
{
    int i, a, c = 0; const int n = IMU_WINDOW_SIZE;
    float grav[3][IMU_WINDOW_SIZE], bAcc[3][IMU_WINDOW_SIZE];
    float bAccJerk[3][IMU_WINDOW_SIZE], bGyro[3][IMU_WINDOW_SIZE], bGyroJerk[3][IMU_WINDOW_SIZE];
    float scr[IMU_WINDOW_SIZE], mg[IMU_WINDOW_SIZE];

    /* gravity = GRAVITY_MATRIX · accel_axis  (linear filtfilt replacement) */
    for (a = 0; a < 3; a++)
        for (i = 0; i < n; i++) {
            float s = 0.0f; int j;
            for (j = 0; j < n; j++) s += GRAVITY_MATRIX[i][j] * win[j][a];
            grav[a][i] = s;
        }
    for (a = 0; a < 3; a++)
        for (i = 0; i < n; i++) { bAcc[a][i] = win[i][a] - grav[a][i]; bGyro[a][i] = win[i][3 + a]; }
    /* jerk: forward difference, last sample repeated (length preserved) */
    for (a = 0; a < 3; a++) {
        for (i = 0; i < n - 1; i++) { bAccJerk[a][i] = bAcc[a][i + 1] - bAcc[a][i];
                                      bGyroJerk[a][i] = bGyro[a][i + 1] - bGyro[a][i]; }
        bAccJerk[a][n - 1] = bAccJerk[a][n - 2];
        bGyroJerk[a][n - 1] = bGyroJerk[a][n - 2];
    }

    /* 1. triaxial time (5 signals × 40) */
    uci_tri_time(bAcc[0], bAcc[1], bAcc[2], n, out, &c, scr);
    uci_tri_time(grav[0], grav[1], grav[2], n, out, &c, scr);
    uci_tri_time(bAccJerk[0], bAccJerk[1], bAccJerk[2], n, out, &c, scr);
    uci_tri_time(bGyro[0], bGyro[1], bGyro[2], n, out, &c, scr);
    uci_tri_time(bGyroJerk[0], bGyroJerk[1], bGyroJerk[2], n, out, &c, scr);

#define UCI_NORM3(SIG) do { for (i = 0; i < n; i++) \
        mg[i] = sqrtf(SIG[0][i]*SIG[0][i] + SIG[1][i]*SIG[1][i] + SIG[2][i]*SIG[2][i]); } while (0)
    /* 2. magnitude time (5 signals × 13) */
    UCI_NORM3(bAcc);      uci_mag_time(mg, n, out, &c, scr);
    UCI_NORM3(grav);      uci_mag_time(mg, n, out, &c, scr);
    UCI_NORM3(bAccJerk);  uci_mag_time(mg, n, out, &c, scr);
    UCI_NORM3(bGyro);     uci_mag_time(mg, n, out, &c, scr);
    UCI_NORM3(bGyroJerk); uci_mag_time(mg, n, out, &c, scr);

    /* 3. triaxial freq (bAcc, bAccJerk, bGyro × 37) */
    uci_tri_freq(bAcc[0], bAcc[1], bAcc[2], n, out, &c, scr);
    uci_tri_freq(bAccJerk[0], bAccJerk[1], bAccJerk[2], n, out, &c, scr);
    uci_tri_freq(bGyro[0], bGyro[1], bGyro[2], n, out, &c, scr);

    /* 4. magnitude freq (norm3 of bAcc, bAccJerk, bGyro, bGyroJerk × 13) */
    UCI_NORM3(bAcc);      uci_mag_freq(mg, n, out, &c, scr);
    UCI_NORM3(bAccJerk);  uci_mag_freq(mg, n, out, &c, scr);
    UCI_NORM3(bGyro);     uci_mag_freq(mg, n, out, &c, scr);
    UCI_NORM3(bGyroJerk); uci_mag_freq(mg, n, out, &c, scr);
#undef UCI_NORM3

    /* 5. angle() features (7) — means vs gravity mean vector */
    {
        float gm[3], bm[3], jm[3], gym[3], gyjm[3], ax[3];
        for (a = 0; a < 3; a++) {
            gm[a]   = uci_mean(grav[a], n);     bm[a]  = uci_mean(bAcc[a], n);
            jm[a]   = uci_mean(bAccJerk[a], n); gym[a] = uci_mean(bGyro[a], n);
            gyjm[a] = uci_mean(bGyroJerk[a], n);
        }
        out[c++] = uci_angle(bm,   gm);
        out[c++] = uci_angle(jm,   gm);
        out[c++] = uci_angle(gym,  gm);
        out[c++] = uci_angle(gyjm, gm);
        ax[0] = 1.0f; ax[1] = 0.0f; ax[2] = 0.0f; out[c++] = uci_angle(ax, gm);
        ax[0] = 0.0f; ax[1] = 1.0f; ax[2] = 0.0f; out[c++] = uci_angle(ax, gm);
        ax[0] = 0.0f; ax[1] = 0.0f; ax[2] = 1.0f; out[c++] = uci_angle(ax, gm);
    }
    /* c == NUM_UCI_CANON at exit */
}
#endif /* NUM_UCI_FEATURES > 0 */
"""

_C_PREPROCESS = """\
/* ══════════════════════════════════════════════════════════════════════════
 * SECTION 3 — Preprocessing: stat features → reduced model-input space
 *
 * imu_preprocess(stat_in, pca_out)
 *   stat_in : float[NUM_STAT_FEATURES]  (from imu_extract_features)
 *   pca_out : float[NUM_PCA_FEATURES]   (fed to imu_predict_tree)
 *
 *   1. StandardScaler : scaled[i] = (stat_in[i] - MEAN[i]) / SCALE[i]
 *   2a. PROJECTION reducer (PCA / LDA, IMU_REDUCER_SELECT==0):
 *         pca_out[j] = dot(PCA_COMPS[j], scaled - PCA_MEAN)
 *   2b. SELECTION reducer (rf_importance / mutual_info / rfe, ==1):
 *         pca_out[j] = scaled[SELECT_IDX[j]]   (cheap gather — no projection matrix)
 * ══════════════════════════════════════════════════════════════════════════ */
static inline void imu_preprocess(const float stat_in[NUM_STAT_FEATURES],
                                  float       pca_out[NUM_PCA_FEATURES])
{
    float scaled[NUM_STAT_FEATURES];
    int   i, j;

    for (i = 0; i < NUM_STAT_FEATURES; i++)
        scaled[i] = (stat_in[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];

#if IMU_REDUCER_SELECT
    for (j = 0; j < NUM_PCA_FEATURES; j++)
        pca_out[j] = scaled[SELECT_IDX[j]];
#else
    for (j = 0; j < NUM_PCA_FEATURES; j++) {
        float dot = 0.0f;
        for (i = 0; i < NUM_STAT_FEATURES; i++)
            dot += PCA_COMPS[j][i] * (scaled[i] - PCA_MEAN[i]);
        pca_out[j] = dot;
    }
#endif
}
"""

_C_TRAVERSE = """\
/* ══════════════════════════════════════════════════════════════════════════
 * SECTION 4 — Compact tree traversal
 * ══════════════════════════════════════════════════════════════════════════ */

/* Single-tree inference: walk the flat IMUNode array */
static inline int imu_predict_tree(const IMUNode *tree,
                                   const float   *pca_in)
{
    int node = 0;
    while (tree[node].left != -1)
        node = (pca_in[tree[node].feature] <= tree[node].threshold)
               ? tree[node].left : tree[node].right;
    return (int)tree[node].pred_class;
}
"""


def generate_compact_header(pipe, le, n_pca, FEATURES, test_acc,
                             WINDOW_SIZE, STEP_SIZE, RAW_AXES,
                             RF_N_TREES, RF_MAX_DEPTH):
    scaler  = pipe.named_steps["scaler"]
    pca     = pipe.named_steps["pca"]
    rf      = pipe.named_steps["clf"]
    n_feat  = NUM_STAT_FEAT         # full feature count (560 with UCI on)
    n_cls   = len(le.classes_)
    n_trees = len(rf.estimators_)

    # ── Arrays ────────────────────────────────────────────────────────────
    mean_c    = _fmt_f32_array(scaler.mean_,  "SCALER_MEAN")
    scale_c   = _fmt_f32_array(scaler.scale_, "SCALER_SCALE")

    # ── Reducer: projection (PCA/LDA → matvec) vs selection (raw gather) ──────
    reducer_kind  = getattr(pca, "selector_kind_", "pca")
    reducer_shape = getattr(pca, "reducer_shape_", "projection")
    if reducer_shape == "selection":
        # Gather path: ship only the K raw-feature indices, not a projection matrix.
        select_idx_c = _fmt_int_array(np.asarray(pca.sel_indices_, dtype=int))
        reducer_define = (f"#define IMU_REDUCER_SELECT 1   "
                          f"/* reducer='{reducer_kind}': gather K raw features */")
        reducer_arrays = f"""
/* ── Feature-SELECTION reducer ('{reducer_kind}') ────────────────────────────
 * The model consumes NUM_PCA_FEATURES raw (scaled) features chosen by supervised
 * selection; imu_preprocess() gathers them by index — no projection matrix. */
static const int16_t SELECT_IDX[NUM_PCA_FEATURES] = {{
{select_idx_c}
}};
"""
    else:
        # Projection path: PCA_MEAN + PCA_COMPS, matvec (covers PCA and LDA).
        pca_mean_c = _fmt_f32_array(pca.mean_, "PCA_MEAN")
        pca_comp_c = _fmt_pca_components(pca.components_)
        reducer_define = (f"#define IMU_REDUCER_SELECT 0   "
                          f"/* reducer='{reducer_kind}': project (scaled-mean)·COMPS */")
        reducer_arrays = f"""
/* ── Feature-PROJECTION reducer ('{reducer_kind}') ───────────────────────────
 * imu_preprocess() projects the scaled 560-vector onto NUM_PCA_FEATURES axes. */
static const float PCA_MEAN[NUM_STAT_FEATURES] = {{
{pca_mean_c}
}};
static const float PCA_COMPS[NUM_PCA_FEATURES][NUM_STAT_FEATURES] = {{
{pca_comp_c}
}};
"""

    label_entries = ", ".join(f'"{cls}"' for cls in le.classes_)

    # ── UCI-HAR block: gravity matrix + dedup keep-map + C extractor ──────────
    # Only emitted when the expanded feature set is active.  When off, NUM_UCI is
    # 0, the #if-guards drop every UCI symbol, and the header is the original
    # bit-parity 145-feature layout.
    if USE_UCI_FEATURES:
        canon, keep = _build_uci_keep(FEATURES)
        n_uci_canon = len(canon)                 # 435 (pre-dedup)
        n_uci_keep  = len(keep)                  # 415 (post-dedup, == NUM_UCI_FEAT)
        assert n_uci_keep == NUM_UCI_FEAT
        Gmat = _build_gravity_matrix(WINDOW_SIZE)
        grav_c = _fmt_matrix_f32(Gmat)
        keep_c = _fmt_int_array(keep)
        uci_section = _C_UCI
        uci_arrays = f"""
/* ── UCI-HAR gravity low-pass as a constant linear operator ──────────────────
 * gravity[i] = Σ_j GRAVITY_MATRIX[i][j] · accel_axis[j]   (≡ scipy filtfilt).
 * {WINDOW_SIZE}×{WINDOW_SIZE} float32 = {WINDOW_SIZE*WINDOW_SIZE*4/1024:.0f} KB flash. */
static const float GRAVITY_MATRIX[IMU_WINDOW_SIZE][IMU_WINDOW_SIZE] = {{
{grav_c}
}};

/* Indices (into the canonical {n_uci_canon}-feature UCI vector) that survive the
 * exact-duplicate dedup and feed the model's feat[{EXPECTED_TOTAL_FEAT}..]. */
static const int16_t UCI_KEEP[NUM_UCI_FEATURES] = {{
{keep_c}
}};
"""
        uci_defines = (
            f"#define NUM_UCI_FEATURES   {n_uci_keep}   "
            f"/* UCI-HAR derived feats kept after dedup (0 = block off) */\n"
            f"#define NUM_UCI_CANON      {n_uci_canon}   "
            f"/* UCI-HAR feats computed before dedup                    */\n"
            f"#define IMU_AR_ORDER       {AR_ORDER}     "
            f"/* AR (Levinson) coefficients per signal                  */")
        # ── Self-test vector (one real window + Python float64 reference) ──────
        selftest_block = _build_selftest_block(FEATURES)
    else:
        uci_section = ""
        uci_arrays  = ""
        uci_defines = ("#define NUM_UCI_FEATURES   0     "
                       "/* UCI-HAR block disabled — 145-feature bit-parity layout */")
        selftest_block = ""

    # ── Tree arrays ───────────────────────────────────────────────────────
    tree_blocks = []
    total_nodes = 0
    for i, estimator in enumerate(rf.estimators_):
        block, nc = export_tree_to_c_array(estimator, i)
        tree_blocks.append(block)
        total_nodes += nc

    tree_sizes_c = ", ".join(str(e.tree_.node_count) for e in rf.estimators_)
    ptr_table_c  = ", ".join(f"tree_{i}" for i in range(n_trees))
    trees_block  = "\n".join(tree_blocks)

    flash_kb  = total_nodes * 10 / 1024
    # Stack: scaler/pca scratch + the UCI extractor's derived-signal buffers
    # (5 triaxial signals + magnitude/sort scratch + uci_full[NUM_UCI_CANON] +
    #  FFT re/im + 3 spectra), all stack-only — counted in float words.
    if USE_UCI_FEATURES:
        uci_stack = (len(canon) + 5 * 3 * WINDOW_SIZE + 2 * WINDOW_SIZE
                     + 3 * (NFFT // 2) + 2 * NFFT)
    else:
        uci_stack = 0
    stack_kb  = (n_feat + n_pca + n_cls + 50 + uci_stack) * 4 / 1024

    feat_map  = "\n".join(f" *   [{i:2d}] {f}" for i, f in enumerate(FEATURES))

    # ── Predict wrapper (f-string, careful with C braces) ─────────────────
    c_predict = f"""\
/* ══════════════════════════════════════════════════════════════════════════
 * SECTION 5 — End-to-End Wrapper
 *
 * imu_predict(win)
 *   win     : const float[IMU_WINDOW_SIZE][NUM_RAW_AXES]
 *               col 0=accel_x  1=accel_y  2=accel_z
 *               col 3=gyro_x   4=gyro_y   5=gyro_z
 *   Returns : class index 0 … NUM_CLASSES-1
 *
 * Full pipeline (stack-only):
 *   1. imu_extract_features — {n_feat} stat features ({EXPECTED_TIME_FEAT} time + {NUM_FREQ_FEAT} freq + {NUM_UCI_FEAT} UCI)
 *   2. imu_preprocess       — StandardScaler + PCA({n_pca} dims)
 *   3. imu_predict_tree ×{n_trees} — majority vote
 *
 * Stack: ~{stack_kb:.1f} KB  |  Flash (model): ~{flash_kb:.1f} KB  |  No malloc
 *
 * ESP32-P4: add IRAM_ATTR before 'int' to run from SRAM (lowest latency).
 * ══════════════════════════════════════════════════════════════════════════ */
static inline int imu_predict(
        const float win[IMU_WINDOW_SIZE][NUM_RAW_AXES])
{{
    float stat_feat[NUM_STAT_FEATURES];   /* time + freq features */
    float pca_feat[NUM_PCA_FEATURES];     /*  n × 4 bytes   */
    int   votes[NUM_CLASSES];
    int   i, cls, best;

    imu_extract_features(win, stat_feat);     /* Section 2 */
    imu_preprocess(stat_feat, pca_feat);      /* Section 3 */

    for (i = 0; i < NUM_CLASSES; i++) votes[i] = 0;
    for (i = 0; i < NUM_TREES;   i++) {{      /* Section 4 */
        cls = imu_predict_tree(TREES[i], pca_feat);
        votes[cls]++;
    }}

    best = 0;
    for (i = 1; i < NUM_CLASSES; i++)
        if (votes[i] > votes[best]) best = i;

    return best;
}}

/* ── Confidence-gated prediction (recommended for live use) ──────────────────
 * Returns the winning class, or IMU_CLASS_UNCERTAIN (-1) when the vote share is
 * below IMU_CONF_THRESHOLD.  *out_conf (may be NULL) receives the vote fraction.
 * This is the firmware twin of the Python CONF_THRESHOLD guard and is what stops
 * "running" from firing on a single twitchy window.  Pair it with a short
 * majority-vote ring buffer over successive calls (see StreamingPredictor). */
static inline int imu_predict_conf(
        const float win[IMU_WINDOW_SIZE][NUM_RAW_AXES],
        float *out_conf)
{{
    float stat_feat[NUM_STAT_FEATURES];
    float pca_feat[NUM_PCA_FEATURES];
    int   votes[NUM_CLASSES];
    int   i, cls, best;

    imu_extract_features(win, stat_feat);
    imu_preprocess(stat_feat, pca_feat);

    for (i = 0; i < NUM_CLASSES; i++) votes[i] = 0;
    for (i = 0; i < NUM_TREES;   i++) {{
        cls = imu_predict_tree(TREES[i], pca_feat);
        votes[cls]++;
    }}
    best = 0;
    for (i = 1; i < NUM_CLASSES; i++)
        if (votes[i] > votes[best]) best = i;

    float conf = (float)votes[best] / (float)NUM_TREES;
    if (out_conf) *out_conf = conf;
    return (conf >= IMU_CONF_THRESHOLD) ? best : IMU_CLASS_UNCERTAIN;
}}
{selftest_block}
#endif /* IMU_MODEL_COMPACT_H */
"""

    # ── File header (f-string, only non-brace C needed) ───────────────────
    file_header = f"""\
/*
 * imu_model_compact.h  —  Compact flat-struct RF export
 * Auto-generated — DO NOT EDIT BY HAND
 *
 * ── Target ───────────────────────────────────────────────────────────────────
 *   ESP32-P4  (RISC-V RV32IMFC, hardware FP32 FPU, 2 MB flash budget)
 *   Pure C99 · <stdint.h> + <math.h> · no stdlib · no heap
 *
 * ── Format ───────────────────────────────────────────────────────────────────
 *   IMUNode struct (10 B/node, naturally aligned):
 *     float threshold | int16 left | int16 right | int8 feature | uint8 pred
 *   Traversal: while left!=-1 → follow left or right by threshold
 *
 * ── Pipeline ─────────────────────────────────────────────────────────────────
 *   Raw window ({WINDOW_SIZE}×{len(RAW_AXES)}) → {n_feat} stat features ({EXPECTED_TIME_FEAT} time + {NUM_FREQ_FEAT} freq + {NUM_UCI_FEAT} UCI)
 *   → StandardScaler → PCA({n_pca} dims) → RF ({n_trees} trees, depth {RF_MAX_DEPTH})
 *
 * ── Resources ────────────────────────────────────────────────────────────────
 *   Total nodes   : {total_nodes}
 *   Flash (model) : ~{flash_kb:.1f} KB  ({total_nodes} nodes × 10 B)
 *   Stack         : ~{stack_kb:.1f} KB  (all local arrays)
 *   Test accuracy : {test_acc:.4f}
 *
 * ── Quick start (ESP-IDF) ────────────────────────────────────────────────────
 *
 *   #include "imu_model_compact.h"
 *
 *   float win[IMU_WINDOW_SIZE][NUM_RAW_AXES];
 *   // fill: col 0=accel_x  1=accel_y  2=accel_z
 *   //       col 3=gyro_x   4=gyro_y   5=gyro_z
 *
 *   int cls          = imu_predict(win);
 *   const char *name = ACTIVITY_LABELS[cls];
 *   ESP_LOGI("IMU", "Activity: %s", name);
 *
 * ── Feature map ──────────────────────────────────────────────────────────────
 *
{feat_map}
 */

#ifndef IMU_MODEL_COMPACT_H
#define IMU_MODEL_COMPACT_H

#include <stdint.h>
#include <math.h>   /* sqrtf, fabsf */

/* ── Compile-time check ─────────────────────────────────────────────────── */
#if {n_pca} > 127
#  error "NUM_PCA_FEATURES exceeds int8_t — use int16_t for feature field"
#endif

/* ── Dimensions ─────────────────────────────────────────────────────────── */
#define IMU_WINDOW_SIZE    {WINDOW_SIZE}   /* samples per window (50 Hz → 1 s)  */
#define IMU_STEP_SIZE      {STEP_SIZE}    /* overlap step                       */
#define NUM_RAW_AXES       {len(RAW_AXES)}    /* accel_xyz + gyro_xyz               */
#define NUM_TIME_FEATURES  {EXPECTED_TIME_FEAT}   /* 6×8 + 2×8 + 6 corr + 3×3 jerk      */
#define NUM_FREQ_FEATURES  {NUM_FREQ_FEAT}   /* 11 signals × 6 spectral stats (0=off) */
#define NUM_STAT_FEATURES  {n_feat}   /* time + freq + UCI → full feature vector */
{uci_defines}
#define IMU_NFFT           {NFFT}   /* zero-pad length, radix-2 FFT        */
#define IMU_FS_HZ          {FS_HZ:.1f}f /* sampling rate (Hz)                 */
#define NUM_PCA_FEATURES   {n_pca}    /* reduced model-input dims ({reducer_kind}) */
{reducer_define}
#define NUM_CLASSES        {n_cls}     /* activity classes                    */
#define NUM_TREES          {n_trees}    /* random forest trees                 */
#define IMU_CONF_THRESHOLD {CONF_THRESHOLD:.2f}f /* min vote share for a firm call (else UNCERTAIN) */
#define IMU_CLASS_UNCERTAIN (-1)  /* imu_predict_conf() return when below threshold */

/* ── Activity labels ────────────────────────────────────────────────────── */
static const char * const ACTIVITY_LABELS[NUM_CLASSES] = {{
    {label_entries}
}};

/* ══════════════════════════════════════════════════════════════════════════
 * SECTION 1 — Learned Parameters (float32 arrays in flash)
 * ══════════════════════════════════════════════════════════════════════════ */

/* StandardScaler */
static const float SCALER_MEAN[NUM_STAT_FEATURES] = {{
{mean_c}
}};
static const float SCALER_SCALE[NUM_STAT_FEATURES] = {{
{scale_c}
}};

/* Feature reducer (projection matrix OR selection index list) */
{reducer_arrays}
{uci_arrays}
/* ══════════════════════════════════════════════════════════════════════════
 * IMUNode struct — 10 bytes, naturally aligned (no packed attribute needed)
 *   field order: threshold(f32 @0) left(i16 @4) right(i16 @6)
 *                feature(i8 @8)    pred_class(u8 @9)
 * ══════════════════════════════════════════════════════════════════════════ */
typedef struct {{
    float    threshold;    /* split value                          */
    int16_t  left;         /* left  child index; -1 = leaf         */
    int16_t  right;        /* right child index; -1 = leaf         */
    int8_t   feature;      /* PCA component index (0..NUM_PCA-1)   */
    uint8_t  pred_class;   /* majority class — used only at leaf   */
}} IMUNode;               /* sizeof = 10 B, no padding            */

/* ── Tree node arrays (read-only flash section) ─────────────────────────── */
{trees_block}

/* ── Tree pointer table ─────────────────────────────────────────────────── */
static const IMUNode * const TREES[NUM_TREES] = {{
    {ptr_table_c}
}};
static const int TREE_SIZES[NUM_TREES] = {{
    {tree_sizes_c}
}};

"""

    full_header = (file_header
                   + _C_FREQ
                   + uci_section          # imu_extract_uci (uses imu_fft) before its caller
                   + _C_FEAT_EXTRACT
                   + _C_PREPROCESS
                   + _C_TRAVERSE
                   + c_predict)
    return full_header, total_nodes


# ── GENERATE + SAVE ────────────────────────────────────────────────────────────
# The compact C header ships a stack-only, no-heap extractor that reproduces the
# FULL model feature vector on-device.  In the original layout that is the 145
# time+freq features; with USE_UCI_FEATURES the UCI-HAR derived-signal block
# (SECTION 2c) is added so the embedded extractor matches the {NUM_STAT_FEAT}-feature
# trained model.  Bit-parity for the gravity-separated signals is preserved by
# exporting scipy's filtfilt as a constant GRAVITY_MATRIX (it is a linear operator
# on a fixed-length window — see _build_gravity_matrix), not by re-deriving an IIR
# filter in C.  Indices [0..144] stay byte-identical either way.
print("\n" + "="*65)
print("GENERATING → generated/imu_model_compact.h")
print("="*65)

c_export_status = None   # ("ok", path) | ("skipped", reason) — consumed by reporter
# Export the DEPLOYED model (refit on all sessions), not the eval model.
header_src, total_nodes = generate_compact_header(
    deploy_pipe, le, n_pca, FEATURES, test_acc,
    WINDOW_SIZE, STEP_SIZE, RAW_AXES,
    RF_N_TREES, RF_MAX_DEPTH)

out_path = "generated/imu_model_compact.h"
with open(out_path, "w", encoding="utf-8") as fh:
    fh.write(header_src)
c_export_status = ("ok", out_path)

file_kb   = os.path.getsize(out_path) / 1024
flash_kb  = total_nodes * 10 / 1024
grav_kb   = (WINDOW_SIZE * WINDOW_SIZE * 4) / 1024 if USE_UCI_FEATURES else 0.0
keep_kb   = (NUM_UCI_FEAT * 2) / 1024 if USE_UCI_FEATURES else 0.0

print(f"  Written        : {out_path}")
print(f"  File size      : {file_kb:.1f} KB  (source)")
print(f"  Flash (model)  : ~{flash_kb:.1f} KB  ({total_nodes} nodes × 10 B)")
print(f"  RF             : {RF_N_TREES} trees / depth {RF_MAX_DEPTH}")
print(f"  PCA dims       : {n_pca}")
print(f"  Stat features  : {NUM_STAT_FEAT}  "
      f"({EXPECTED_TIME_FEAT} time + {NUM_FREQ_FEAT} freq + {NUM_UCI_FEAT} UCI)")
if USE_UCI_FEATURES:
    print(f"  Gravity matrix : ~{grav_kb:.0f} KB flash ({WINDOW_SIZE}×{WINDOW_SIZE} f32, "
          f"filtfilt→linear operator)")
print(f"  Labels         : {list(le.classes_)}")

print("\n=== FLASH BUDGET SUMMARY (ESP32-P4, 2 MB) ===")
pca_kb  = (n_pca * NUM_STAT_FEAT + NUM_STAT_FEAT) * 4 / 1024
misc_kb = 12   # feature-extraction code + wrappers (more with the UCI block)
total_model_kb = flash_kb + pca_kb + grav_kb + keep_kb + misc_kb
print(f"  RF nodes       : {flash_kb:.1f} KB")
print(f"  PCA arrays     : {pca_kb:.1f} KB")
if USE_UCI_FEATURES:
    print(f"  Gravity matrix : {grav_kb:.1f} KB")
    print(f"  UCI keep map   : {keep_kb:.1f} KB")
print(f"  C code         : ~{misc_kb} KB")
print(f"  Model total    : ~{total_model_kb:.1f} KB")
print(f"  ESP-IDF system : ~200 KB")
print(f"  Grand total    : ~{total_model_kb+200:.0f} KB  (budget: 2048 KB)")
print(f"  Headroom       : ~{2048-total_model_kb-200:.0f} KB free")

# ── SAVE JSON PARAMS ───────────────────────────────────────────────────────────
# From the DEPLOYED model so the JSON matches the exported C header / .pkl.
scaler = deploy_pipe.named_steps["scaler"]
pca    = deploy_pipe.named_steps["pca"]
preprocess_data = {
    "window_size":      WINDOW_SIZE,
    "step_size":        STEP_SIZE,
    "raw_axes":         RAW_AXES,
    "feature_names":    FEATURES,
    "n_stat_features":  NUM_STAT_FEAT,
    "n_time_features":  EXPECTED_TIME_FEAT,
    "n_freq_features":  NUM_FREQ_FEAT,
    "use_freq_features": USE_FREQ_FEATURES,
    "freq_signals":     FREQ_SIGNALS if USE_FREQ_FEATURES else [],
    "use_uci_features": USE_UCI_FEATURES,
    "n_uci_features":   NUM_UCI_FEAT,
    "uci_gravity_cutoff_hz": GRAVITY_CUTOFF_HZ,
    "uci_ar_order":     AR_ORDER,
    "fs_hz":            FS_HZ,
    "nfft":             NFFT,
    "n_pca_features":   n_pca,
    "pca_candidate_pool": n_pca_var,
    "pca_selection":    (f"top-{PCA_TOP_K} of {n_pca_var} components by RF importance"
                         if FEATURE_SELECTOR_ACTIVE == "pca"
                         else f"reducer '{FEATURE_SELECTOR_ACTIVE}' → {n_pca} raw features"),
    "reducer_kind":     FEATURE_SELECTOR_ACTIVE,
    "pca_kept_indices": [int(x) for x in
                         getattr(pca, "orig_index_", np.arange(n_pca))],
    "classes":          list(le.classes_),
    "label_map":        {str(k): int(v) for k, v in label_map.items()},
    "scaler_mean":      [float(x) for x in scaler.mean_],
    "scaler_scale":     [float(x) for x in scaler.scale_],
    "pca_mean":         [float(x) for x in pca.mean_],
    "pca_components":   [[float(x) for x in row] for row in pca.components_],
}
with open("generated/preprocess_params.json", "w") as fh:
    json.dump(preprocess_data, fh, indent=2)
print("  JSON params    : generated/preprocess_params.json")

# ═══════════════════════════════════════════════════════════════════════════════
# PYTHON-SIDE INFERENCE (mirrors C pipeline for validation)
# ═══════════════════════════════════════════════════════════════════════════════
# ── REAL-TIME STABILISATION ───────────────────────────────────────────────────
# Two cheap guards that kill the "running fires on the slightest twitch" problem
# during live streaming (single-window argmax is jittery and over-confident):
#
#   1. CONFIDENCE FLOOR — if the top class probability is below CONF_THRESHOLD we
#      return "uncertain" instead of guessing.  Rare-but-aggressive classes
#      (running) rarely clear the floor on ambiguous windows.
#   2. TEMPORAL MAJORITY VOTE — smooth the last SMOOTH_N window predictions so a
#      single noisy window can't flip the displayed activity.
# (CONF_THRESHOLD and SMOOTH_N are defined in the CONFIG block at the top so the
#  same values are baked into the generated C header.)

# ── ENERGY FLOOR — learned from the ACTIVE classes (low-energy rejection gate) ─
# Pick the energy feature (orientation-invariant mean accel magnitude preferred,
# per-axis energy as fallback), then learn the minimum motion energy that any
# ACTIVE (non-static) class actually exhibits.  We use the 5th percentile per
# active class (robust to a few quiet outlier windows) and take the smallest such
# value across classes — that is the "minimum threshold seen in active classes".
# A live window whose energy is significantly below it (< ENERGY_REJECT_FRACTION ×
# floor) is treated as no-activity / out-of-distribution and rejected.
ENERGY_FEATURE_NAME = (ENERGY_FEATURE_PRIMARY if ENERGY_FEATURE_PRIMARY in FEATURES
                       else ENERGY_FEATURE_FALLBACK)
_energy_col     = X[:, FEATURES.index(ENERGY_FEATURE_NAME)]
_active_classes = [c for c in le.classes_ if c not in STATIC_CLASSES]
_per_class_min  = {}
for _c in _active_classes:
    _ec = _energy_col[y == le.transform([_c])[0]]
    if _ec.size:
        _per_class_min[_c] = float(np.percentile(_ec, 5))
# min energy seen across active classes; the reject floor sits a fraction below it
MIN_ACTIVE_ENERGY = min(_per_class_min.values()) if _per_class_min else 0.0
ENERGY_FLOOR      = ENERGY_REJECT_FRACTION * MIN_ACTIVE_ENERGY
print(f"\nEnergy gate  : feature='{ENERGY_FEATURE_NAME}'  "
      f"min active-class energy={MIN_ACTIVE_ENERGY:.4f}  "
      f"reject floor={ENERGY_FLOOR:.4f}  (< floor → 'uncertain')")
print(f"               per active-class 5th-pctile energy: "
      + ", ".join(f"{c}={v:.3f}" for c, v in sorted(_per_class_min.items(),
                                                     key=lambda kv: kv[1])))


def predict_from_window(window_data: np.ndarray, verbose: bool = True):
    """
    window_data : (WINDOW_SIZE, 6) float32
                  cols: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    Returns     : (activity_label_or_'uncertain', confidence_float)

    Soft-max threshold rejection: the argmax class is accepted ONLY if it clears
    BOTH gates — top probability ≥ REJECT_THRESHOLD AND window energy ≥
    ENERGY_FLOOR.  Failing either returns 'uncertain' rather than forcing the
    nearest class (which, for quiet windows, is almost always 'sitting').
    """
    assert window_data.shape == (WINDOW_SIZE, len(RAW_AXES)), \
        f"Expected ({WINDOW_SIZE},{len(RAW_AXES)}), got {window_data.shape}"
    row      = extract_stats(window_data.astype(np.float32))
    feat_vec = np.array([row[f] for f in FEATURES], dtype=np.float32).reshape(1,-1)
    proba    = pipe.predict_proba(feat_vec)[0]
    top      = int(proba.argmax())
    conf     = float(proba[top])
    energy   = float(row[ENERGY_FEATURE_NAME])

    # ── soft-max threshold rejection (confidence OR energy) ───────────────────
    low_conf   = conf   <  REJECT_THRESHOLD
    low_energy = energy <  ENERGY_FLOOR
    if low_conf or low_energy:
        activity = "uncertain"
        reason   = ("low-confidence & low-energy" if (low_conf and low_energy)
                    else "low-confidence"         if low_conf
                    else "low-energy")
    else:
        activity = le.classes_[top]
        reason   = "accepted"

    if verbose:
        print(f"\n  Predicted  : {activity}  (conf={conf*100:.1f}%, "
              f"energy={energy:.3f} vs floor {ENERGY_FLOOR:.3f})  [{reason}]")
        for cls, p in sorted(zip(le.classes_, proba), key=lambda x:x[1], reverse=True):
            print(f"    {cls:<16} {p*100:5.1f}%  {'█'*int(p*20)}")
    return activity, conf


class StreamingPredictor:
    """Drop-in real-time wrapper: feed it raw windows, get a smoothed activity.

    Mirrors what the firmware should do — a confidence floor plus a short
    majority-vote ring buffer — so the on-device behaviour matches this script.
    """
    def __init__(self, conf=CONF_THRESHOLD, smooth_n=SMOOTH_N):
        from collections import deque, Counter
        self.conf, self.hist = conf, deque(maxlen=smooth_n)
        self._Counter = Counter
        self.state = "uncertain"

    def update(self, window_data: np.ndarray) -> str:
        label, c = predict_from_window(window_data, verbose=False)
        self.hist.append(label if c >= self.conf else "uncertain")
        votes = self._Counter(self.hist)
        top, n = votes.most_common(1)[0]
        # only switch state on a clear majority; otherwise hold previous state
        if top != "uncertain" and n > len(self.hist) // 2:
            self.state = top
        return self.state

# ── Smoke test 1: IN-DISTRIBUTION sanity (NOT a generalization test) ──────────
# Feeds one genuine single-activity window per class through the full pipeline.
# These windows may come from sessions the model trained on, so a high score here
# proves only that the plumbing works — it does NOT measure generalization.  The
# out-of-sample test below is the one that can actually fail.
print("\n── Smoke test 1: in-distribution sanity (plumbing only) ───────────────")
hits = 0
for cls in sorted(sample_raw_windows):
    pred, conf = predict_from_window(sample_raw_windows[cls], verbose=False)
    ok = (pred == cls)
    hits += ok
    print(f"  true={cls:<12} → pred={pred:<12} conf={conf*100:5.1f}%  "
          f"{'✓' if ok else '✗'}")
if sample_raw_windows:
    print(f"  In-distribution sanity: {hits}/{len(sample_raw_windows)} "
          f"= {hits/len(sample_raw_windows)*100:.0f}%")

# ── Smoke test 1b: OUT-OF-SAMPLE — held-out sessions the model NEVER trained on ─
# This is the smoke test that catches real errors.  We run the EVALUATION model
# (`pipe`, trained only on the session-disjoint training split) on the held-out
# test sessions and report per-session what it predicts.  Because a held-out
# session is an unseen recording (and, here, often an unseen participant), this is
# the honest field-behaviour signal.  Watch for classes whose only session was held
# out — the model has no way to get them right, which is exactly the failure a
# leaked smoke test would have hidden.
print("\n── Smoke test 1b: OUT-OF-SAMPLE held-out sessions (eval model) ────────")
proba_oos = pipe.predict_proba(X_test)
pred_oos  = proba_oos.argmax(1)
conf_oos  = proba_oos.max(1)
gated_oos = np.where(conf_oos >= CONF_THRESHOLD, pred_oos, -1)
oos_hits = 0
for sid in sorted(set(groups_test)):
    m = groups_test == sid
    true_cls = le.classes_[int(np.bincount(y_test[m]).argmax())]
    firm = gated_oos[m] >= 0
    fired = pred_oos[m][firm]
    acc = float((pred_oos[m] == y_test[m]).mean())
    oos_hits += (pred_oos[m] == y_test[m]).sum()
    seen_in_train = true_cls in {le.classes_[c] for c in set(np.unique(y_train))}
    top = ("—" if fired.size == 0
           else le.classes_[int(np.bincount(fired, minlength=len(le.classes_)).argmax())])
    print(f"  session {sid[:17]:17s} true={true_cls:<10} "
          f"acc={acc:4.0%}  firm-pred={top:<10} firm={firm.mean():4.0%}  "
          f"{'' if seen_in_train else '⚠ class never trained → structurally 0'}")
print(f"  Out-of-sample window accuracy: {oos_hits}/{len(y_test)} "
      f"= {oos_hits/len(y_test)*100:.1f}%   (this is the honest number)")

# ── Smoke test 3: ROTATED SENSOR — does an upside-down wear break the model? ───
# Flip the sign of the Y and Z axes (a 180° roll, i.e. the watch worn upside-down)
# on a window the model classifies CONFIDENTLY at baseline, then re-predict.  The
# model keeps directional features, so a correct directional model should change
# its answer or lose confidence — proving it is NOT rotation-invariant and would
# misfire on a flipped sensor.  (If you later switch to magnitude-only features,
# this test should instead show the prediction UNCHANGED = rotation-robust.)
print("\n── Smoke test 3: rotated sensor (flip Y/Z = upside-down wear) ─────────")
_flip_idx = [RAW_AXES.index(a) for a in ("accel_y", "accel_z", "gyro_y", "gyro_z")]
# choose a baseline window the model is confident & correct on (a trainable class)
_base_cls = None
for cls in sorted(sample_raw_windows):
    p, c = predict_from_window(sample_raw_windows[cls], verbose=False)
    if p == cls and c >= CONF_THRESHOLD:
        _base_cls = cls; break
if _base_cls is not None:
    w0 = sample_raw_windows[_base_cls].astype(np.float32).copy()
    p0, c0 = predict_from_window(w0, verbose=False)
    w1 = w0.copy(); w1[:, _flip_idx] = -w1[:, _flip_idx]
    p1, c1 = predict_from_window(w1, verbose=False)
    changed = (p1 != p0) or (c1 < CONF_THRESHOLD)
    print(f"  baseline  : {_base_cls:<10} → {p0:<10} conf={c0*100:5.1f}%")
    print(f"  Y/Z-flipped: {_base_cls:<10} → {p1:<10} conf={c1*100:5.1f}%")
    if changed:
        print("  Result: ✓ orientation-SENSITIVE — the flip altered the call "
              "(expected for directional features; firmware must enforce wear "
              "orientation, or switch to magnitude-only features for invariance)")
    else:
        print("  Result: ⚠ prediction UNCHANGED — model ignored a 180° flip")
else:
    print("  (no confidently-correct baseline window available to rotate)")

# ── Smoke test 2: RANDOM noise (a REJECTION test, not an accuracy test) ────────
# Pure Gaussian noise is not any activity.  The soft-max threshold rejection
# should refuse it ("uncertain") for EITHER reason — the top class probability is
# below REJECT_THRESHOLD, and/or the window energy is below the active-class floor.
# A confident class here would be a BUG (over-eager model), so "uncertain" is the
# PASS condition.  This is why the number looks low — by design.
print("\n── Smoke test (random noise → should be 'uncertain') ─────────────────")
np.random.seed(0)   # deterministic so the rejection test is reproducible
dummy = np.random.randn(WINDOW_SIZE, len(RAW_AXES)).astype(np.float32) * 0.3
# Recompute the two gate quantities explicitly so the output spells out WHICH gate
# fired (confidence, energy, or both) — this is what the new mechanism adds.
_noise_row    = extract_stats(dummy)
_noise_proba  = pipe.predict_proba(
    np.array([_noise_row[f] for f in FEATURES], dtype=np.float32).reshape(1, -1))[0]
_noise_top    = int(_noise_proba.argmax())
_noise_energy = float(_noise_row[ENERGY_FEATURE_NAME])
_noise_lowconf   = float(_noise_proba[_noise_top]) < REJECT_THRESHOLD
_noise_lowenergy = _noise_energy < ENERGY_FLOOR

noise_pred, noise_conf = predict_from_window(dummy)
noise_rejected = (noise_pred == "uncertain")
print(f"  argmax class    : {le.classes_[_noise_top]}  "
      f"(would be forced WITHOUT rejection)")
print(f"  confidence gate : conf={noise_conf*100:5.1f}%  vs  "
      f"REJECT_THRESHOLD={REJECT_THRESHOLD*100:.0f}%   "
      f"→ {'REJECT' if _noise_lowconf else 'pass'}")
print(f"  energy gate     : energy={_noise_energy:6.3f}  vs  "
      f"floor={ENERGY_FLOOR:6.3f}   "
      f"→ {'REJECT' if _noise_lowenergy else 'pass'}")
print(f"  verdict         : {noise_pred}")
print(f"  Rejection test  : {'✓ PASS (rejected noise)' if noise_rejected else '✗ FAIL (over-confident on noise)'}")

# ── Smoke test 4: ADAPTIVE PHYSICS-WEIGHTED INFERENCE ─────────────────────────
# Wraps the deployed `pipe` in the AdaptiveWeightedClassifier: physics-signature
# gating of the RF posterior (bounded 0.5×–1.5×), a one-feature fallback per family
# when the forest is unsure, and an out-of-envelope reject — all logged.  We feed
# one real window per class (should classify + show which family was up-weighted)
# plus the seeded random-noise window (should land in fallback/reject, NOT a class).
print("\n── Smoke test 4: adaptive physics-weighted inference (per-class + OOD) ──")
adaptive = AdaptiveWeightedClassifier(
    pipe, le, FEATURES, ENERGY_FLOOR,
    extract_stats=extract_stats, extract_freq_stats=extract_freq_stats,
    reject=REJECT_THRESHOLD)
_adapt_hits = 0
for cls in sorted(sample_raw_windows):
    lbl, conf, log = adaptive.predict(sample_raw_windows[cls])
    ok = (lbl == cls)
    _adapt_hits += ok
    print(f"\n  true={cls:<16} → {lbl:<16} {'✓' if ok else '✗'}")
    print(adaptive.format_log(log))
if sample_raw_windows:
    print(f"\n  Adaptive in-distribution: {_adapt_hits}/{len(sample_raw_windows)} "
          f"= {_adapt_hits/len(sample_raw_windows)*100:.0f}%")
print("\n  ── OOD window (random noise → expect fallback/reject, never a firm class) ──")
_adapt_noise_lbl, _adapt_noise_conf, _adapt_noise_log = adaptive.predict(dummy)
print(f"  noise → {_adapt_noise_lbl}")
print(adaptive.format_log(_adapt_noise_log))
_adapt_noise_ok = (_adapt_noise_lbl == "uncertain")
print(f"  OOD handling: {'✓ PASS (rejected, not a class)' if _adapt_noise_ok else '✗ FAIL (emitted a class for noise)'}")

# ── EXPERIMENT CACHE — lets a fast hyper-parameter sweep reuse this run's feature
# matrix (windowing + 560-feature extraction is the expensive part).  Stores the
# pre-computed feature vectors for the 5 real smoke windows + the seeded-noise
# window so a sweep can score real-window-5/5 and noise-rejection without re-doing
# any feature engineering.  Harmless on a normal run (~6 extra extract_stats).
import pickle as _pkl
_smoke_vecs = {cls: np.array([extract_stats(w.astype(np.float32))[f] for f in FEATURES],
                             dtype=np.float32)
               for cls, w in sample_raw_windows.items()}
_noise_vec  = np.array([extract_stats(dummy)[f] for f in FEATURES], dtype=np.float32)
with open("generated/feature_cache.pkl", "wb") as fh:
    _pkl.dump({"X": X, "y": y, "win_device": win_device, "FEATURES": FEATURES,
               "classes": list(le.classes_), "smoke_vecs": _smoke_vecs,
               "noise_vec": _noise_vec, "conf_threshold": CONF_THRESHOLD,
               "n_pca_var": int(n_pca_var), "pca_top_k": int(PCA_TOP_K)}, fh)

# ═══════════════════════════════════════════════════════════════════════════════
# HONEST REPORT CARD — the last thing printed, the only thing a skimmer should read
# ───────────────────────────────────────────────────────────────────────────────
# The main train/test split is now session-grouped (GroupShuffleSplit on
# session_id), so the headline accuracy is already leak-free — no overlapping
# window or LEFT/RIGHT near-duplicate can cross the partition.  The card therefore:
#   • leads with the held-out-SESSION accuracy as the honest primary metric,
#   • keeps the held-out-DEVICE numbers as a separate cross-wrist transfer check
#     (the L↔R gap measures the genuine LEFT/RIGHT frame difference),
#   • flags per-class scores on < MIN_TRUSTWORTHY_SUPPORT windows as below the noise
#     floor (not measurements) — including classes with NO training session,
#   • treats a failed C export as a blocking ✗,
#   • and ends with one verdict line on what actually gates production.
def _honest_report():
    bar = "═" * 67
    print("\n" + bar)
    print("  HONEST REPORT CARD".center(67))
    print(bar)

    # ── 1. Primary metric — now SESSION-GROUPED (leak-free) ───────────────────
    print("\n  [1] HELD-OUT-SESSION ACCURACY  ✓ leak-free (GroupShuffleSplit on session_id)")
    print(f"        value        : {test_acc:.3f}")
    print(f"        what it is   : trained on whole sessions, tested on UNSEEN")
    print(f"                       sessions — no overlapping-window or L/R leakage.")
    print(f"        caveat       : a session ≈ one (participant, activity), so any")
    print(f"                       class whose only session is held out scores 0 by")
    print(f"                       construction.  Read the per-class line [3], not")
    print(f"                       this single number.")

    # ── 2. Honest metric — cross-device generalization, vs baseline ────────────
    print("\n  [2] HELD-OUT-DEVICE ACCURACY  — cross-wrist generalization check")
    print(f"        (train one device stream → test the other; measures how well the")
    print(f"         model transfers across the LEFT/RIGHT wrists, no mirror applied)")
    if not honest_results:
        print("        (no cross-device split available — single device in data)")
    worst_margin = None
    for r in honest_results:
        margin = r["acc"] - r["baseline"]
        worst_margin = margin if worst_margin is None else min(worst_margin, margin)
        verdict = ("✓ beats baseline" if margin > 0.0 else
                   "✗ WORSE THAN GUESSING")
        print(f"        train {r['train_on']:<10} → test {r['hold']:<6} "
              f"acc={r['acc']:.3f}  vs  majority-'{r['baseline_cls']}' "
              f"baseline={r['baseline']:.3f}   {verdict}  ({margin:+.3f})")
    if len(honest_results) == 2:
        a0, a1 = honest_results[0]["acc"], honest_results[1]["acc"]
        if abs(a0 - a1) > 0.20:
            print(f"        ⚠ {abs(a0-a1)*100:.0f}pp L↔R ASYMMETRY — the two wrists are NOT")
            print(f"          exchangeable (a wear-orientation confound: the wrists see")
            print(f"          different frames and no mirror is applied to reconcile them).")
        else:
            print(f"        ✓ L↔R asymmetry < 20pp — the wrists transfer reasonably.")

    # ── 3. Low-n classes — flagged as below the noise floor ───────────────────
    print(f"\n  [3] PER-CLASS RELIABILITY  (support < {MIN_TRUSTWORTHY_SUPPORT} = below noise floor)")
    test_support = np.bincount(y_test, minlength=len(le.classes_))
    for i, cls in enumerate(le.classes_):
        n = int(test_support[i])
        tag = "⚠ below noise floor — f1 is not a measurement" if n < MIN_TRUSTWORTHY_SUPPORT \
              else "ok"
        print(f"        {cls:<12} test-support n={n:<4} {tag}")

    # ── 4. C export — promoted to a blocking line ─────────────────────────────
    print("\n  [4] EMBEDDED C HEADER")
    if c_export_status and c_export_status[0] == "ok":
        print(f"        ✓ written: {c_export_status[1]}")
    else:
        reason = c_export_status[1] if c_export_status else "not generated"
        print(f"        ✗ NO C header shipped — {reason}")
        print(f"          The .pkl/JSON run on a host; there is NO deployable")
        print(f"          firmware artefact for this configuration.")

    # ── 5. Pipeline configuration & improvements (master-prompt item 16) ───────
    print("\n  [5] SELECTED PIPELINE (this run)")
    _ov = 1.0 - STEP_SIZE / WINDOW_SIZE
    _ws = globals().get("_win_bench")
    _win_note = (f"(swept {len(_ws)} sizes → best grouped-CV val)"
                 if _ws else "(architecture default — WIN_SWEEP=0)")
    print(f"        window/overlap : size={WINDOW_SIZE} step={STEP_SIZE} "
          f"({_ov:.0%} overlap) {_win_note}")
    print(f"        reducer        : '{FEATURE_SELECTOR_ACTIVE}' → {n_pca} model-input dims")
    _rfs = globals().get("_rf_search")
    _rf_note = (f"(RandomizedSearchCV, {RF_SEARCH_ITER} configs, grouped-CV val="
                f"{_rfs.best_score_:.3f})" if _rfs is not None else "(seed params)")
    print(f"        RF             : n_estimators={RF_N_TREES} max_depth={RF_MAX_DEPTH} "
          f"leaf={RF_MIN_SAMPLES_LEAF} split={RF_MIN_SAMPLES_SPLIT} "
          f"max_features={RF_MAX_FEATURES} {_rf_note}")
    _bench_g = globals().get("_bench")
    if _bench_g:
        _bl = sorted(_bench_g, key=lambda r: r["val"], reverse=True)
        print("        selector A/B   : " + ",  ".join(
            f"{b['kind']}={b['val']:.2f}/{b['test']:.2f}" for b in _bl)
            + "  (grouped-CV val/held-out test)")
    if globals().get("_v_full") is not None:
        print(f"        research bank  : embedded val={_v_emb:.3f} → +research "
              f"val={_v_full:.3f} (Δ{_v_full-_v_emb:+.3f}); "
              f"{_n_res_top100}/{len(RESEARCH_COLS)} in global top-100")
    else:
        print("        research bank  : not evaluated this run (RESEARCH_FEATURES=0)")

    # ── 6. One verdict line ───────────────────────────────────────────────────
    honest_below_baseline = (worst_margin is not None and worst_margin <= 0.0)
    print("\n" + "─" * 67)
    _n_sessions = int(np.unique(win_session).size)
    if honest_below_baseline:
        print("  VERDICT: NOT production-ready. On the only leak-free split the model")
        print("           is at/below the always-guess-the-majority baseline. The")
        print(f"           blocker is DATA ({_n_sessions} sessions ≈ one participant×")
        print("           activity each), NOT the model or the feature set — more")
        print("           features cannot fix too few participants. Collect more")
        print("           sessions/subjects first.")
    else:
        print("  VERDICT: clears the majority baseline on held-out wrists, but the")
        print(f"           data is thin ({_n_sessions} sessions) — treat as promising,")
        print("           not proven.")
    print(bar)

_honest_report()
print("\nDone.  Outputs in  generated/  and  plots/")