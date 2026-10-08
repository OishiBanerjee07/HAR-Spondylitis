"""
Dev/validation harness for the UCI-HAR C extractor port (NOT shipped).

Goal: prove that the *float32 algorithms I am about to transcribe into C*
reproduce the source `extract_uci_features()` (the trained-model ground truth)
on real-scale windows, BEFORE writing the C.  No C compiler is available on
this host, so this twin is the strongest parity check we can run locally.

It also derives the two data-driven artefacts the C generator needs:
  * GRAVITY matrix  G  (gravity = G @ accel_axis, exact filtfilt replacement)
  * UCI_KEEP indices   (which of the canonical 435 UCI features survive dedup)
"""
import os, sys, json
import numpy as np

# ── Load the SOURCE config + feature functions verbatim (no IO/training) ──────
SRC = open("HAR_3.py", encoding="utf-8").read().splitlines(keepends=True)
# config constants (lines 29..152) + pure function defs (lines 339..689)
snippet = "".join(SRC[28:152]) + "".join(SRC[154:235]) + "".join(SRC[338:689])
import textwrap  # noqa
from scipy.stats import skew as sp_skew, kurtosis as sp_kurt
import pandas as pd  # noqa
ns = {"__name__": "_src", "np": np, "os": os, "sys": sys, "json": json,
      "textwrap": textwrap, "sp_skew": sp_skew, "sp_kurt": sp_kurt, "pd": pd}
exec(compile(snippet, "HAR_3_src_subset", "exec"), ns)

extract_uci_features = ns["extract_uci_features"]
_grav_split          = ns["_grav_split"]
WINDOW_SIZE          = ns["WINDOW_SIZE"]
NFFT                 = ns["NFFT"]
FS_HZ                = ns["FS_HZ"]
AR_ORDER             = ns["AR_ORDER"]
print(f"Source loaded: WINDOW={WINDOW_SIZE} NFFT={NFFT} FS={FS_HZ} AR={AR_ORDER}")

# ── 1. Gravity matrix G (filtfilt is linear → gravity = G @ x) ────────────────
N = WINDOW_SIZE
G = np.zeros((N, N), dtype=np.float64)
for i in range(N):
    e = np.zeros(N); e[i] = 1.0
    G[:, i] = _grav_split(e)
G32 = G.astype(np.float32)

# ── 2. Canonical 435 UCI order + dedup keep indices ───────────────────────────
probe = np.random.default_rng(1).standard_normal((N, 6)).astype(np.float32)
canon = list(extract_uci_features(probe).keys())          # insertion order = C order
assert len(canon) == 435, len(canon)
params = json.load(open("generated/preprocess_params.json"))
FEATURES = params["feature_names"]
surv = FEATURES[145:]                                     # surviving UCI names
canon_pos = {name: i for i, name in enumerate(canon)}
KEEP = [canon_pos[name] for name in surv]
assert [canon[i] for i in KEEP] == surv, "keep-index ordering mismatch"
print(f"Canonical UCI feats: {len(canon)}   survive dedup: {len(KEEP)}")

# ══════════════════════════════════════════════════════════════════════════════
# FLOAT32 TWIN of the planned C arithmetic
# (uses my own median/percentile/mad/iqr/entropy/moment/levinson, numpy FFT only)
# ══════════════════════════════════════════════════════════════════════════════
f32 = np.float32

def t_sortcopy(x):
    return np.sort(np.asarray(x, dtype=np.float64))       # C will qsort a copy

def t_median(x):
    s = t_sortcopy(x); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

def t_percentile(x, p):                                   # numpy 'linear' interp
    s = t_sortcopy(x); n = len(s)
    pos = p / 100.0 * (n - 1)
    lo = int(np.floor(pos)); frac = pos - lo
    if lo + 1 >= n:
        return s[lo]
    return s[lo] + frac * (s[lo + 1] - s[lo])

def t_mad(x):
    m = t_median(x)
    return t_median(np.abs(np.asarray(x, dtype=np.float64) - m))

def t_iqr(x):
    return t_percentile(x, 75) - t_percentile(x, 25)

def t_mean(x):  return float(np.mean(x))
def t_std(x):   return float(np.std(x))
def t_energy(x):return float(np.mean(np.asarray(x, dtype=np.float64) ** 2))

def t_entropy(x):
    x = np.abs(np.asarray(x, dtype=np.float64)); s = x.sum()
    if s <= 1e-12: return 0.0
    p = x / s; nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(len(x)))

def t_stat(name, x):
    return {"mean": t_mean, "std": t_std, "mad": t_mad, "max": lambda v: float(np.max(v)),
            "min": lambda v: float(np.min(v)), "energy": t_energy, "iqr": t_iqr,
            "entropy": t_entropy}[name](x)

def t_skew(x):
    x = np.asarray(x, dtype=np.float64); mu = x.mean(); sd = x.std()
    if sd <= 1e-9: return 0.0
    return float((((x - mu) / sd) ** 3).mean())

def t_kurt(x):
    x = np.asarray(x, dtype=np.float64); mu = x.mean(); sd = x.std()
    if sd <= 1e-9: return 0.0
    return float((((x - mu) / sd) ** 4).mean() - 3.0)

def t_corr(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.std() <= 1e-9 or b.std() <= 1e-9: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def t_ar(x, order=AR_ORDER):                              # Levinson-Durbin port
    x = np.asarray(x, dtype=np.float64) - np.mean(x); n = len(x)
    r = np.array([np.dot(x[:n - k], x[k:]) for k in range(order + 1)]) / max(n, 1)
    if r[0] <= 1e-12: return [0.0] * order
    a = np.zeros(order + 1); a[0] = 1.0; e = r[0]
    for i in range(1, order + 1):
        acc = r[i] + sum(a[j] * r[i - j] for j in range(1, i))
        k = -acc / e; prev = a.copy()
        for j in range(1, i): a[j] = prev[j] + k * prev[i - j]
        a[i] = k; e *= (1.0 - k * k)
        if e <= 1e-12: break
    return [float(v) for v in a[1:order + 1]]

_FREQS = (np.arange(1, NFFT // 2 + 1) * (FS_HZ / NFFT)).astype(np.float64)
def t_spectrum(x):                                        # magnitude, DC dropped
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    return np.abs(np.fft.rfft(x, n=NFFT))[1:]

def t_meanfreq(mag):
    s = mag.sum(); return float((_FREQS * mag).sum() / s) if s > 1e-12 else 0.0

def t_angle(u, v):
    u = np.asarray(u, float); v = np.asarray(v, float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12: return 0.0
    return float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))

def t_norm3(sig): return np.sqrt((sig ** 2).sum(axis=1))
def t_jerk(sig):
    d = np.diff(sig, axis=0); return np.vstack([d, d[-1:]])

AX = ("X", "Y", "Z")

def twin_uci(win):
    """Return 435 features in canonical order, float32 arithmetic."""
    w = win.astype(np.float32)
    a3 = w[:, 0:3]; g3 = w[:, 3:6]
    grav = (a3.T @ G32.T).T.astype(np.float32)   # per-axis gravity = G @ axis
    grav = np.column_stack([G32 @ a3[:, 0], G32 @ a3[:, 1], G32 @ a3[:, 2]]).astype(np.float32)
    bAcc = (a3 - grav).astype(np.float32)
    bAccJerk = t_jerk(bAcc).astype(np.float32)
    bGyro = g3; bGyroJerk = t_jerk(bGyro).astype(np.float32)
    out = []

    def tri_time(sig):
        cols = [sig[:, 0], sig[:, 1], sig[:, 2]]
        for st in ("mean", "std", "mad", "max", "min"):
            for c in cols: out.append(t_stat(st, c))
        out.append(float(np.mean(np.sum(np.abs(sig), axis=1))))       # sma
        for st in ("energy", "iqr", "entropy"):
            for c in cols: out.append(t_stat(st, c))
        for c in cols:
            for coef in t_ar(c): out.append(coef)
        out.append(t_corr(cols[0], cols[1])); out.append(t_corr(cols[0], cols[2]))
        out.append(t_corr(cols[1], cols[2]))

    def mag_time(x):
        for st in ("mean", "std", "mad", "max", "min"): out.append(t_stat(st, x))
        out.append(float(np.mean(np.abs(x))))                          # sma
        for st in ("energy", "iqr", "entropy"): out.append(t_stat(st, x))
        for coef in t_ar(x): out.append(coef)

    def tri_freq(sig):
        specs = [t_spectrum(sig[:, i]) for i in range(3)]
        for st in ("mean", "std", "mad", "max", "min"):
            for m in specs: out.append(t_stat(st, m))
        out.append(float(np.mean([np.sum(np.abs(m)) for m in specs])))  # sma
        for st in ("energy", "iqr", "entropy"):
            for m in specs: out.append(t_stat(st, m))
        for m in specs: out.append(float(int(np.argmax(m))))            # maxInds
        for m in specs: out.append(t_meanfreq(m))
        for m in specs: out.append(t_skew(m))
        for m in specs: out.append(t_kurt(m))

    def mag_freq(x):
        mag = t_spectrum(x)
        for st in ("mean", "std", "mad", "max", "min"): out.append(t_stat(st, mag))
        out.append(float(np.sum(np.abs(mag))))                         # sma
        for st in ("energy", "iqr", "entropy"): out.append(t_stat(st, mag))
        out.append(float(int(np.argmax(mag))))
        out.append(t_meanfreq(mag)); out.append(t_skew(mag)); out.append(t_kurt(mag))

    for sig in (bAcc, grav, bAccJerk, bGyro, bGyroJerk): tri_time(sig)
    for sig in (bAcc, grav, bAccJerk, bGyro, bGyroJerk): mag_time(t_norm3(sig))
    for sig in (bAcc, bAccJerk, bGyro):                  tri_freq(sig)
    for sig in (bAcc, bAccJerk, bGyro, bGyroJerk):       mag_freq(t_norm3(sig))

    gmean = grav.mean(axis=0)
    out.append(t_angle(bAcc.mean(axis=0), gmean))
    out.append(t_angle(bAccJerk.mean(axis=0), gmean))
    out.append(t_angle(bGyro.mean(axis=0), gmean))
    out.append(t_angle(bGyroJerk.mean(axis=0), gmean))
    out.append(t_angle([1, 0, 0], gmean))
    out.append(t_angle([0, 1, 0], gmean))
    out.append(t_angle([0, 0, 1], gmean))
    return np.array(out, dtype=np.float64)

# ── 3. Compare twin vs SOURCE on many real-scale windows ──────────────────────
rng = np.random.default_rng(7)
worst = 0.0; worst_name = None
for trial in range(400):
    # accel ~ ±2 g around gravity, gyro ~ ±5 rad/s — realistic IMU scale
    win = np.empty((N, 6), dtype=np.float32)
    win[:, 0:3] = rng.standard_normal((N, 3)) * rng.uniform(0.05, 2.0) + rng.uniform(-9.8, 9.8)
    win[:, 3:6] = rng.standard_normal((N, 3)) * rng.uniform(0.05, 5.0)
    ref_d = extract_uci_features(win)
    ref = np.array([ref_d[n] for n in canon], dtype=np.float64)
    ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
    tw = twin_uci(win)
    err = np.abs(ref - tw)
    # relative-tolerant: large-magnitude freq energies dominate abs error
    scale = np.maximum(np.abs(ref), 1.0)
    rel = err / scale
    j = int(rel.argmax())
    if rel[j] > worst:
        worst = rel[j]; worst_name = canon[j]; worst_abs = err[j]; worst_ref = ref[j]

print(f"\nTWIN vs SOURCE over 400 windows:")
print(f"  worst rel err = {worst:.2e}  at  {worst_name}")
print(f"  (abs={worst_abs:.3e}, ref={worst_ref:.3e})")
print("  PASS" if worst < 1e-3 else "  *** FAIL — formula mismatch ***")

# ── 4. Persist artefacts the C generator will consume ─────────────────────────
np.save("generated/_gravity_matrix.npy", G32)
json.dump({"canon": canon, "keep": KEEP},
          open("generated/_uci_canon_keep.json", "w"))
print("\nSaved generated/_gravity_matrix.npy and generated/_uci_canon_keep.json")

# ── 5. END-TO-END parity: does twin-UCI vs source-UCI change predictions? ──────
# The 145-feature time+freq prefix already has proven C parity, so hold it fixed
# (use source values for both) and swap ONLY the UCI block: source vs twin.
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
# Re-create TopPCAComponents in __main__ so the pickle can resolve it.
_cls_src = "".join(SRC[906:946])  # class TopPCAComponents(...) ... transform()
_cls_ns = dict(ns); _cls_ns.update({"BaseEstimator": BaseEstimator,
    "TransformerMixin": TransformerMixin, "PCA": PCA,
    "RandomForestClassifier": RandomForestClassifier, "class_weight": None})
exec(compile(_cls_src, "TopPCA_src", "exec"), _cls_ns)
sys.modules["__main__"].TopPCAComponents = _cls_ns["TopPCAComponents"]
pipe = joblib.load("generated/imu_pipeline.pkl")
extract_stats = ns["extract_stats"]
scaler = pipe.named_steps["scaler"]; pca = pipe.named_steps["pca"]

rng = np.random.default_rng(123)
n_disagree = 0; max_proba_d = 0.0; max_scaled_d = 0.0; max_feat_abs = 0.0
NTEST = 300
for _ in range(NTEST):
    win = np.empty((N, 6), dtype=np.float32)
    win[:, 0:3] = rng.standard_normal((N, 3)) * rng.uniform(0.05, 2.0) + rng.uniform(-9.8, 9.8)
    win[:, 3:6] = rng.standard_normal((N, 3)) * rng.uniform(0.05, 5.0)
    full = extract_stats(win)                      # source: all 560 by name
    src_vec = np.array([full[f] for f in FEATURES], dtype=np.float64)
    src_vec = np.nan_to_num(src_vec, nan=0.0, posinf=0.0, neginf=0.0)
    # twin vector: source 145 prefix + twin UCI (canonical→keep)
    tw_canon = twin_uci(win)
    tw_uci = np.array([tw_canon[i] for i in KEEP], dtype=np.float64)
    tw_vec = src_vec.copy(); tw_vec[145:] = tw_uci
    max_feat_abs = max(max_feat_abs, np.abs(src_vec - tw_vec).max())
    # scaled-space divergence
    ss = (src_vec - scaler.mean_) / scaler.scale_
    ts = (tw_vec  - scaler.mean_) / scaler.scale_
    max_scaled_d = max(max_scaled_d, np.linalg.norm(ss - ts))
    ps = pipe.predict_proba(src_vec.reshape(1, -1).astype(np.float32))[0]
    pt = pipe.predict_proba(tw_vec.reshape(1, -1).astype(np.float32))[0]
    max_proba_d = max(max_proba_d, np.abs(ps - pt).max())
    if ps.argmax() != pt.argmax(): n_disagree += 1

print(f"\nEND-TO-END (twin-UCI vs source-UCI through trained pipeline, {NTEST} windows):")
print(f"  max raw-feature abs diff   : {max_feat_abs:.3e}")
print(f"  max scaled-vector L2 diff  : {max_scaled_d:.3e}  (560-dim standardized)")
print(f"  max abs d predict_proba    : {max_proba_d:.3e}")
print(f"  class disagreements        : {n_disagree}/{NTEST}")
print("  PREDICTION PARITY OK" if n_disagree == 0 and max_proba_d < 1e-2
      else "  *** prediction divergence — investigate ***")

# ── 6. REAL-WINDOW parity (the deployment distribution) ───────────────────────
print("\nLoading real windows from dataset (one-time, ~slow)...")
_read_xlsx = ns["_read_xlsx"]; FLIP_COLS = ns["FLIP_COLS"]
RAW_AXES = ns["RAW_AXES"]; STEP = ns["STEP_SIZE"]
dfr = _read_xlsx("Production_IMU_Dataset.xlsx")
rmask = dfr["device_id"].str.upper().str.strip() == "RIGHT"
for col in FLIP_COLS: dfr.loc[rmask, col] = -dfr.loc[rmask, col]
dfr = dfr.sort_values(["session_id", "device_id", "timestamp"]).reset_index(drop=True)
real_wins = []
for (sid, dev), grp in dfr.groupby(["session_id", "device_id"], sort=False):
    raw = grp[RAW_AXES].values.astype(np.float32)
    for s in range(0, len(raw) - N + 1, STEP):
        real_wins.append(raw[s:s + N])
        if len(real_wins) >= 400: break
    if len(real_wins) >= 400: break
print(f"  collected {len(real_wins)} real windows")

n_dis = 0; mx_proba = 0.0; mx_feat = 0.0; mx_scaled = 0.0
for win in real_wins:
    full = extract_stats(win)
    src_vec = np.nan_to_num(np.array([full[f] for f in FEATURES], dtype=np.float64))
    tw_canon = twin_uci(win)
    tw_vec = src_vec.copy()
    tw_vec[145:] = np.array([tw_canon[i] for i in KEEP], dtype=np.float64)
    mx_feat = max(mx_feat, np.abs(src_vec - tw_vec).max())
    ss = (src_vec - scaler.mean_) / scaler.scale_
    ts = (tw_vec - scaler.mean_) / scaler.scale_
    mx_scaled = max(mx_scaled, np.linalg.norm(ss - ts))
    ps = pipe.predict_proba(src_vec.reshape(1, -1).astype(np.float32))[0]
    pt = pipe.predict_proba(tw_vec.reshape(1, -1).astype(np.float32))[0]
    mx_proba = max(mx_proba, np.abs(ps - pt).max())
    if ps.argmax() != pt.argmax(): n_dis += 1
print(f"  REAL max raw-feature abs diff : {mx_feat:.3e}")
print(f"  REAL max scaled L2 diff       : {mx_scaled:.3e}")
print(f"  REAL max abs d predict_proba  : {mx_proba:.3e}")
print(f"  REAL class disagreements      : {n_dis}/{len(real_wins)}")
print("  REAL PREDICTION PARITY OK" if n_dis == 0 and mx_proba < 1e-2
      else "  *** real divergence ***")
