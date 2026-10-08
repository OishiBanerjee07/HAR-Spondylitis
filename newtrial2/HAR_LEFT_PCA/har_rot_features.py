#!/usr/bin/env python3
"""
har_rot_features.py — feature / preprocessing ENGINE for the 12-way (Body x
Direction x Side) arm-rotation HAR model (merged spec v2).

WHY A SEPARATE, SIDE-EFFECT-FREE MODULE
---------------------------------------
Importing HAR_4_PCA.py re-runs its whole training pipeline (it has module-level
side effects).  This module imports NOTHING that trains and does NOTHING on
import except define functions/constants, so both the new hierarchical trainer
(HAR_ROT_HIER.py) AND the firmware golden-vector generator can `import
har_rot_features` cheaply.

WHAT IS REPRODUCED EXACTLY (carried over from the validated pipeline, C-parity)
------------------------------------------------------------------------------
  * the 79 core time-domain stats           (EXPECTED_TIME_FEAT = 79)
  * the 66 frequency-domain stats            (USE_FREQ_FEATURES)
  * the UCI-HAR derived-signal block (~435)  (USE_UCI_FEATURES)
  * the exact-duplicate dedup pass
Every constant, formula and index order below matches the source pipeline; only
FS_HZ (100 -> 50) and the Hz-denominated constants derived from it change, per
the confirmed 50 Hz sampling rate.

WHAT IS NEW (the 12-class rotational spec)
------------------------------------------
  * 50 Hz preprocessing chain: median filter -> Butterworth LP 10 Hz ->
    Madgwick AHRS fusion -> optional Savitzky-Golay
  * canonical body-frame remap for the LEFT wrist + CW = +ve sign convention
  * gravity / linear-accel split (Butterworth 0.3 Hz, zero-phase)
  * purpose-built Group B (body part), Group D (signed direction, PROTECTED),
    Group S (side, dual-device) features
  * Family P phase/directional features (odd under CW<->CCW reversal)

NOTE ON UNITS / MOUNTING (open items — see HAR_ROT_HIER.py DATASET CONFIG):
  * gyro units (deg/s vs rad/s) -> GYRO_IN_DEG
  * exact LEFT-wrist axis remap -> LEFT_CANONICAL_REMAP (depends on physical
    mounting; unit-tested by canonical_frame_selfcheck())
"""
from __future__ import annotations
import numpy as np
from scipy.stats import skew as sp_skew, kurtosis as sp_kurt

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — sampling-rate-derived constants (50 Hz confirmed)
# ══════════════════════════════════════════════════════════════════════════════
FS_HZ        = 50.0            # confirmed sampling rate (spec v2 §2)
WINDOW_SIZE  = 100             # 2 s @ 50 Hz
STEP_SIZE    = 50             # 50 % overlap (kept from source; DO NOT retune silently)
NFFT         = 128             # power-of-two radix-2 FFT (exact C parity)

# Feature-layer toggles (identical semantics to the source pipeline)
USE_FREQ_FEATURES = True
USE_UCI_FEATURES  = True
EXPECTED_TIME_FEAT  = 79
GRAVITY_CUTOFF_HZ   = 0.3      # UCI / Group-B gravity low-pass cutoff
AR_ORDER            = 4

# 6 raw axes, FIXED order (raw input contract §1)
RAW_AXES   = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
CORR_PAIRS = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
CORR_NAMES = [("accel_x", "accel_y"), ("accel_x", "accel_z"), ("accel_y", "accel_z"),
              ("gyro_x", "gyro_y"),   ("gyro_x", "gyro_z"),   ("gyro_y", "gyro_z")]
JERK_IDX   = [0, 1, 2]

# 11 signals get spectral features (parallels the time-domain signal set)
FREQ_SIGNALS = ["accel_x", "accel_y", "accel_z",
                "gyro_x",  "gyro_y",  "gyro_z",
                "accel_mag", "gyro_mag",
                "jerk_x", "jerk_y", "jerk_z"]
FREQ_STATS_PER_SIGNAL = 6
NUM_FREQ_FEAT = len(FREQ_SIGNALS) * FREQ_STATS_PER_SIGNAL if USE_FREQ_FEATURES else 0
EXPECTED_TOTAL_FEAT = EXPECTED_TIME_FEAT + NUM_FREQ_FEAT   # 145 original prefix

# ── NEW preprocessing constants (spec v2 §3) ──────────────────────────────────
MEDIAN_KERNEL   = 3            # despike median filter (kernel 3-5)
LP_CUTOFF_HZ    = 10.0         # Butterworth low-pass cutoff (do NOT cut below ~5 Hz)
LP_ORDER        = 4
SAVGOL_WIN      = 7            # optional Savitzky-Golay on angular-velocity streams
SAVGOL_POLY     = 2
USE_SAVGOL      = False        # off by default (skip if it attenuates rotation peaks)
MADGWICK_BETA   = 0.1          # AHRS gain; complementary-filter fallback below
GYRO_IN_DEG     = True         # dataset gyro units — set from DATASET CONFIG

# ── Group-B / Group-S band + rest constants ───────────────────────────────────
BODY_LOWBAND_HZ = (0.5, 2.0)   # arm-pendulum band
BODY_MIDBAND_HZ = (2.0, 5.0)   # wrist-flick band
SIDE_XCORR_LAG_S = 0.2         # +/- lag for cross-arm correlation
FAMILY_P_MAX_LAG = int(0.1 * FS_HZ)   # 5 samples @ 50 Hz (spec: int(0.1*fs))

# AC-band frequency axis (DC bin dropped) — depends on FS_HZ & NFFT
_FREQS_AC = np.fft.rfftfreq(NFFT, d=1.0 / FS_HZ)[1:]   # (64,)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — faithful reproduction of the 79 + 66 + UCI extractors (C-parity)
# ══════════════════════════════════════════════════════════════════════════════
def _safe_corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0


def _axis_stats(x: np.ndarray, col_name: str) -> dict:
    """8 statistics for a 1-D float64 signal: mean std range rms energy skew kurt zcr."""
    n      = len(x)
    mu     = x.mean()
    sd     = x.std()
    energy = (x ** 2).mean()
    zcr    = float(((x[:-1] * x[1:]) < 0).sum()) / (n - 1)
    return {
        f"{col_name}__mean":   float(mu),
        f"{col_name}__std":    float(sd),
        f"{col_name}__range":  float(x.max() - x.min()),
        f"{col_name}__rms":    float(np.sqrt(energy)),
        f"{col_name}__energy": float(energy),
        f"{col_name}__skew":   float(sp_skew(x, bias=True)),
        f"{col_name}__kurt":   float(sp_kurt(x, fisher=True, bias=True)),
        f"{col_name}__zcr":    zcr,
    }


def _freq_axis_stats(x: np.ndarray, name: str) -> dict:
    """6 spectral statistics on the AC bins of the zero-padded PSD (DC dropped)."""
    X      = np.fft.rfft(x, n=NFFT)
    psd    = np.abs(X) ** 2
    psd_ac = psd[1:]                       # drop DC
    total  = float(psd_ac.sum()) + 1e-12
    pn     = psd_ac / total
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
    """66 features (11 signals x 6 stats) in FREQ_SIGNALS order."""
    w   = win.astype(np.float64)
    sig = {
        "accel_x": w[:, 0], "accel_y": w[:, 1], "accel_z": w[:, 2],
        "gyro_x":  w[:, 3], "gyro_y":  w[:, 4], "gyro_z":  w[:, 5],
    }
    sig["accel_mag"] = np.sqrt(w[:, 0] ** 2 + w[:, 1] ** 2 + w[:, 2] ** 2)
    sig["gyro_mag"]  = np.sqrt(w[:, 3] ** 2 + w[:, 4] ** 2 + w[:, 5] ** 2)
    sig["jerk_x"]    = np.diff(w[:, 0])
    sig["jerk_y"]    = np.diff(w[:, 1])
    sig["jerk_z"]    = np.diff(w[:, 2])
    feats = {}
    for name in FREQ_SIGNALS:
        feats.update(_freq_axis_stats(sig[name], name))
    return feats


# ── UCI-HAR derived-signal extractor helpers ──────────────────────────────────
def _grav_split(x: np.ndarray) -> np.ndarray:
    """Low-pass an accel axis to estimate its gravity component (UCI 0.3 Hz)."""
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(3, GRAVITY_CUTOFF_HZ / (FS_HZ / 2.0), btype="low")
        pad  = min(len(x) - 1, 3 * (max(len(a), len(b)) - 1))
        return filtfilt(b, a, x, padlen=pad)
    except Exception:
        alpha = 0.02                       # zero-phase EMA fallback (fwd then back)
        g = x.astype(np.float64).copy()
        for i in range(1, len(g)):           g[i] = alpha * x[i]     + (1 - alpha) * g[i - 1]
        for i in range(len(g) - 2, -1, -1):  g[i] = alpha * g[i + 1] + (1 - alpha) * g[i]
        return g


def _mad(x):
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
    """AR(order) coefficients via Levinson-Durbin on the biased autocorrelation."""
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
    if name == "mean":    return float(np.mean(x))
    if name == "std":     return float(np.std(x))
    if name == "mad":     return _mad(x)
    if name == "max":     return float(np.max(x))
    if name == "min":     return float(np.min(x))
    if name == "energy":  return float(np.mean(np.asarray(x) ** 2))
    if name == "iqr":     return _iqr(x)
    if name == "entropy": return _entropy(x)
    return 0.0


def _spectrum(x):
    """FFT-magnitude spectrum (DC dropped) of a detrended signal + its freq axis."""
    x   = np.asarray(x, dtype=np.float64) - np.mean(x)
    mag = np.abs(np.fft.rfft(x, n=NFFT))[1:]
    return mag, _FREQS_AC


def _mean_freq(mag, freqs):
    s = mag.sum()
    return float((freqs * mag).sum() / s) if s > 1e-12 else 0.0


def _norm3(sig):
    return np.sqrt((sig ** 2).sum(axis=1))


def _angle(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))


def _uci_triaxial_time(name, sig):
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
    """Reconstruct the UCI-HAR derived-signal set (~435 features pre-dedup)."""
    w  = win.astype(np.float64)
    a3 = w[:, 0:3]
    g3 = w[:, 3:6]
    grav = np.column_stack([_grav_split(a3[:, 0]),
                            _grav_split(a3[:, 1]),
                            _grav_split(a3[:, 2])])
    bAcc = a3 - grav

    def _jerk(sig):
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


def extract_base_stats(win: np.ndarray) -> dict:
    """The 79 core stats + optional 66 freq + optional UCI block (shared base,
    per device).  Identical layout/formulas to the source pipeline for C parity."""
    feats = {}
    w = win.astype(np.float64)
    for ai, col in enumerate(RAW_AXES):
        feats.update(_axis_stats(w[:, ai], col))
    accel_mag = np.sqrt(w[:, 0] ** 2 + w[:, 1] ** 2 + w[:, 2] ** 2)
    gyro_mag  = np.sqrt(w[:, 3] ** 2 + w[:, 4] ** 2 + w[:, 5] ** 2)
    feats.update(_axis_stats(accel_mag, "accel_mag"))
    feats.update(_axis_stats(gyro_mag,  "gyro_mag"))
    for (ia, ib), (ca, cb) in zip(CORR_PAIRS, CORR_NAMES):
        feats[f"corr__{ca}__{cb}"] = _safe_corr(w[:, ia], w[:, ib])
    for ai in JERK_IDX:
        col  = RAW_AXES[ai]; ax_l = col.split("_")[1]
        j    = np.diff(w[:, ai])
        feats[f"jerk_{ax_l}__mean"] = float(j.mean())
        feats[f"jerk_{ax_l}__std"]  = float(j.std())
        feats[f"jerk_{ax_l}__rms"]  = float(np.sqrt((j ** 2).mean()))
    if USE_FREQ_FEATURES:
        feats.update(extract_freq_stats(win))
    if USE_UCI_FEATURES:
        feats.update(extract_uci_features(win))
    return feats


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — 50 Hz preprocessing chain (spec v2 §3, per channel, per device)
#   order: median filter -> Butterworth LP 10 Hz -> Madgwick fusion -> [Savgol]
# ══════════════════════════════════════════════════════════════════════════════
def median_filter_stream(x: np.ndarray, kernel: int = MEDIAN_KERNEL) -> np.ndarray:
    """Despike each column with a centred median filter (kernel 3-5, odd)."""
    try:
        from scipy.signal import medfilt
        k = kernel if kernel % 2 == 1 else kernel + 1
        return np.column_stack([medfilt(x[:, i], k) for i in range(x.shape[1])])
    except Exception:
        return x


def butter_lowpass_stream(x: np.ndarray, cutoff=LP_CUTOFF_HZ, order=LP_ORDER,
                          fs=FS_HZ) -> np.ndarray:
    """Zero-phase Butterworth low-pass (sensor-noise removal).  Cutoff MUST stay
    >~5 Hz — arm-rotation fundamentals sit at 0.5-3 Hz and their harmonics carry
    body-part information."""
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(order, cutoff / (fs / 2.0), btype="low")
        pad  = min(len(x) - 1, 3 * (max(len(a), len(b)) - 1))
        return np.column_stack([filtfilt(b, a, x[:, i], padlen=pad)
                                for i in range(x.shape[1])])
    except Exception:
        return x


def savgol_stream(x: np.ndarray, win=SAVGOL_WIN, poly=SAVGOL_POLY) -> np.ndarray:
    """Optional light Savitzky-Golay smoothing on angular-velocity streams."""
    try:
        from scipy.signal import savgol_filter
        w = min(win if win % 2 == 1 else win + 1, len(x) - (1 - len(x) % 2))
        if w <= poly:
            return x
        return np.column_stack([savgol_filter(x[:, i], w, poly)
                                for i in range(x.shape[1])])
    except Exception:
        return x


# ── Madgwick AHRS (IMU-only: accel + gyro) ────────────────────────────────────
def _quat_mult(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def madgwick_stream(gyro: np.ndarray, accel: np.ndarray, fs=FS_HZ,
                    beta=MADGWICK_BETA, gyro_in_deg=None) -> np.ndarray:
    """Madgwick IMU update over a window -> quaternion orientation stream (N,4).

    gyro  : (N,3) angular velocity (deg/s if GYRO_IN_DEG else rad/s)
    accel : (N,3) acceleration (units irrelevant — normalised internally)
    Returns unit quaternions [w,x,y,z] per sample.  Run only AFTER median + LP
    filtering (needs low-noise gyro).  Fallback: complementary_stream()."""
    gyro_in_deg = GYRO_IN_DEG if gyro_in_deg is None else gyro_in_deg
    g = np.deg2rad(gyro) if gyro_in_deg else np.asarray(gyro, float)
    a = np.asarray(accel, float)
    dt = 1.0 / fs
    q  = np.array([1.0, 0.0, 0.0, 0.0])
    out = np.zeros((len(g), 4))
    for t in range(len(g)):
        gx, gy, gz = g[t]
        ax, ay, az = a[t]
        n = np.linalg.norm([ax, ay, az])
        qdot = 0.5 * _quat_mult(q, np.array([0.0, gx, gy, gz]))
        if n > 1e-9:                                   # accel corrective step
            ax, ay, az = ax / n, ay / n, az / n
            q1, q2, q3, q4 = q
            f = np.array([2 * (q2 * q4 - q1 * q3) - ax,
                          2 * (q1 * q2 + q3 * q4) - ay,
                          2 * (0.5 - q2 * q2 - q3 * q3) - az])
            J = np.array([[-2 * q3,  2 * q4, -2 * q1, 2 * q2],
                          [ 2 * q2,  2 * q1,  2 * q4, 2 * q3],
                          [ 0.0,    -4 * q2, -4 * q3, 0.0]])
            grad = J.T @ f
            ng = np.linalg.norm(grad)
            if ng > 1e-9:
                qdot = qdot - beta * (grad / ng)
        q = q + qdot * dt
        q = q / (np.linalg.norm(q) + 1e-12)
        out[t] = q
    return out


def complementary_stream(gyro: np.ndarray, accel: np.ndarray, fs=FS_HZ,
                         alpha=0.98, gyro_in_deg=None) -> np.ndarray:
    """Complementary-filter fallback if Madgwick is unstable at 50 Hz.  Returns a
    quaternion stream so downstream orientation features are format-compatible."""
    gyro_in_deg = GYRO_IN_DEG if gyro_in_deg is None else gyro_in_deg
    g = np.deg2rad(gyro) if gyro_in_deg else np.asarray(gyro, float)
    a = np.asarray(accel, float)
    dt = 1.0 / fs
    roll = pitch = 0.0
    out = np.zeros((len(g), 4))
    for t in range(len(g)):
        gx, gy, _ = g[t]
        ax, ay, az = a[t]
        roll_a  = np.arctan2(ay, az)
        pitch_a = np.arctan2(-ax, np.sqrt(ay * ay + az * az) + 1e-12)
        roll    = alpha * (roll + gx * dt)  + (1 - alpha) * roll_a
        pitch   = alpha * (pitch + gy * dt) + (1 - alpha) * pitch_a
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        out[t] = [cr * cp, sr * cp, cr * sp, -sr * sp]
    return out


def quaternion_sweep(quat: np.ndarray) -> float:
    """Total orientation angle traversed across a window (Group-B feature).
    Sum of geodesic angles between consecutive unit quaternions."""
    q = np.asarray(quat, float)
    dots = np.abs(np.clip(np.sum(q[:-1] * q[1:], axis=1), -1.0, 1.0))
    return float(2.0 * np.arccos(dots).sum())


def preprocess_device_stream(raw6: np.ndarray) -> dict:
    """Full per-device preprocessing on a (N,6) [accel3, gyro3] stream.
    Returns filtered raw, gravity, linear accel, quaternion + angle streams."""
    x = median_filter_stream(np.asarray(raw6, float))
    x = butter_lowpass_stream(x)
    accel, gyro = x[:, 0:3], x[:, 3:6]
    if USE_SAVGOL:
        gyro = savgol_stream(gyro)
        x = np.column_stack([accel, gyro])
    grav   = np.column_stack([_grav_split(accel[:, i]) for i in range(3)])
    linacc = accel - grav                                 # gravity-free (§3)
    try:
        quat = madgwick_stream(gyro, accel)
    except Exception:
        quat = complementary_stream(gyro, accel)
    return {"raw6": x, "accel": accel, "gyro": gyro,
            "gravity": grav, "linacc": linacc, "quat": quat}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — canonical body frame (spec v2 §2) : LEFT-wrist remap + CW=+ve
# ══════════════════════════════════════════════════════════════════════════════
# The exact remap depends on physical mounting (OPEN item).  A remap is a list of
# 6 (src_index, sign) pairs mapping OUTPUT axis k -> sign * input[src].  Identity
# is [(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)].  The default LEFT remap below is the
# common "mirror about X": accel_x flips (polar vector, one axis) and gyro_y,
# gyro_z flip (axial vector, other two) — bring both wrists into one frame.
IDENTITY_REMAP       = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
LEFT_CANONICAL_REMAP = [(0, -1), (1, 1), (2, 1), (3, 1), (4, -1), (5, -1)]


def apply_remap(raw6: np.ndarray, remap) -> np.ndarray:
    """Reorder/sign-flip a (N,6) stream into the canonical body frame."""
    x = np.asarray(raw6, float)
    out = np.empty_like(x)
    for k, (src, sgn) in enumerate(remap):
        out[:, k] = sgn * x[:, src]
    return out


def canonicalize_device(raw6: np.ndarray, side: str,
                        left_remap=None, right_remap=None) -> np.ndarray:
    """Canonicalise one device's (N,6) stream given its side ('LEFT'/'RIGHT')."""
    left_remap  = LEFT_CANONICAL_REMAP if left_remap is None else left_remap
    right_remap = IDENTITY_REMAP if right_remap is None else right_remap
    return apply_remap(raw6, left_remap if str(side).upper().startswith("L")
                       else right_remap)


def canonical_frame_selfcheck(both_hand_clk_left6: np.ndarray,
                              both_hand_clk_right6: np.ndarray,
                              left_remap=None, right_remap=None) -> dict:
    """Unit test (§2): after canonicalisation a 'Both Hand CLK' window must show
    SAME-SIGN mean gyro on both devices.  Returns per-axis signs + a pass flag."""
    L = canonicalize_device(both_hand_clk_left6,  "LEFT",  left_remap, right_remap)
    R = canonicalize_device(both_hand_clk_right6, "RIGHT", left_remap, right_remap)
    lg = L[:, 3:6].mean(axis=0)
    rg = R[:, 3:6].mean(axis=0)
    same = np.sign(lg) == np.sign(rg)
    dom  = int(np.argmax(np.abs(lg) + np.abs(rg)))      # dominant rotation axis
    return {"left_mean_gyro": lg.tolist(), "right_mean_gyro": rg.tolist(),
            "same_sign_per_axis": same.tolist(),
            "dominant_axis_agrees": bool(same[dom]), "pass": bool(same[dom])}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — purpose-built rotational features (spec v2 §4)
#   Group B  : body part (Shoulder vs Hand)   — ratios/orientation-sweep/bands
#   Group D  : direction (CW vs CCW)          — SIGNED, PROTECTED, never folded
#   Group S  : side (Left/Right/Both)         — requires dual-device windows
#   Family P : phase/directional (odd under CW<->CCW)
# All feature NAMES are prefixed so protection/selection logic can match them.
# ══════════════════════════════════════════════════════════════════════════════
def _band_energy(sig1d: np.ndarray, lo: float, hi: float) -> float:
    """AC power of a 1-D signal within [lo,hi] Hz (DC dropped)."""
    mag = np.abs(np.fft.rfft(sig1d - np.mean(sig1d), n=NFFT))[1:] ** 2
    band = (_FREQS_AC >= lo) & (_FREQS_AC < hi)
    return float(mag[band].sum())


def _cumtrapz(y: np.ndarray, dt: float) -> np.ndarray:
    """Cumulative trapezoidal integral (angular displacement per axis)."""
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum((y[1:] + y[:-1]) * 0.5 * dt)
    return out


def group_B_features(prep: dict, prefix: str = "B") -> dict:
    """Group B — Body part (Shoulder vs Hand).  Invest here (old bottleneck)."""
    gyro   = prep["gyro"]
    linacc = prep["linacc"]
    f = {}
    e_gyro = float(np.sum(gyro ** 2))
    e_lin  = float(np.sum(linacc ** 2)) + 1e-12
    # THE separating physics: rotation-in-place (Hand, high) vs big arc (Shoulder, low)
    f[f"{prefix}__gyro_accel_energy_ratio"] = e_gyro / e_lin
    lin_mag = _norm3(linacc)
    f[f"{prefix}__linacc_mag_var"]  = float(np.var(lin_mag))
    f[f"{prefix}__linacc_mag_peak"] = float(np.max(lin_mag))
    for i, ax in enumerate("xyz"):
        f[f"{prefix}__linacc_p2p_{ax}"] = float(np.ptp(linacc[:, i]))
    f[f"{prefix}__orient_sweep"] = quaternion_sweep(prep["quat"])
    lowb = sum(_band_energy(linacc[:, i], *BODY_LOWBAND_HZ) for i in range(3))
    midb = sum(_band_energy(linacc[:, i], *BODY_MIDBAND_HZ) for i in range(3)) + 1e-12
    f[f"{prefix}__low_mid_band_ratio"] = float(lowb / midb)
    return f


def group_D_features(prep: dict, prefix: str = "D") -> dict:
    """Group D — Direction (CW vs CCW).  SIGNED features only (PROTECTED); these
    bypass magnitude/variance reducer heuristics.  Names carry the protect prefix."""
    gyro = prep["gyro"]
    quat = prep["quat"]
    dt   = 1.0 / FS_HZ
    f = {}
    means = gyro.mean(axis=0)
    for i, ax in enumerate("xyz"):
        f[f"{prefix}__gyro_signed_mean_{ax}"] = float(means[i])
    # angular displacement per axis (cumulative trapezoid); net sign = direction
    disp = np.array([_cumtrapz(gyro[:, i], dt)[-1] for i in range(3)])
    for i, ax in enumerate("xyz"):
        f[f"{prefix}__ang_disp_{ax}"] = float(disp[i])
    dom = int(np.argmax(np.abs(means)))                    # dominant rotation axis
    f[f"{prefix}__ang_disp_dominant"] = float(disp[dom])
    for i, ax in enumerate("xyz"):
        pos = float(np.max(gyro[:, i])); neg = float(np.min(gyro[:, i]))
        f[f"{prefix}__gyro_pospeak_{ax}"]  = pos
        f[f"{prefix}__gyro_negpeak_{ax}"]  = neg
        f[f"{prefix}__gyro_peak_asym_{ax}"] = pos + neg     # asymmetry (signed)
    for i, ax in enumerate("xyz"):
        g = gyro[:, i]
        f[f"{prefix}__gyro_zcr_{ax}"] = float(((g[:-1] * g[1:]) < 0).sum()) / (len(g) - 1)
    # cross-product trajectory sign on the orientation projected to rotation plane
    # (Madgwick-derived, NOT double-integrated accel).  D<0 -> CW, D>0 -> CCW.
    vx, vy = quat[:, 1], quat[:, 2]                         # 2 non-dominant quat axes
    D = float(np.mean(vx[:-1] * vy[1:] - vy[:-1] * vx[1:]))
    f[f"{prefix}__crossprod_sign"] = D
    return f


def group_D_feature_names(prefix: str = "D") -> list:
    """The exact set of Group-D (protected direction) feature names for a prefix."""
    names = []
    for ax in "xyz":
        names.append(f"{prefix}__gyro_signed_mean_{ax}")
    for ax in "xyz":
        names.append(f"{prefix}__ang_disp_{ax}")
    names.append(f"{prefix}__ang_disp_dominant")
    for ax in "xyz":
        names += [f"{prefix}__gyro_pospeak_{ax}", f"{prefix}__gyro_negpeak_{ax}",
                  f"{prefix}__gyro_peak_asym_{ax}"]
    for ax in "xyz":
        names.append(f"{prefix}__gyro_zcr_{ax}")
    names.append(f"{prefix}__crossprod_sign")
    return names


def group_S_features(prep_L: dict, prep_R: dict, rest_thresh: float | None = None,
                     prefix: str = "S") -> dict:
    """Group S — Side (Left/Right/Both).  REQUIRES both devices' synced streams.
    rest_thresh: RMS-ang-vel floor for the active-device count (calibrated or the
    training 20th percentile).  If a device is missing, pass its prep as None and
    the caller flags the window as unsupported for Head S."""
    f = {}
    gL, gR = prep_L["gyro"], prep_R["gyro"]
    eL = float(np.sum(gL ** 2)); eR = float(np.sum(gR ** 2))
    f[f"{prefix}__L_gyro_energy"] = eL
    f[f"{prefix}__R_gyro_energy"] = eR
    eps = 1e-9
    f[f"{prefix}__log_energy_ratio"] = float(np.log((eL + eps) / (eR + eps)))
    rmsL = float(np.sqrt(np.mean(np.sum(gL ** 2, axis=1))))
    rmsR = float(np.sqrt(np.mean(np.sum(gR ** 2, axis=1))))
    f[f"{prefix}__L_rms_angvel"] = rmsL
    f[f"{prefix}__R_rms_angvel"] = rmsR
    # cross-arm correlation of gyro-magnitude streams within +/- SIDE_XCORR_LAG_S
    mL = _norm3(gL) - np.mean(_norm3(gL))
    mR = _norm3(gR) - np.mean(_norm3(gR))
    max_lag = int(SIDE_XCORR_LAG_S * FS_HZ)
    denom = (np.linalg.norm(mL) * np.linalg.norm(mR)) + 1e-12
    best = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = np.dot(mL[lag:], mR[:len(mR) - lag]) if lag < len(mR) else 0.0
        else:
            c = np.dot(mL[:len(mL) + lag], mR[-lag:]) if -lag < len(mL) else 0.0
        best = max(best, c / denom)
    f[f"{prefix}__cross_arm_xcorr"] = float(best)
    if rest_thresh is not None:
        f[f"{prefix}__active_device_count"] = float(int(rmsL > rest_thresh)
                                                    + int(rmsR > rest_thresh))
    return f


def family_P_features(prep: dict, prefix: str = "P") -> dict:
    """Family P — phase/directional, SIGNED, odd under CW<->CCW reversal (§ optional
    P/T/S block).  Symmetric statistics cannot separate mirror-symmetric gestures."""
    accel = prep["linacc"]
    gyro  = prep["gyro"]
    f = {}
    for i, ax in enumerate("xyz"):
        f[f"{prefix}__gyro_signed_mean_{ax}"] = float(np.mean(gyro[:, i]))
    # signed "curl" mean(x*dy - y*dx) via np.gradient, on linear-accel and gyro
    for tag, sig in (("acc", accel), ("gyro", gyro)):
        d = np.gradient(sig, axis=0)
        for (i, j, nm) in ((0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")):
            curl = float(np.mean(sig[:, i] * d[:, j] - sig[:, j] * d[:, i]))
            f[f"{prefix}__curl_{tag}_{nm}"] = curl
    # quadrature cross-spectrum sum(Im(FFT(x)*conj(FFT(y)))) for gyro pairs
    F = [np.fft.rfft(gyro[:, i] - gyro[:, i].mean(), n=NFFT) for i in range(3)]
    for (i, j, nm) in ((0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")):
        f[f"{prefix}__quad_xspec_{nm}"] = float(np.sum(np.imag(F[i] * np.conj(F[j]))))
    # lagged cross-correlation asymmetry + signed peak lag (max lag = int(0.1*fs))
    for (i, j, nm) in ((0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")):
        a = gyro[:, i] - gyro[:, i].mean()
        b = gyro[:, j] - gyro[:, j].mean()
        asym = 0.0; best_lag = 0; best_val = -np.inf
        for lag in range(1, FAMILY_P_MAX_LAG + 1):
            rp = np.dot(a[lag:], b[:len(b) - lag])
            rn = np.dot(a[:len(a) - lag], b[lag:])
            asym += (rp - rn)
            if abs(rp) > best_val:
                best_val, best_lag = abs(rp), lag
        f[f"{prefix}__xcorr_asym_{nm}"]     = float(asym)
        f[f"{prefix}__xcorr_peaklag_{nm}"]  = float(best_lag)
    return f


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — exact-duplicate dedup (spec §2) — applied after windowing
# ══════════════════════════════════════════════════════════════════════════════
def dedup_exact(df_win, feature_cols, protected_prefix_len: int):
    """Drop any feature that is numerically identical to an EARLIER kept feature.
    The first `protected_prefix_len` columns (original 79+66 layout) are never
    dropped.  Returns (kept_cols, dropped_map)."""
    protected = feature_cols[:protected_prefix_len]
    candidates = feature_cols[protected_prefix_len:]
    kept_vals  = {c: df_win[c].values for c in protected}
    kept_order = list(protected)
    dropped, dup_of = [], {}
    for c in candidates:
        v = df_win[c].values
        match = next((kc for kc in kept_order
                      if np.allclose(v, kept_vals[kc], rtol=1e-6, atol=1e-9)), None)
        if match is not None:
            dropped.append(c); dup_of[c] = match
        else:
            kept_vals[c] = v; kept_order.append(c)
    return kept_order, dup_of
