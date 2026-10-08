#!/usr/bin/env python3
"""
Directional / location feature engineering for the four rotation gestures that the
Random Forest cannot tell apart:

    Hand_CLK_ROT   Hand_ACLK_ROT   Shoulder_CLK_ROT   Shoulder_ACLK_ROT

WHY THE CURRENT FEATURES FAIL
-----------------------------
Clockwise and anti-clockwise versions of the SAME gesture have identical amplitude
and frequency profiles — they are time-/sign-reflections of each other.  Every
symmetric statistic the pipeline uses (mean of |accel|, std, max, min, |FFT|,
energy) is INVARIANT to that reflection, so CW and CCW collapse onto the same point.
Rotation DIRECTION lives entirely in the *sign of the cross-axis phase relationship*
— features that are ODD under rotation reversal.  Sensor LOCATION (shoulder vs
hand) lives in *inertia / radius*: a shoulder turn swings a long, heavy lever (large
linear acceleration per rad/s, low jerk); a wrist turn is fast and low-inertia (high
jerk per rad/s).

This script builds three feature families and shows the new ones rescue the 4-way
split:

  FAMILY P  (phase / directional — SIGNED, odd under CW<->CCW):
     • signed mean angular velocity per gyro axis            (raw direction)
     • signed swept-area "curl"  mean(a_i * da_j - a_j * da_i) for axis pairs
       (the z-component of r x dr/dt — sign = sense of rotation in that plane)
     • quadrature cross-spectrum  sum Im(FFT(a_i) * conj(FFT(a_j)))
       (sign = which axis leads in phase = direction)
     • lagged cross-correlation asymmetry + signed peak-lag for gyro pairs
  FAMILY T  (torque / inertia / location — distinguishes shoulder vs hand):
     • jerk (d/dt linear-accel) and angular-acceleration (d/dt gyro) rms & max
     • radius proxy = rms(linear-accel) / rms(gyro)   (large lever => large)
     • gyro dominant frequency, jerk : angular-accel ratio
  FAMILY S  (static baseline — the symmetric features that currently fail):
     • mean/std/max/min of accel_mag & gyro_mag, per-axis std, |FFT| energy

Step 1 (gravity / linear-accel split via Butterworth), step 2 (windowed phase
features), step 3 (derivative/torque features) and step 4 (feature importance on
exactly these 4 classes, session-grouped) are all below.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

CLASSES   = ["Hand_CLK_ROT", "Hand_ACLK_ROT", "Shoulder_CLK_ROT", "Shoulder_ACLK_ROT"]
ACCEL     = ["accel_x", "accel_y", "accel_z"]
GYRO      = ["gyro_x",  "gyro_y",  "gyro_z"]
AXIS_PAIRS = [(0, 1), (0, 2), (1, 2)]          # xy, xz, yz
PAIR_NAME  = {(0, 1): "xy", (0, 2): "xz", (1, 2): "yz"}


# ── 1. GRAVITY / LINEAR-ACCELERATION SPLIT (Butterworth) ─────────────────────
def gravity_linear_split(acc, fs, cutoff=0.3, order=3):
    """Low-pass the accelerometer to estimate gravity; the high-pass remainder is
    the body/linear acceleration.  acc: (n,3).  Returns (gravity, linear)."""
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype="low")
    padlen = min(len(acc) - 1, 3 * max(len(a), len(b)))
    grav = np.column_stack([filtfilt(b, a, acc[:, i], padlen=padlen) for i in range(3)])
    return grav, acc - grav


# ── phase-feature helpers (the directional core) ─────────────────────────────
def signed_curl(x, y):
    """Mean signed swept-area rate  x*dy - y*dx.  For motion tracing a loop in the
    (x,y) plane this is twice the enclosed-area rate; its SIGN is the sense of
    rotation (CW vs CCW) and flips under time reversal."""
    dx = np.gradient(x); dy = np.gradient(y)
    return float(np.mean(x * dy - y * dx))


def quad_cross_spectrum(x, y):
    """Sum of the imaginary part of the cross-spectrum  Im(X . conj(Y)).  Nonzero
    only when x and y are out of phase; its sign tells which axis leads => direction."""
    X = np.fft.rfft(x - x.mean()); Y = np.fft.rfft(y - y.mean())
    return float(np.imag(X * np.conj(Y)).sum())


def xcorr_dir(x, y, max_lag):
    """Lagged cross-correlation directionality between two axes.
    Returns (antisymmetric integral, signed peak lag).
      • antisym = sum_{t>0} R(+t) - R(-t)   (0 for a symmetric/non-rotating pair)
      • peak lag sign = whether y leads or lags x  => rotation sense."""
    x = x - x.mean(); y = y - y.mean()
    denom = np.std(x) * np.std(y) * len(x) + 1e-12
    r = np.correlate(x, y, mode="full") / denom        # length 2n-1, center = lag 0
    c = len(x) - 1
    lags = np.arange(1, min(max_lag, c) + 1)
    antisym = float(np.sum(r[c + lags] - r[c - lags]))
    full_lags = np.arange(-c, c + 1)
    peak_lag = float(full_lags[np.argmax(np.abs(r))])
    return antisym, peak_lag


def dom_freq(x, fs):
    X = np.abs(np.fft.rfft(x - x.mean()))
    f = np.fft.rfftfreq(len(x), d=1.0 / fs)
    return float(f[np.argmax(X[1:]) + 1]) if len(X) > 1 else 0.0


# ── 2+3. PER-WINDOW FEATURE EXTRACTION ───────────────────────────────────────
def window_features(acc, gyro, fs):
    """acc, gyro : (n,3) raw window.  Returns a dict of P/T/S features."""
    grav, lin = gravity_linear_split(acc, fs)
    max_lag = max(2, int(0.1 * fs))                    # search +/- 100 ms of phase lag
    f = {}

    # ── FAMILY P — phase / directional (SIGNED) ──────────────────────────────
    # raw signed angular velocity: its sign is the rotation direction
    for i, nm in enumerate(GYRO):
        f[f"P_signed_mean_{nm}"] = float(gyro[:, i].mean())
    # signed swept-area curl on linear-accel and on gyro, per plane
    for (i, j) in AXIS_PAIRS:
        p = PAIR_NAME[(i, j)]
        f[f"P_curl_lin_{p}"]  = signed_curl(lin[:, i],  lin[:, j])
        f[f"P_curl_gyro_{p}"] = signed_curl(gyro[:, i], gyro[:, j])
        f[f"P_quad_gyro_{p}"] = quad_cross_spectrum(gyro[:, i], gyro[:, j])
        asym, plag = xcorr_dir(gyro[:, i], gyro[:, j], max_lag)
        f[f"P_xcorrAsym_gyro_{p}"] = asym
        f[f"P_peakLag_gyro_{p}"]   = plag

    # ── FAMILY T — torque / inertia / location ───────────────────────────────
    jerk    = np.diff(lin,  axis=0) * fs               # d(linear accel)/dt
    ang_acc = np.diff(gyro, axis=0) * fs               # d(omega)/dt
    jerk_mag = np.linalg.norm(jerk, axis=1)
    aacc_mag = np.linalg.norm(ang_acc, axis=1)
    lin_mag  = np.linalg.norm(lin, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    f["T_jerk_rms"]      = float(np.sqrt(np.mean(jerk_mag ** 2)))
    f["T_jerk_max"]      = float(jerk_mag.max())
    f["T_angacc_rms"]    = float(np.sqrt(np.mean(aacc_mag ** 2)))
    f["T_angacc_max"]    = float(aacc_mag.max())
    f["T_radius_proxy"]  = float(np.sqrt(np.mean(lin_mag ** 2)) /
                                 (np.sqrt(np.mean(gyro_mag ** 2)) + 1e-9))
    f["T_jerk_angacc_ratio"] = float(f["T_jerk_rms"] / (f["T_angacc_rms"] + 1e-9))
    f["T_gyro_domfreq"]  = dom_freq(gyro_mag, fs)

    # ── FAMILY S — static symmetric baseline (the failing set) ───────────────
    for nm, sig in (("accel_mag", lin_mag), ("gyro_mag", gyro_mag)):
        f[f"S_{nm}_mean"] = float(sig.mean())
        f[f"S_{nm}_std"]  = float(sig.std())
        f[f"S_{nm}_max"]  = float(sig.max())
        f[f"S_{nm}_min"]  = float(sig.min())
    for i, nm in enumerate(ACCEL):
        f[f"S_std_{nm}"] = float(lin[:, i].std())
    for i, nm in enumerate(GYRO):
        f[f"S_std_{nm}"] = float(gyro[:, i].std())
    f["S_accel_fft_energy"] = float((np.abs(np.fft.rfft(lin_mag - lin_mag.mean())) ** 2).sum())
    f["S_gyro_fft_energy"]  = float((np.abs(np.fft.rfft(gyro_mag - gyro_mag.mean())) ** 2).sum())
    return f


def build_dataset(csv_path, fs, win_sec=1.0, step_sec=0.5, hand="LEFT"):
    df = pd.read_csv(csv_path)
    df = df[df["activity_label"].isin(CLASSES)]
    df = df[df["device_id"].astype(str).str.upper().str.strip() == hand]
    # consistent sensor frame: a single wrist (CW/CCW sign is wrist-frame dependent)
    df = df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)
    W, S = int(win_sec * fs), int(step_sec * fs)
    rows, labels, groups = [], [], []
    for sid, g in df.groupby("session_id", sort=False):
        acc = g[ACCEL].to_numpy(float); gyr = g[GYRO].to_numpy(float)
        lab = g["activity_label"].to_numpy()
        for s in range(0, len(g) - W + 1, S):
            e = s + W
            vals, cnts = np.unique(lab[s:e], return_counts=True)
            rows.append(window_features(acc[s:e], gyr[s:e], fs))
            labels.append(vals[cnts.argmax()])
            groups.append(sid)
    return pd.DataFrame(rows), np.array(labels), np.array(groups)


# ── 4. EVALUATE — feature importance on the 4 problematic classes ────────────
def rf():
    return RandomForestClassifier(n_estimators=300, max_depth=10,
                                  min_samples_leaf=3, max_features="sqrt",
                                  class_weight="balanced", random_state=42, n_jobs=-1)


def grouped_cv_eval(Xdf, y, groups, cols, title, n_splits=4):
    X = Xdf[cols].to_numpy(float)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    yp = cross_val_predict(rf(), X, y, groups=groups, cv=cv, n_jobs=-1)
    acc = accuracy_score(y, yp)
    print(f"  {title:<32} feats={len(cols):3d}  session-grouped CV acc = {acc:.3f}")
    return acc, yp


def main():
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    csv = next((p for p in (os.path.join(here, "ALL_activities_master.csv"),
                            os.path.join(here, "Production_IMU_Dataset.csv"))
                if os.path.exists(p)), None)
    if csv is None:
        sys.exit("dataset not found in this folder")

    # auto-detect sampling rate from the timestamps (this dataset is ~250 Hz)
    fs = 250.0
    try:
        t = pd.to_datetime(pd.read_csv(csv, usecols=["timestamp"])["timestamp"].iloc[:2000])
        dt = t.diff().dt.total_seconds().median()
        if dt and dt > 0:
            fs = round(1.0 / dt)
    except Exception:
        pass
    print("=" * 74)
    print(f"DIRECTIONAL FEATURE STUDY — 4 rotation classes  (fs≈{fs:.0f} Hz, LEFT wrist)")
    print("=" * 74)

    Xdf, y, groups = build_dataset(csv, fs)
    print(f"windows: {len(y)}  | per class: "
          + ", ".join(f"{c}={int((y==c).sum())}" for c in CLASSES))

    P = [c for c in Xdf.columns if c.startswith("P_")]
    T = [c for c in Xdf.columns if c.startswith("T_")]
    S = [c for c in Xdf.columns if c.startswith("S_")]

    # ── 4-way comparison: baseline vs the new families ───────────────────────
    print("\n[4] Session-grouped 4-class accuracy by feature family:")
    grouped_cv_eval(Xdf, y, groups, S,         "S  static baseline (symmetric)")
    grouped_cv_eval(Xdf, y, groups, P,         "P  phase / directional")
    grouped_cv_eval(Xdf, y, groups, T,         "T  torque / location")
    grouped_cv_eval(Xdf, y, groups, P + T,     "P+T directional + location")
    _, yp_all = grouped_cv_eval(Xdf, y, groups, P + T + S, "P+T+S all")

    # confusion matrix for the all-feature model
    cm = confusion_matrix(y, yp_all, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=[c.replace("_ROT", "") for c in CLASSES]
                           ).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
    ax.set_title("4-class confusion (P+T+S, session-grouped CV)")
    fig.tight_layout(); fig.savefig(os.path.join(here, "rotation_confusion.png"), dpi=130)
    plt.close(fig)

    # ── the two binary questions the families are meant to answer ────────────
    dir_lbl = np.where(np.isin(y, ["Hand_CLK_ROT", "Shoulder_CLK_ROT"]), "CW", "CCW")
    loc_lbl = np.where(np.isin(y, ["Hand_CLK_ROT", "Hand_ACLK_ROT"]), "HAND", "SHOULDER")
    print("\n  Direction-only (CW vs CCW):")
    grouped_cv_eval(Xdf, dir_lbl, groups, S, "    S static")
    grouped_cv_eval(Xdf, dir_lbl, groups, P, "    P directional")
    print("  Location-only (HAND vs SHOULDER):")
    grouped_cv_eval(Xdf, loc_lbl, groups, S, "    S static")
    grouped_cv_eval(Xdf, loc_lbl, groups, T, "    T torque/location")

    # ── feature importance of the NEW (P+T) features on the 4 classes ────────
    model = rf().fit(Xdf[P + T].to_numpy(float), y)
    imp = pd.Series(model.feature_importances_, index=P + T).sort_values(ascending=False)
    print("\n[4] Top directional/location features (RF importance, 4-class fit):")
    for name, v in imp.head(15).items():
        fam = {"P": "phase", "T": "torque"}[name[0]]
        print(f"      {v:.4f}  [{fam:6}] {name}")
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.head(15)[::-1].plot.barh(ax=ax, color="teal")
    ax.set_title("Top phase/location feature importances (4 rotation classes)")
    ax.set_xlabel("RF importance")
    fig.tight_layout(); fig.savefig(os.path.join(here, "rotation_feature_importance.png"), dpi=130)
    plt.close(fig)
    print("\nSaved: rotation_confusion.png, rotation_feature_importance.png")
    print("=" * 74)


if __name__ == "__main__":
    main()
