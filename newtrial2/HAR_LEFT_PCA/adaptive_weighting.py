#!/usr/bin/env python3
"""
Adaptive feature-weighting wrapper for the 27-feature / RandomForest IMU classifier.

WHY THIS DESIGN (read before tuning)
------------------------------------
The deployed estimator is a RandomForest.  RF split decisions are threshold tests
on individual features and are INVARIANT to any monotonic per-feature rescaling —
so multiplying a feature value by 1.3x before the forest changes nothing.  Naive
"boost the feature weight into the RF input" is therefore a silent no-op.

We instead route the adaptive multipliers to the two places where they genuinely
change the output, while leaving the RF and its feature set untouched:

  1. POSTERIOR GATING — the multipliers scale the RF's per-class probabilities,
     grouped by activity family, by how well the live window matches that family's
     measured physics signature.  Bounded to [0.5, 1.5] so no family can hijack a
     decision (the same lesson that tamed the 'running' attractor); it can only
     tilt a near-tie.  This is a Bayesian-style prior reweight and stays fully
     interpretable.
  2. DISTANCE-FREE PHYSICS FALLBACK — when the RF is unsure (or the window is
     out-of-envelope) we drop to a one-feature rule per family (gravity angle for
     static, peak energy / gyro_x sign for locomotion, gyro_y sign + gyro_x
     amplitude for rotation), which is maximally robust to noise on other axes.

All thresholds below are set in the GAP between active and inactive classes that
was measured directly on this dataset (see DEFAULT_THRESHOLDS), so they have real
margin and a human can audit every decision from the confidence log.

This module is self-contained: it imports nothing from HAR_4.py (importing that
script would re-run the whole training pipeline).  The fitted `pipe`, the
`LabelEncoder`, the FEATURES list, the energy floor and the two feature
extractors are injected by the caller.
"""
from __future__ import annotations
import math
import re
import numpy as np


# ── ACTIVITY-FAMILY MAP ───────────────────────────────────────────────────────
# Canonical (lower-cased) class names → family.  Unmapped classes get a neutral
# 1.0 multiplier (no gating), so the wrapper degrades safely on a new label set.
#
# The DEPLOYED dataset (ROT_activities_master.csv) is entirely ROTATION classes,
# named {SIDE}_{BODY}_{DIR}: SIDE∈{B,L,R}, BODY∈{HAND,SHOU}, DIR∈{CLK,ACLK}
# (e.g. B_HAND_ACLK, L_SHOU_CLK).  They are recognised by _ROT_LABEL_RE below so
# the wrapper maps them to the "rotation" family.  The old locomotion/static/
# contextual sets are kept only for backward-compatibility with the legacy label
# set — none of them exist in this dataset, so those branches never fire.
_STATIC = {"sitting", "standing"}
_LOCO   = {"walking", "running", "upstairs", "downstairs"}
_ROT    = {"hand_clk_rot", "hand_aclk_rot", "shoulder_clk_rot", "shoulder_aclk_rot"}
_CTX    = {"medicine", "left_drinking", "left_phone_call",
           "right_drinking", "right_phone_call"}
_ROT_LABEL_RE = re.compile(r"^[blr]_(hand|shou)_(a?clk)$")   # new rotation taxonomy


def group_of(cls: str):
    c = str(cls).strip().lower()
    if c in _STATIC: return "static"
    if c in _LOCO:   return "locomotion"
    if c in _ROT:    return "rotation"
    if c in _CTX:    return "contextual"
    if _ROT_LABEL_RE.match(c): return "rotation"     # {B,L,R}_{HAND,SHOU}_{CLK,ACLK}
    return None


def _direction_of(cls: str):
    """CLK / ACLK from a class label, or None if it carries no direction token."""
    c = str(cls).strip().upper()
    if c.endswith("ACLK") or "ACLK" in c: return "ACLK"
    if c.endswith("CLK")  or "CLK"  in c: return "CLK"
    return None


# ── MEASURED-SIGNATURE THRESHOLDS (the gap between active & inactive classes) ──
# Each "_w" is the soft transition width (deg/s or g) for the sigmoid band, so the
# gate is smooth, never a hard on/off switch.
DEFAULT_THRESHOLDS = {
    # STATIC — sitting gyro std ≈1.7, standing ≈3.6; next-quietest class ≈55 → 10x margin
    "static_gyro_std": 5.0,   "static_gyro_w": 1.5,
    "static_amag_std": 0.03,  "static_amag_w": 0.01,   # static 0.005–0.007 vs drinking 0.11
    "static_grav_axis": 0.6,  # standing ax_mean 0.97 / sitting az_mean 0.89
    # LOCOMOTION — walk 0.26, up 0.30, down 0.40, run 1.19; static ~0.006
    "loco_amag_std": 0.15,    "loco_amag_w": 0.05,
    "loco_entropy": 0.63,     "loco_entropy_w": 0.04,   # periodic gait (≤0.60) vs broadband
    "loco_run_amag": 0.7,     # running 1.19 vs ≤0.4 other locomotion
    "loco_stair_gx": 6.0,     # upstairs +11.9 / downstairs −15.6; flat walk ≈+1.6
    # broadband-noise OOD: real classes' accel_mag spectral entropy ≤0.603 (running);
    # white noise ≈0.707 → 0.66 sits in the gap and rejects it regardless of energy.
    "broadband_entropy": 0.66,
    # ROTATION — gyro_y mean ±20..28; locomotion |gy_mean| ≤9
    "rot_gy": 15.0,           "rot_gy_w": 4.0,
    "rot_gx_std": 90.0,       "rot_gx_w": 15.0,
    "rot_hand_gx_std": 85.0,  # ROT data: HAND gx_std≈105 vs SHOU≈65 → 85 splits them
    # DIRECTION cue reliability gate (measured on ROT_activities_master.csv):
    # CLK → gy_mean POSITIVE, ACLK → NEGATIVE, but the sign is only clean for the
    # both-hand rotations (|gy_mean|≈60 for HAND, ≈30 for SHOU).  Left-hand rotations
    # median near 0 but INDIVIDUAL windows transiently reach ~28, so a low gate makes
    # the cue misfire on them.  Pin the gate at 40 (comfortably above the left-hand
    # window tail, at/below the both-hand HAND level) so the cue fires only on
    # unambiguous strong rotations; everything else defers to the RandomForest.
    # (The legacy code used the OPPOSITE sign convention and a single hard rule.)
    "rot_gy_strong": 40.0,
    # CONTEXTUAL/medicine — single-axis dominance: gx_std≈125, gy/gz≈14–15 → ratio≈8
    "ctx_gx_lo": 100.0, "ctx_gx_hi": 150.0, "ctx_gx_w": 12.0,
    "ctx_ratio": 3.0,   "ctx_ratio_w": 0.8,
}


# ── SOFT GATES (smooth membership in [0,1]) ───────────────────────────────────
def _sig_above(x, thr, w): return 1.0 / (1.0 + math.exp(-(x - thr) / max(w, 1e-9)))
def _sig_below(x, thr, w): return 1.0 / (1.0 + math.exp(-(thr - x) / max(w, 1e-9)))
def _sig_band(x, lo, hi, w): return _sig_above(x, lo, w) * _sig_below(x, hi, w)
def _gmean(*vals): return float(np.prod(vals) ** (1.0 / len(vals)))


class AdaptiveWeightedClassifier:
    """Wraps a fitted sklearn pipeline + LabelEncoder with physics-aware gating,
    a low-confidence fallback, and per-prediction confidence diagnostics."""

    def __init__(self, pipe, le, features, energy_floor,
                 extract_stats, extract_freq_stats,
                 raw_axes=("accel_x", "accel_y", "accel_z",
                           "gyro_x", "gyro_y", "gyro_z"),
                 reject=0.65, align_floor=0.35, clip_limit=None,
                 thresholds=None):
        self.pipe, self.le, self.F = pipe, le, list(features)
        self.energy_floor = float(energy_floor)
        self._extract_stats = extract_stats
        self._extract_freq  = extract_freq_stats
        self.raw_axes = list(raw_axes)
        self.reject = float(reject)
        self.align_floor = float(align_floor)
        self.clip_limit = clip_limit
        self.T = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
        self.groups = [group_of(c) for c in self.le.classes_]
        # Groups that actually have classes in THIS model — decisions/logging are
        # restricted to these so a spurious 'locomotion' alignment on a rotation-only
        # dataset can never win the group vote or drive the fallback.
        self._active_groups = {g for g in self.groups if g is not None}
        self._cls_index = {c: i for i, c in enumerate(self.le.classes_)}

    # ── 1. cheap interpretable signatures straight off the raw window ──────────
    def signatures(self, w: np.ndarray) -> dict:
        w = np.asarray(w, dtype=np.float64)
        ax, ay, az = w[:, 0], w[:, 1], w[:, 2]
        gx, gy, gz = w[:, 3], w[:, 4], w[:, 5]
        amag = np.sqrt(ax * ax + ay * ay + az * az)
        f = self._extract_freq(w)               # exact training FFT path (NFFT=128)
        return {
            "ax_mean": float(ax.mean()), "az_mean": float(az.mean()),
            "gx_mean": float(gx.mean()), "gy_mean": float(gy.mean()),
            "gx_std": float(gx.std()), "gy_std": float(gy.std()),
            "gz_std": float(gz.std()), "gx_range": float(np.ptp(gx)),
            "amag_mean": float(amag.mean()), "amag_std": float(amag.std()),
            "amag_fft_energy": float(f["accel_mag__fft_energy"]),
            "amag_fft_dommag": float(f["accel_mag__fft_dommag"]),
            "amag_fft_entropy": float(f["accel_mag__fft_entropy"]),
            "accel_absmax": float(np.max(np.abs(w[:, :3]))),
        }

    # ── §2 group alignment scores A_g ∈ [0,1] (geometric mean → AND semantics) ─
    def alignments(self, s: dict) -> dict:
        T = self.T
        a_static = _gmean(
            _sig_below(max(s["gx_std"], s["gy_std"], s["gz_std"]),
                       T["static_gyro_std"], T["static_gyro_w"]),
            _sig_below(s["amag_std"], T["static_amag_std"], T["static_amag_w"]))
        # locomotion = motion present  AND  periodic (low spectral entropy)  AND
        # NOT a wrist rotation (|gyro_y mean| stays small; rotations sit at ±20..28).
        a_loco = _gmean(
            _sig_above(s["amag_std"], T["loco_amag_std"], T["loco_amag_w"]),
            _sig_below(s["amag_fft_entropy"], T["loco_entropy"], T["loco_entropy_w"]),
            _sig_below(abs(s["gy_mean"]), T["rot_gy"], T["rot_gy_w"]))
        a_rot = _gmean(
            _sig_above(abs(s["gy_mean"]), T["rot_gy"], T["rot_gy_w"]),
            _sig_above(s["gx_std"], T["rot_gx_std"], T["rot_gx_w"]))
        ratio = s["gx_std"] / max(s["gy_std"], s["gz_std"], 1e-3)
        a_ctx = _gmean(
            _sig_band(s["gx_std"], T["ctx_gx_lo"], T["ctx_gx_hi"], T["ctx_gx_w"]),
            _sig_above(ratio, T["ctx_ratio"], T["ctx_ratio_w"]))
        return {"static": a_static, "locomotion": a_loco,
                "rotation": a_rot, "contextual": a_ctx}

    def multipliers(self, A: dict) -> dict:
        # A_g=0 → 0.5x, A_g=1 → 1.5x  (smooth, bounded; your requested range)
        return {g: 0.5 + A[g] for g in A}

    # ── §1 global out-of-envelope gate ────────────────────────────────────────
    def ood_flags(self, s: dict) -> list:
        flags = []
        if not all(np.isfinite(v) for v in s.values()):
            flags.append("nan_stat")
        if s["amag_mean"] < self.energy_floor:
            flags.append("low_energy")
        if s["amag_fft_entropy"] > self.T["broadband_entropy"]:
            flags.append("broadband_noise")   # flatter spectrum than any real activity
        if self.clip_limit is not None and s["accel_absmax"] >= self.clip_limit:
            flags.append("accel_clip")
        return flags

    # ── §3 low-confidence physics fallback (only ever returns a REAL model class) ─
    # Contract: this NEVER invents a label outside self.le.classes_.  It re-ranks the
    # RandomForest posterior with a physics cue only where that cue is measured to be
    # reliable, and otherwise defers to the forest's own best guess.
    def _fallback(self, s: dict, A: dict, p: np.ndarray):
        T = self.T
        classes = self.le.classes_
        order   = np.argsort(p)[::-1]            # class indices, best RF prob first

        # Direction cue (CLK↔gy>0, ACLK↔gy<0) — trusted ONLY for strong |gy_mean|,
        # which on this dataset means the both-hand rotations; left-hand rotations
        # sit near gy≈0 where the sign is noise, so we fall through to the RF there.
        if abs(s["gy_mean"]) >= T["rot_gy_strong"]:
            want = "CLK" if s["gy_mean"] > 0 else "ACLK"
            for i in order:
                if _direction_of(classes[i]) == want:
                    return classes[int(i)], (
                        f"rotation:dir={want}(gy_mean={s['gy_mean']:.0f}) + RF-rank")
        # Weak/absent direction cue → trust the forest's top valid class.
        return classes[int(order[0])], "rf-argmax (no reliable physics cue)"

    # ── full prediction: gating → fallback/reject → log ───────────────────────
    def predict(self, w: np.ndarray):
        s   = self.signatures(w)
        ood = self.ood_flags(s)
        A   = self.alignments(s)
        m   = self.multipliers(A)

        # base RF posterior over the UNCHANGED feature set
        row  = self._extract_stats(np.asarray(w, dtype=np.float32))
        xvec = np.array([row[f] for f in self.F], dtype=np.float32).reshape(1, -1)
        p    = self.pipe.predict_proba(xvec)[0]

        # §2 posterior gating (neutral 1.0 for unmapped classes)
        gmul = np.array([m.get(g, 1.0) if g else 1.0 for g in self.groups])
        pg   = p * gmul
        pg   = pg / pg.sum()

        base_top  = int(p.argmax())
        gated_top = int(pg.argmax())
        # Restrict the group vote to families that actually have classes in THIS
        # model, so a spurious 'locomotion'/'static' alignment on a rotation-only
        # dataset can't win the vote, drive the fallback, or mislabel the log.
        A_active  = {g: A[g] for g in A if g in self._active_groups} or A
        g_star    = max(A_active, key=A_active.get)
        max_align = max(A_active.values())
        final_conf = float(math.sqrt(max(A_active[g_star], 0.0) * float(pg[gated_top])))

        up   = sorted([g for g in m if m[g] > 1.05], key=lambda g: -m[g])
        down = sorted([g for g in m if m[g] < 0.95], key=lambda g:  m[g])
        log = {
            "raw_stats": {k: round(v, 4) for k, v in s.items()},
            "group_alignment": {g: round(A[g], 3) for g in A},
            "multipliers":     {g: round(m[g], 3) for g in m},
            "base":  {"top": self.le.classes_[base_top],  "conf": round(float(p[base_top]), 3)},
            "gated": {"top": self.le.classes_[gated_top], "conf": round(float(pg[gated_top]), 3)},
            "final_confidence": round(final_conf, 3),
            "winning_group": g_star,
            "ood_flags": ood,
        }

        if ood:
            log["mode"] = "reject"; log["fallback_rule"] = None
            log["deviation_reason"] = "out-of-envelope: " + ",".join(ood)
            return "uncertain", final_conf, log

        if float(pg[gated_top]) < self.reject or max_align < self.align_floor:
            label, rule = self._fallback(s, A, pg)
            log["mode"] = "fallback"; log["fallback_rule"] = rule
            log["deviation_reason"] = (
                f"low gated conf ({pg[gated_top]:.2f}) or weak alignment "
                f"(maxA={max_align:.2f}) → {g_star} rule '{rule}'")
            return label, final_conf, log

        log["mode"] = "normal"; log["fallback_rule"] = None
        log["deviation_reason"] = (
            ("up=" + ",".join(f"{g}×{m[g]:.2f}" for g in up) if up else "")
            + ("  down=" + ",".join(f"{g}×{m[g]:.2f}" for g in down) if down else "")
        ).strip() or "no family strongly preferred"
        return self.le.classes_[gated_top], final_conf, log

    # ── pretty one-prediction diagnostic block ────────────────────────────────
    @staticmethod
    def format_log(log: dict) -> str:
        A = log["group_alignment"]; m = log["multipliers"]
        lines = [
            f"    mode={log['mode']:<8} final_conf={log['final_confidence']:.2f}"
            f"  win-group={log['winning_group']}",
            "    align  " + "  ".join(f"{g[:4]}={A[g]:.2f}" for g in A),
            "    mult   " + "  ".join(f"{g[:4]}×{m[g]:.2f}" for g in m),
            f"    base={log['base']['top']}({log['base']['conf']:.2f}) "
            f"→ gated={log['gated']['top']}({log['gated']['conf']:.2f})",
        ]
        if log["fallback_rule"]:
            lines.append(f"    fallback-rule: {log['fallback_rule']}")
        if log["ood_flags"]:
            lines.append(f"    OOD: {','.join(log['ood_flags'])}")
        lines.append(f"    why: {log['deviation_reason']}")
        return "\n".join(lines)
