# readme_summary.md — Implementation of the readme.md pipeline in HAR_4_PCA.py

**Date:** 2026-07-11
**Deliverable:** [HAR_4_PCA.py](HAR_4_PCA.py) — complete implementation of every stage,
parameter and methodology in [readme.md](readme.md), evaluated with the **held-out
participant ID methodology**. The previous 4,500-line pipeline is preserved untouched as
[HAR_4_PCA_legacy.py](HAR_4_PCA_legacy.py).

---

## 1. Headline result (honest, leak-free)

| Metric | Legacy pipeline (before) | New readme pipeline (after) |
|---|---|---|
| Evaluation protocol | held-out **session** (participants leak across train/test) | held-out **participant** (leave-one-participant-out) |
| Classes | 8 (R_* classes dropped) | **12** (full readme class set) |
| Signal preprocessing | none (raw axes straight into windowing) | Stage 0–3 filter stack (resample → median → Butterworth → SavGol → ISIF-BU) |
| Honest test accuracy | **0.294** (run_hybrid_k80.log; deployed hybrid selector) | **0.677** |
| Honest macro-F1 | ~0.22 | **0.687** |
| Train accuracy (overfit gap) | 0.970 (gap ≈ 0.68) | ~0.999 (gap ≈ 0.32) |
| Chance level | 0.125 | 0.083 |

The new pipeline more than **doubles the honest accuracy while solving a strictly harder
problem** (12 classes instead of 8, and a stricter split: the legacy protocol only held out
sessions, so the same person's other sessions were always in training; the new protocol
never lets the model see the test participant at all).

**Diagnostic contrast:** a naive random window split on the same features scores **0.998** —
that number is meaningless (overlapping windows + same-participant data leak across the
split) and is printed only to show how much leakage inflates accuracy. This is exactly why
the readme and the task brief demand group-aware splitting.

### Per-fold results (leave-one-participant-out)

| Held-out participant | n windows | test acc | macro-F1 |
|---|---|---|---|
| 1 | 1023 | 0.855 | 0.861 |
| 2 | 1285 | 0.702 | 0.673 |
| 3 | 904 | 0.506 | 0.432 |
| 4 | 896 | 0.770 | 0.749 |
| 5 | 651 | 0.731 | 0.557 |
| 6 | 266 | **0.000** | 0.000 |
| **Aggregate** | **5025** | **0.677** | **0.687** |

Excluding the anomalous participant 6 (see §5), aggregate accuracy is **0.715**.

---

## 2. Explicit confirmation: held-out participant ID methodology

- Windows are cut **per session**; sessions belong to exactly one participant, so no window
  ever mixes participants.
- Splitting uses `GroupKFold(n_splits=6)` with `groups = participant_id` — each fold holds
  out **one entire participant**; the model in that fold never sees a single sample,
  window, or session from them.
- The feature-importance pruner (top-70 selection) is a transformer **inside the sklearn
  Pipeline**, so it is **refit from scratch on each fold's training participants only** —
  no selection-on-test leakage.
- The headline 0.677 aggregates the out-of-fold predictions: every one of the 5,025 windows
  was predicted by a model that never saw its participant.
- All preprocessing (Stages 0–3) is per-session, per-device signal processing with fixed
  (non-learned) parameters, so it cannot leak label or participant information.

---

## 3. Stage-by-stage mapping (readme requirement → implementation)

| readme | Requirement | Implementation in HAR_4_PCA.py |
|---|---|---|
| §0 | 12 classes, not 8 | R_* drop removed; all 12 `{B,L,R}_{SHOU,HAND}_{CLK,ACLK}` kept |
| §0 | mag_x/y/z dead → drop | mag columns never loaded (`USECOLS`) |
| §0 | irregular timestamps → must resample | Stage 0 `resample_session()` |
| Stage 0 | uniform 50 Hz grid (Δt = 20 ms), linear interp, LEFT/RIGHT re-aligned on the same grid | per session: grid over the overlap of both devices' time spans; `np.interp` per channel; cross-device features are therefore timestamp-matched |
| Stage 1 | median despike, kernel 3, per axis per device | `scipy.signal.medfilt(x, 3)` in `despike_denoise()` |
| Stage 2a | Butterworth order 4, zero-phase `filtfilt`, 10 Hz accel / 15 Hz gyro | `butter(4, fc/(fs/2)); filtfilt(b, a, x)` — spikes removed *before* the IIR filter, per readme §4 |
| Stage 2b | Savitzky-Golay win 7 / polyorder 2, `deriv=1`, jerk & angular accel **only** (side branch) | `sg_derivative()`: `savgol_filter(x, 7, 2, deriv=1, delta=1/fs)` on the Butterworth output; feeds features only, never replaces Stage 2a |
| Stage 3 | **ISIF-BU** primary attitude estimator: quaternion + gyro-bias state, 6-DOF (no mag), saturation g(k)=2(1+e^{−b(k−c)})^{−1}−1 with b=2.1, c=0, boundary layer Δ=0.5·R_t, chi-square gate σ=0.05 / μ=2, QR/SVD not Cholesky, accel normalised to unit gravity, dead mag channels *not* fed in, state carried continuously across the session | `isif_bu_attitude()`: 7-state [q0..q3, bgx..bgz]; strap-down quaternion prediction; SIF sliding-mode update with the readme's saturation function; Bayesian (Kalman) refinement gated by χ²(0.95, 2)=5.99; **all inversions via SVD, SIF gain via SVD pseudo-inverse — no Cholesky anywhere**; one continuous run per session per device |
| Stage 3 | Madgwick fallback, β ≈ 0.041 | `madgwick_attitude()`, selectable via `ATTITUDE_FILTER=madgwick` |
| Stage 4 | 100 samples / 2.0 s window, 50-sample stride (50 % overlap); ≥50 % label coverage, discard mixed windows | `WINDOW_SIZE=100, STEP_SIZE=50`; sessions are single-activity (verified), so no mixed window can occur; windows never span sessions |
| Stage 5 | time-domain stats per channel (accel, gyro, jerk, ang-accel, roll/pitch/yaw): mean std min max range RMS MAD skew kurt ZCR + SMA triads | 15 channels × 10 stats + 2 SMA per device |
| Stage 5 | frequency-domain: FFT on the 100-sample window (0.5 Hz bins): dominant freq, spectral energy, spectral entropy, band energy 0–3 / 3–8 / 8–15 Hz | 6 spectral features × 15 channels per device |
| Stage 5 | orientation-derived: net rotation angle yaw(end)−yaw(0) (sign = CLK/ACLK), range/std of roll/pitch/yaw, signed peak angular velocity | `net_roll/net_pitch/net_yaw` on the session-unwrapped ISIF-BU trace; range/std covered by the time stats on roll/pitch/yaw; `peak_angvel_x/y/z` (signed) + `peak_gyro_mag` |
| Stage 5 | cross-axis correlations; cross-device LEFT↔RIGHT correlation (Shoulder-vs-Hand discriminator); amplitude ratios ‖·‖_L/‖·‖_R | 6 within-device axis-pair correlations; 8 LEFT-vs-RIGHT channel/magnitude correlations + 2 amplitude ratios |
| Stage 5 | ~230+ raw features → prune to top 60–80 by RF importance | 520 raw (LEFT + RIGHT + cross-device) → `RFImportanceTopK(k=70)` **inside the pipeline** (leak-safe, refit per fold); readme's ~230 estimate counted one device — concatenating LEFT+RIGHT per its own Stage-5 instruction doubles it |
| Stage 6 | RF: 18 trees, max_depth 8, min_samples_leaf 2, min_samples_split 4, max_features='sqrt' | exactly these values, unchanged |
| Stage 6 | group CV by participant, not random rows | leave-one-participant-out `GroupKFold` (§2 above) |
| Stage 6 | class-balance check **after** windowing; `class_weight='balanced'` if skewed | window counts range 207–496 (max/min = 2.40 > 1.5) → `class_weight='balanced'` applied automatically |

Nothing in the readme was skipped. Two things the readme *states as fact* differ from the
data on disk and were handled explicitly:

1. **Dataset:** `ROT_activities_master.csv` today has 487,794 rows / 120 sessions / 11
   classes (no `B_HAND_ACLK`). The file matching the readme's description (65 sessions,
   12 classes) is `ROT_activities_master2.csv`. The loader takes the **union de-duplicated
   by session** (125 sessions, all 12 classes, 6 participants) — master.csv wins for the
   60 duplicated sessions.
2. **ISIF-BU paper:** the referenced paper is not in the repo, so the filter was
   implemented from the readme's own complete specification (state, models, saturation
   function, boundary layer, χ² gate, decomposition requirements, reset policy).

---

## 4. Why accuracy improved (rationale)

- **Participant-grouped evaluation aside, the biggest modeling win is the filter stack.**
  The legacy pipeline fed raw, irregularly-sampled, spiky BLE data straight into feature
  extraction; every FFT-based feature silently assumed a 20 ms grid that didn't exist.
  Resampling (Stage 0) makes the spectral features truthful; median+Butterworth (1, 2a)
  remove burst glitches and out-of-band noise before they contaminate window statistics.
- **The ISIF-BU orientation trace adds the single most discriminative signal family.** Net
  yaw change per window directly encodes CLK vs ACLK (the readme's "close to a
  single-feature discriminator"), and roll/pitch statistics encode arm posture
  (shoulder vs hand). These features dominate the deployed top-70
  (see plots/readme_feature_importance.png).
- **Cross-device features separate B_* from single-hand classes:** shoulder rotations move
  both wrists coherently (high LEFT↔RIGHT correlation), hand rotations don't.
- **Leak-safe importance pruning (520 → 70)** keeps the shallow 18-tree forest from
  drowning in noisy columns, per the readme's Stage-6 sizing argument.

## 5. Honest caveats

- **Participant 6 scored 0.000 — this is a data-collection defect, not randomness.** The
  prediction cross-tab shows a perfect mirror pattern: `L_HAND_ACLK`→`R_HAND_CLK` (68/70),
  `L_HAND_CLK`→`R_HAND_ACLK` (32/34), `R_HAND_CLK`→`L_HAND_ACLK` (38/46),
  `B_SHOU_ACLK`→`B_SHOU_CLK` (36/40). Side and direction are *both* inverted — the exact
  signature of the LEFT/RIGHT devices being **worn on swapped wrists** for this
  participant. The model recognizes the motions; the labels are mirrored. If the collection
  notes confirm the swap, relabeling participant 6 would raise the aggregate to roughly
  0.72–0.75. I did **not** silently relabel the data — that needs confirmation from
  whoever ran the collection.
- **The L_SHOU_CLK / L_SHOU_ACLK pair is the weakest (F1 ≈ 0.34)**, mostly confused with
  each other and with B_SHOU_*. Left-shoulder rotation is covered by only 4 of 6
  participants, so each LOPO fold trains it from as few as 3 people.
- **Per-fold variance is large (0.51–0.86 excluding P6).** With 6 participants, each fold's
  test set is one person's style. More participants — not more features — is what moves
  this next, and the train accuracy near 1.0 says within-participant memorization is still
  happening (expected for an 18-tree forest on 70 features; the honest metric is unaffected
  because of the grouping).
- The 0.998 random-split number demonstrates the leakage the old row/window splits allow;
  never report it as model quality.

## 6. Files & how to run

| File | Role |
|---|---|
| [HAR_4_PCA.py](HAR_4_PCA.py) | the new readme pipeline (this deliverable) |
| [HAR_4_PCA_legacy.py](HAR_4_PCA_legacy.py) | byte-exact backup of the previous pipeline |
| generated/har_readme_model.pkl | deployed pipeline (pruner + RF) fit on all participants |
| generated/har_readme_metrics.json | all metrics reported above, machine-readable |
| generated/har_readme_features.json | 520 raw + 70 deployed feature names |
| generated/preproc_cache.npz | Stage 0–3 cache (auto-invalidated on config/data change) |
| plots/lopo_confusion_matrix.png | 12-class confusion matrix, held-out-participant |
| plots/lopo_fold_accuracy.png | per-participant fold accuracy |
| plots/readme_feature_importance.png | top-30 deployed features |

```
python HAR_4_PCA.py                      # full run (Stages 0–3 cached after first run)
ATTITUDE_FILTER=madgwick python ...      # readme's fallback attitude filter
TOPK_FEATURES=80 python ...              # prune width (readme range 60–80)
REBUILD_CACHE=1 python ...               # force re-filtering
```

First run ≈ 8–10 min (ISIF-BU is the heavy stage, as the readme warns); cached reruns ≈ 3 min.
