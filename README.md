IMU Rotational-Activity Classification Pipeline
Filter stack + feature engineering for Random Forest (50 Hz, 2 s / 100-sample window)
0. What the data actually is (verified from ROT_activities_master.csv)
Fact
Value
Why it matters
Rows
1,048,572
—
Devices
LEFT, RIGHT wrist units
2 independent 6-DOF IMUs per sample window
Classes
12, not 8: {B,L,R}_{SHOU,HAND}_{CLK,ACLK}
Body/Left/Right × Shoulder/Hand × Clockwise/Anti-clockwise
Participants
6 (1–6), 65 sessions
Must split by participant, not by row
Channels
accel_x/y/z, gyro_x/y/z, mag_x/y/z
mag_x/y/z = 0.0 for every row — magnetometer is dead/unused
Nominal rate
~50 Hz average
Timestamps arrive in bursts (BLE packet batching): inter-sample gaps range 1 ms–188 ms, not a clean 20 ms grid
Accel range
±4.6 (unit-less, consistent with g)
includes gravity (not gravity-removed)
Gyro range
±520 (consistent with deg/s)
large excursions confirm fast wrist/shoulder rotation
Two consequences that change the pipeline vs. a "textbook" IMU pipeline:
No magnetometer → any orientation filter (Madgwick, EKF, ISIF-BU) must run in 6-DOF IMU-only mode. Yaw will drift; roll/pitch (gravity-referenced) will not.
Timestamps are irregular → you must resample to a uniform 50 Hz grid before any filter runs, or every filter above (all of which assume fixed Δt) will be silently wrong.
1. What each filter actually does (and where it belongs)
Filter
Category
Purpose
Verdict for this task
Median filter
Non-linear despiking
Kills single-sample spikes/dropouts without smearing edges
✅ Use — Stage 1, cheap insurance against BLE burst glitches
Moving average
FIR low-pass (simplest)
Smooths, but blurs edges, adds lag, poor stop-band
❌ Skip — Butterworth strictly better here
Savitzky-Golay
Polynomial-fit smoother
Smooths while preserving peak shape and slope (good for derivatives)
✅ Use — Stage 2b, only for jerk/derivative features
Butterworth (low-pass, IIR)
Frequency-domain denoise
Maximally flat pass-band, sharp roll-off, cheap
✅ Use — Stage 2a, primary denoiser (run zero-phase)
FIR / IIR (general)
Filter design families
Butterworth is an IIR filter; FIR would need ~4× the order for equal roll-off
Covered by Butterworth choice above
Kalman filter
Linear state estimator
Optimal only if the motion model is linear + noise is Gaussian & known
⚠️ Arm rotation is nonlinear → not used directly
Extended Kalman Filter (EKF)
Nonlinear state estimator
Linearizes via Jacobian each step; degrades under fast/abrupt turns (exactly the paper's failure case)
⚠️ Superseded by ISIF-BU below
Madgwick
Complementary orientation filter
Fast, gradient-descent quaternion fusion of accel+gyro(+mag)
✅ Use as fallback if compute budget is too tight for ISIF-BU
ISIF-BU (uploaded paper)
Robust nonlinear state estimator (SIF + Bayesian update)
Designed exactly for abrupt turning-rate model mismatch — sliding-mode robustness + Bayesian refinement, QR/SVD instead of Cholesky (never fails to decompose)
✅ Use as primary orientation/attitude estimator — a sudden CLK↔ACLK reversal is precisely the "turning-rate jump" scenario the paper validates against
Key insight driving the design: CLK vs ACLK is a sign-of-rotation problem. The paper's own test scenario (ϑ_t → ϑ_t + π/50 at t = 100 s, a step change in turning rate) is structurally identical to a person reversing rotation direction mid-window. ISIF-BU was shown in the paper to cut ARMSE by 14–44% over CSIF/EKF-style filters under exactly this kind of abrupt-turn mismatch — so it is the correct choice for estimating a clean, low-noise rotation-angle trace per window, not just a "nice to have."
2. Recommended pipeline (end-to-end order)
Raw CSV (LEFT + RIGHT, irregular timestamps, mag=0)
        │
Stage 0 │ Resample & sync          → uniform 50 Hz grid, linear interp, drop mag_*
        ▼
Stage 1 │ Median despike           → window = 3 samples, per axis, per device
        ▼
Stage 2a│ Butterworth low-pass     → order 4, zero-phase (filtfilt)
        │                            accel cutoff 10 Hz | gyro cutoff 15 Hz
        ▼
Stage 2b│ Savitzky-Golay (branch)  → window 7, polyorder 2 → jerk / angular-accel only
        ▼
Stage 3 │ ISIF-BU attitude filter  → per device: quaternion / roll-pitch-yaw + gyro bias
        │                            (Madgwick = fallback if latency-constrained)
        ▼
Stage 4 │ Windowing                → 100 samples (2.0 s), 50-sample stride (50% overlap)
        ▼
Stage 5 │ Feature extraction       → per device, then concat LEFT+RIGHT (+ cross-device)
        ▼
Stage 6 │ Random Forest            → 18 trees, max_depth 8, min_samples_leaf 2,
        │                            min_samples_split 4, max_features='sqrt'
        ▼
     Prediction (12 classes)
Run Stage 1–3 independently per device (LEFT, RIGHT) since they are physically separate sensors; only merge at the feature vector in Stage 5.
3. Stage-by-stage configuration
Stage 0 — Resample & sync
Target grid: fixed Δt = 20 ms (50 Hz), built from session start → session end.
Linear-interpolate accel/gyro onto the grid per device.
Drop mag_x, mag_y, mag_z (always zero — carrying them only adds noise/leakage risk to the model).
Re-align LEFT and RIGHT onto the same grid so cross-device features (Stage 5) are timestamp-matched.
Stage 1 — Median despike
Window: 3 samples (5 if you see visible glitch survivors).
Apply independently to each of the 6 channels (accel_x/y/z, gyro_x/y/z), per device.
scipy.signal.medfilt(x, kernel_size=3)
Stage 2a — Butterworth low-pass (primary denoiser)
Order: 4.
Cutoff: 10 Hz for accel, 15 Hz for gyro (Nyquist = 25 Hz at 50 Hz sampling; human limb motion energy is almost entirely <10 Hz, fast wrist snap can reach ~12–15 Hz).
Zero-phase: use filtfilt, not lfilter — you are processing offline/batch, so there is no reason to accept the phase lag of a causal filter.
b, a = butter(4, cutoff/(fs/2), btype='low'); x_f = filtfilt(b, a, x)
Stage 2b — Savitzky-Golay (derivative branch only)
Window length: 7 samples, polynomial order: 2.
Purpose: compute smooth jerk (d(accel)/dt) and angular acceleration (d(gyro)/dt) for feature extraction — do not use this to replace Stage 2a; it's a parallel branch feeding Stage 5 only.
savgol_filter(x, window_length=7, polyorder=2, deriv=1, delta=1/fs)
Stage 3 — ISIF-BU attitude estimator (primary), Madgwick (fallback)
State per device: x = [q0, q1, q2, q3, bgx, bgy, bgz] (quaternion + gyro bias), or the simpler [roll, pitch, yaw, bias] Euler form if you want fewer states for the shallow RF trees to exploit downstream.
Mapping from the paper's ISIF-BU (Algorithm 1) to this problem:
Process model f(·): quaternion propagation via bias-corrected gyro rate (standard strap-down integration), replacing the paper's turning-model f(x).
Measurement model h(·): predicted gravity vector in body frame from the current quaternion, compared against the Stage-2a-filtered accelerometer reading (magnetometer term = 0, so drop it from h(·) and R_t — do not feed the dead mag channels in as zero-variance measurements, that will corrupt Σ_zz).
Decomposition: QR + SVD (not Cholesky) for Σ_t|t-1 and Σ^(p)_t|t — required because sudden direction reversals are exactly the kind of model mismatch that makes Cholesky fail (paper §3.2, Remark 2).
Saturation function g(k) = 2(1+e^{-b(k-c)})^{-1} - 1: set c = 0, b = 2.1 (paper's recommended sweet spot is b ∈ [1.7, 2.9]; b = 2.1 is what the paper itself uses in its final comparison).
Sliding boundary layer: Δ = 0.5 R_t (accelerometer measurement covariance).
Chi-square confidence factor: σ = 0.05 (95% confidence), μ = dim(z) − 1 = 2 (gravity vector reduced to 2 independent constraints after normalization, or 3 if used un-normalized — pick 2 if you normalize the accel vector to unit gravity first, which you should, since only direction — not magnitude — is informative for orientation).
Reset per window: initialize the quaternion at the start of every 2 s window from the previous window's last estimate (don't reset to identity) so yaw drift never exceeds ~2 s worth — with no magnetometer this keeps drift low single-digit degrees, which is irrelevant since CLK/ACLK is decided by the sign and magnitude of yaw change across the window, not absolute yaw.
Output per window, per device: a clean [roll(t), pitch(t), yaw(t)] (or quaternion) trace at 50 Hz — this feeds Stage 5.
Fallback: if compute budget doesn't allow ISIF-BU (it is the heaviest filter — paper reports ~1.3 s per 300-step Monte Carlo run vs. ~0.6–0.9 s for EKF-class filters), use Madgwick (IMU-only, no mag) with gain β ≈ 0.041 as a cheaper substitute. Expect the classifier to lose some accuracy on activities that reverse direction mid-window, since Madgwick has no explicit robustness to abrupt turning-rate mismatch.
Stage 4 — Windowing
Window length: 100 samples = 2.0 s (fixed, as specified).
Stride: 50 samples (50% overlap) — doubles effective training data and is standard for HAR; drop to 0% overlap only if session count/session length is a strict constraint.
Label a window by the activity_label that covers ≥ 50% of it (should be 100% inside your recording protocol, but guard against boundary windows spanning two labels — discard those).
Stage 5 — Feature extraction (per window, per device → then concatenate)
Time-domain (per channel: accel_x/y/z, gyro_x/y/z, jerk_x/y/z, ang-accel_x/y/z, roll/pitch/yaw): mean, std, min, max, range, RMS, mean absolute deviation, skewness, kurtosis, zero-crossing rate, signal magnitude area (SMA, accel and gyro triads).
Frequency-domain (per channel, FFT on the 100-sample window — 100 samples gives 50 usable bins at 50 Hz, i.e. 0.5 Hz resolution, enough for this purpose): dominant frequency, spectral energy, spectral entropy, energy in 0–3 Hz / 3–8 Hz / 8–15 Hz bands.
Orientation-derived (the highest-value features for this specific task):
Net rotation angle = yaw(t=100) − yaw(t=0) from the ISIF-BU trace → sign directly encodes CLK vs ACLK; this is close to a single-feature discriminator.
Range and std of roll/pitch/yaw over the window.
Peak angular velocity and its sign (max(|gyro_z|) * sign(gyro_z at peak)).
Quaternion component stats (mean, std) if you keep the quaternion form instead of Euler.
Cross-axis / cross-device (captures body-location, i.e. Shoulder vs Hand, Left vs Right vs Body):
Correlation between accel axes and between gyro axes (captures rotation-plane).
Correlation between LEFT-device and RIGHT-device signals (shoulder rotations move both wrists coherently; hand rotations are near-independent — this is your main Shoulder-vs-Hand and Left/Right/Body discriminator).
Amplitude ratio ‖accel‖_LEFT / ‖accel‖_RIGHT and same for gyro.
Feature budget: ~10 time-domain × 15 channels ≈ 150, +4 freq-domain × 15 ≈ 60, + ~10 orientation, + ~8 cross-device ≈ ~230 raw features per device-pair-window. For an 18-tree, depth-8 forest, prune this down via feature importance (fit once, keep top ~60–80) before finalizing — a shallow, narrow forest overfits and slows down with >200 noisy columns; it does not need them.
Stage 6 — Random Forest
Your current settings (18 trees, depth 8, min_samples_leaf 2, min_samples_split 4, max_features='sqrt') are reasonable for ~60–80 features. Two changes that will move accuracy more than any filter tweak:
Group cross-validation by participant_id (leave-one-participant-out), not random row/window split — random splits leak overlapping-window and same-session data across train/test and will overstate accuracy.
Class-balance check: your 12 classes range 64k–102k raw rows — roughly balanced, but confirm balance again after windowing, not before; add class_weight='balanced' if windows skew.
4. Why this order and not another
Despike before smoothing (Stage 1 before 2a): a median filter run after Butterworth can't undo the ringing a spike causes when it convolves through the low-pass filter. Spikes must die first.
Denoise before state estimation (Stage 2 before 3): ISIF-BU (like any Kalman-family filter) assumes its input noise covariance R_t is stationary and Gaussian; feeding it raw spiky data violates that assumption and destabilizes the Bayesian update. Feed it the Butterworth-cleaned signal.
State estimation before windowing/features (Stage 3 before 4): orientation is a stateful estimate — it needs continuous evolution across the whole session, not per-window restarts (aside from the drift-reset carry-over described above). Windowing first would break the recursive filter's history.
Savitzky-Golay stays a side branch, not the main denoiser: its strength (preserving derivative shape) is wasted on the accel/gyro channels the RF consumes directly — Butterworth's steeper roll-off removes more true noise there. Save S-G for exactly the channels where derivative shape matters (jerk/ang-accel).
5. Summary checklist
[ ] Resample to uniform 50 Hz grid per device before touching any filter
[ ] Drop dead mag_x/y/z columns
[ ] Median filter (k=3) → Butterworth low-pass (order 4, filtfilt, 10/15 Hz) on accel/gyro
[ ] Savitzky-Golay (win 7, order 2) → jerk & angular-acceleration features only
[ ] ISIF-BU (6-DOF, no mag), b=2.1, c=0, Δ=0.5R_t, σ=0.05, QR+SVD decompositions → roll/pitch/yaw per device, carried continuously across the session, not reset per window
[ ] Window: 100 samples / 2.0 s, 50% overlap
[ ] Features: time + frequency + orientation-derived + cross-device correlation (~230 raw → prune to top 60–80 by importance)
[ ] Random Forest: current hyperparameters are fine; fix evaluation to leave-one-participant-out
