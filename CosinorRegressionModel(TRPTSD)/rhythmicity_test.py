"""
Valid rhythmicity significance test (AR(1) red-noise surrogates).

Why
---
two_harmonic_cosinor.py reports a nested F-test for the 12-h term whose
p-values are all ~0.  That test assumes the hourly residuals are i.i.d., but
they are heavily autocorrelated, so the effective sample size is tiny compared
to n and the p-values are meaningless.  This script replaces that with a Monte
Carlo test against an AR(1) ("red noise") null that *preserves the short-range
autocorrelation* but, being first-order, cannot itself manufacture a peak at
24 h or 12 h.  Excess amplitude over the AR(1) null is therefore genuine
rhythmicity.

Two statistics are tested per patient (both on the 7-day-detrended signal):

  1. static_A24 : amplitude of a single fixed-phase 24-h cosinor over the whole
                  record.  Tests a *phase-locked* circadian rhythm.
  2. windowed_median_A24 : median of the per-3-day-window 24-h amplitudes
                  (the windowed_cosinor.py statistic).  Tests *local* 24-h
                  rhythmicity and is robust to slow phase drift.

A patient significant on (2) but not (1) has a real but phase-drifting rhythm.
The 12-h static amplitude is tested the same way for completeness.

For each statistic: p = (1 + #{null >= observed}) / (B + 1).
Observed amplitudes also get a moving-block-bootstrap 95% CI.

Outputs (plots/rhythmicity_test/):
  - <patient>_nulltest.svg    null histograms with observed lines
  - summary_grid.svg          all patients
  - rhythmicity_summary.csv
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lfilter

import rns_signals as rns

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "plots", "rhythmicity_test")
os.makedirs(OUT_DIR, exist_ok=True)

DETREND_WINDOW_H = 24 * 7
WINDOW_DAYS = 3
STEP_DAYS = 1
MIN_FRAC_PRESENT = 0.5
N_SURROGATE = 499              # per-series; 24 series, so keep moderate (p floor 0.002)
N_BOOTSTRAP = 300
BLOCK_H = 24
RNG = np.random.default_rng(0)

W24 = 2 * np.pi / 24.0
W12 = 2 * np.pi / 12.0


# -------------------------- detrend -------------------------- #

def detrend_rolling(y: pd.Series, window_h: int = DETREND_WINDOW_H) -> pd.Series:
    trend = y.rolling(window=window_h, min_periods=window_h // 4, center=True).mean()
    return y - trend


# -------------------------- cosinor statistics -------------------------- #

def _design2(t):
    return np.column_stack([
        np.ones_like(t),
        np.cos(W24 * t), np.sin(W24 * t),
        np.cos(W12 * t), np.sin(W12 * t),
    ])


def _design1(t):
    return np.column_stack([np.ones_like(t), np.cos(W24 * t), np.sin(W24 * t)])


def static_amps(t_hours, y):
    """Return (A24, A12) from a 2-harmonic fit on present samples."""
    m = ~np.isnan(y)
    if m.sum() < 5:
        return np.nan, np.nan
    coef, *_ = np.linalg.lstsq(_design2(t_hours[m]), y[m], rcond=None)
    a24 = float(np.hypot(coef[1], coef[2]))
    a12 = float(np.hypot(coef[3], coef[4]))
    return a24, a12


def windowed_median_a24(t_hours, y, span_h):
    """Median single-harmonic 24-h amplitude across 3-day sliding windows."""
    win_h = WINDOW_DAYS * 24
    step_h = STEP_DAYS * 24
    need = MIN_FRAC_PRESENT * win_h
    amps = []
    start = 0.0
    while start + win_h <= span_h + step_h:
        sel = (t_hours >= start) & (t_hours < start + win_h)
        if sel.sum() >= need:
            tw, yw = t_hours[sel], y[sel]
            m = ~np.isnan(yw)
            if m.sum() >= 12:
                coef, *_ = np.linalg.lstsq(_design1(tw[m]), yw[m], rcond=None)
                amps.append(np.hypot(coef[1], coef[2]))
        start += step_h
    return float(np.median(amps)) if amps else np.nan


# -------------------------- AR(1) null + bootstrap -------------------------- #

def fit_ar1(y_interp: np.ndarray):
    """Lag-1 autocorrelation rho and innovation sd for a mean-0 series."""
    y = y_interp - np.nanmean(y_interp)
    y0, y1 = y[:-1], y[1:]
    rho = float(np.clip(np.dot(y0, y1) / np.dot(y0, y0), -0.999, 0.999))
    sigma = float(np.std(y))
    sigma_eps = sigma * np.sqrt(max(1.0 - rho ** 2, 1e-6))
    return rho, sigma_eps


def ar1_surrogates(rho, sigma_eps, n, n_sur):
    """(n_sur x n) AR(1) paths via lfilter; first samples burned by long run."""
    burn = 200
    eps = RNG.standard_normal((n_sur, n + burn)) * sigma_eps
    paths = lfilter([1.0], [1.0, -rho], eps, axis=1)
    return paths[:, burn:]


def moving_block_bootstrap_ci(t_hours, y, statistic):
    """95% CI for a statistic via moving-block bootstrap on present samples.

    Resamples contiguous (t, y) *pairs* as blocks so each sample keeps its own
    phase (reordering y against a fixed t would scramble the rhythm and bias
    the amplitude CI downward).
    """
    m = ~np.isnan(y)
    t, yv = t_hours[m], y[m]
    n = len(yv)
    if n < BLOCK_H * 2:
        return np.nan, np.nan
    n_blocks = int(np.ceil(n / BLOCK_H))
    max_start = n - BLOCK_H
    vals = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        starts = RNG.integers(0, max_start + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(BLOCK_H)[None, :]).ravel()[:n]
        vals[b] = statistic(t[idx], yv[idx])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


# -------------------------- per-patient test -------------------------- #

def test_series(s):
    df = s["data"]
    t0 = df["t"].iloc[0]
    t_h = (df["t"] - t0).dt.total_seconds().to_numpy() / 3600.0
    y_dt = detrend_rolling(df["y"]).to_numpy()
    span_h = t_h[-1]

    # observed statistics
    a24_obs, a12_obs = static_amps(t_h, y_dt)
    wmed_obs = windowed_median_a24(t_h, y_dt, span_h)

    # AR(1) null built from the autocorrelation of the detrended series
    y_interp = pd.Series(y_dt).interpolate(limit_direction="both").to_numpy()
    rho, sigma_eps = fit_ar1(y_interp)
    nan_mask = np.isnan(y_dt)
    sur = ar1_surrogates(rho, sigma_eps, len(y_dt), N_SURROGATE)

    null_a24 = np.empty(N_SURROGATE)
    null_a12 = np.empty(N_SURROGATE)
    null_wmed = np.empty(N_SURROGATE)
    for i in range(N_SURROGATE):
        ys = sur[i].copy()
        ys[nan_mask] = np.nan                      # same sampling as observed
        a24, a12 = static_amps(t_h, ys)
        null_a24[i], null_a12[i] = a24, a12
        null_wmed[i] = windowed_median_a24(t_h, ys, span_h)

    def pval(obs, null):
        if not np.isfinite(obs):
            return np.nan
        return float((1 + np.sum(null >= obs)) / (len(null) + 1))

    p24 = pval(a24_obs, null_a24)
    p12 = pval(a12_obs, null_a12)
    pw = pval(wmed_obs, null_wmed)

    a24_lo, a24_hi = moving_block_bootstrap_ci(
        t_h, y_dt, lambda t, y: static_amps(t, y)[0])

    # Phase-locking index: how much of the locally-present 24-h amplitude
    # survives a single whole-record fixed-phase fit.  ~1 => phase-locked,
    # ~0 => phase-drifting (the static cosine averages the rhythm away).
    pli = (a24_obs / wmed_obs) if (np.isfinite(a24_obs) and wmed_obs > 0) else np.nan

    return {
        "patient": s["patient"], "label": s["label"],
        "signal_kind": s["signal_kind"], "detector": s["detector"],
        "signal_col": s["pattern_col"], "sat_frac": s["sat_frac"],
        "is_representative": s["is_representative"],
        "ar1_rho": rho,
        "static_A24": a24_obs, "static_A24_ci_lo": a24_lo, "static_A24_ci_hi": a24_hi,
        "p_static_A24": p24,
        "static_A12": a12_obs, "p_static_A12": p12,
        "windowed_median_A24": wmed_obs,
        "p_windowed_A24": pw,
        "phase_locking_index": pli,
        "_null_a24": null_a24, "_null_a12": null_a12, "_null_wmed": null_wmed,
    }


# -------------------------- plotting -------------------------- #

def _null_panel(ax, null, obs, ci, title, color):
    null = null[np.isfinite(null)]
    ax.hist(null, bins=30, color=color, alpha=0.6, density=True)
    if np.isfinite(obs):
        ax.axvline(obs, color="C3", lw=2, label=f"observed={obs:.2f}")
        if ci is not None and np.isfinite(ci[0]):
            ax.axvspan(ci[0], ci[1], color="C3", alpha=0.15)
    hi = np.percentile(null, 95) if len(null) else np.nan
    if np.isfinite(hi):
        ax.axvline(hi, color="black", ls="--", lw=1.0, label="null 95th pct")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylabel("density")


def plot_patient(res, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _null_panel(axes[0], res["_null_a24"], res["static_A24"],
                (res["static_A24_ci_lo"], res["static_A24_ci_hi"]),
                f"static 24-h amp  (p={res['p_static_A24']:.3g})", "C0")
    _null_panel(axes[1], res["_null_wmed"], res["windowed_median_A24"], None,
                f"windowed-median 24-h amp  (p={res['p_windowed_A24']:.3g})", "C2")
    _null_panel(axes[2], res["_null_a12"], res["static_A12"], None,
                f"static 12-h amp  (p={res['p_static_A12']:.3g})", "C4")
    for ax in axes:
        ax.set_xlabel("amplitude")
    fig.suptitle(
        f"AR(1) rhythmicity test — {res['label']}  |  {res['signal_kind']} / "
        f"{res['signal_col']}  |  rho={res['ar1_rho']:.3f}  |  "
        f"{N_SURROGATE} surrogates  |  saturated {res['sat_frac']*100:.0f}%",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_grid(results, out_path):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.6 * n))
    if n == 1:
        axes = axes[None, :]
    for i, res in enumerate(results):
        _null_panel(axes[i, 0], res["_null_a24"], res["static_A24"],
                    (res["static_A24_ci_lo"], res["static_A24_ci_hi"]),
                    f"{res['label']}  static A24 (p={res['p_static_A24']:.3g})", "C0")
        _null_panel(axes[i, 1], res["_null_wmed"], res["windowed_median_A24"], None,
                    f"{res['label']}  windowed-median A24 (p={res['p_windowed_A24']:.3g})",
                    "C2")
    for ax in axes[-1]:
        ax.set_xlabel("amplitude")
    fig.suptitle("AR(1) rhythmicity test — static vs drift-robust statistic",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------- main -------------------------- #

def _verdict(res):
    rhythmic = (res["p_static_A24"] < 0.05) or (res["p_windowed_A24"] < 0.05)
    pli = res["phase_locking_index"]
    if not rhythmic:
        return "no rhythm above red noise"
    if not np.isfinite(pli):
        return "rhythmic"
    if pli >= 0.6:
        return "phase-locked rhythm"
    if pli < 0.4:
        return "phase-drifting rhythm"
    return "partially phase-locked"


def main():
    results = []
    for s in rns.iter_series():
        if s["data"]["y"].notna().sum() < WINDOW_DAYS * 24:
            continue
        res = test_series(s)
        results.append(res)
        kind_dir = os.path.join(OUT_DIR, s["signal_kind"])
        os.makedirs(kind_dir, exist_ok=True)
        plot_patient(res, os.path.join(kind_dir, f"{s['label']}_nulltest.svg"))
        print(f"[ok] {res['label']:14s} {res['signal_kind']:9s} "
              f"rho={res['ar1_rho']:.3f}  "
              f"A24={res['static_A24']:6.2f}(p={res['p_static_A24']:.3g})  "
              f"winA24={res['windowed_median_A24']:6.2f}(p={res['p_windowed_A24']:.3g})  "
              f"PLI={res['phase_locking_index'] if np.isfinite(res['phase_locking_index']) else float('nan'):.2f}  "
              f"sat={res['sat_frac']*100:3.0f}%  -> {_verdict(res)}")

    # summary grids per signal kind (detection: representative detector only)
    det = sorted([r for r in results
                  if r["signal_kind"] == "detection" and r["is_representative"]],
                 key=lambda r: r["patient"])
    rx = sorted([r for r in results if r["signal_kind"] == "stim_rx"],
                key=lambda r: r["patient"])
    if det:
        plot_summary_grid(det, os.path.join(OUT_DIR, "summary_grid_detection.svg"))
    if rx:
        plot_summary_grid(rx, os.path.join(OUT_DIR, "summary_grid_stim_rx.svg"))

    cols = ["label", "patient", "signal_kind", "detector", "signal_col",
            "is_representative", "sat_frac", "ar1_rho",
            "static_A24", "static_A24_ci_lo", "static_A24_ci_hi", "p_static_A24",
            "static_A12", "p_static_A12",
            "windowed_median_A24", "p_windowed_A24", "phase_locking_index"]
    df_out = pd.DataFrame([{k: r[k] for k in cols} for r in results])
    df_out["verdict"] = [_verdict(r) for r in results]
    csv_path = os.path.join(OUT_DIR, "rhythmicity_summary.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"[ok] {len(results)} series -> {csv_path}")


if __name__ == "__main__":
    main()
