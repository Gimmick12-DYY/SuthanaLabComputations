"""
Time-resolved (sliding-window) 24-h cosinor, per (patient, signal) series.

Motivation
----------
The static two-harmonic cosinor (two_harmonic_cosinor.py) fits a single
fixed-phase rhythm across each patient's entire record.  That cannot tell a
genuinely arrhythmic signal apart from a real 24-h rhythm whose *phase drifts*
over weeks/months (a fixed-phase cosine averages the latter to ~0).

This script fits an independent single-harmonic 24-h cosinor inside a short
sliding window (default 3 days, stepped 1 day) and tracks how amplitude and
acrophase move over time.  A drifting acrophase with locally-high amplitude is
the drift signature; scattered phase with low amplitude is true arrhythmia.

Now runs over every (patient, signal) series from rns_signals: each active
detection pattern AND "Episode starts with RX".  Hourly counts saturate at the
254 register ceiling; we keep raw values but report a per-window saturation
fraction so cap-biased windows are visible (see rns_signals.CAP).

Per-window amplitude gets a moving-block-bootstrap 95% CI (block = 24 h).
Phase stability per series is the amplitude-weighted resultant length R of the
per-window acrophases (R~1 => locked phase, R~0 => drifting).

Outputs (plots/windowed_cosinor/<signal_kind>/):
  - <label>_windowed.svg             4-panel per series (+ saturation overlay)
  - comparison_grid_<kind>.svg       amplitude(t) & acrophase(t), one row/patient
  - windowed_fits.csv                one row per (series, window)
  - phase_stability_summary.csv      one row per series
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import rns_signals as rns

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "plots", "windowed_cosinor")
os.makedirs(OUT_DIR, exist_ok=True)

DETREND_WINDOW_H = 24 * 7          # 7-day rolling mean detrend
WINDOW_DAYS = 3                    # sliding-window length
STEP_DAYS = 1                      # step between windows
MIN_FRAC_PRESENT = 0.5             # require >=50% of hours present in a window
N_BOOTSTRAP = 200                  # moving-block bootstrap reps for amplitude CI
BLOCK_H = 24                       # bootstrap block length (hours)
RNG = np.random.default_rng(0)

W24 = 2 * np.pi / 24.0


# -------------------------- detrend + cosinor -------------------------- #

def detrend_rolling(y: pd.Series, window_h: int = DETREND_WINDOW_H) -> pd.Series:
    trend = y.rolling(window=window_h, min_periods=window_h // 4, center=True).mean()
    return y - trend


def _design(t_hours: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.ones_like(t_hours), np.cos(W24 * t_hours), np.sin(W24 * t_hours),
    ])


def _amp_acro(b: float, g: float):
    # acrophase = clock time (h) of the PEAK.  For y = b cos(wt) + g sin(wt),
    # the maximum is at wt = atan2(g, b).
    amp = float(np.hypot(b, g))
    acro = float((np.arctan2(g, b) % (2 * np.pi)) * 24.0 / (2 * np.pi))
    return amp, acro


def fit_window(t_hours: np.ndarray, y: np.ndarray):
    mask = ~np.isnan(y)
    if mask.sum() < 12:
        return None
    t, yv = t_hours[mask], y[mask]
    X = _design(t)
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    yhat = X @ coef
    resid = yv - yhat
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    ss_res = float(np.sum(resid ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    amp, acro = _amp_acro(coef[1], coef[2])
    return {"n": int(mask.sum()), "mesor": float(coef[0]), "amp": amp,
            "acro_h": acro, "r2": float(r2), "_X": X, "_yhat": yhat, "_resid": resid}


def bootstrap_amp_ci(X, yhat, resid, n_boot=N_BOOTSTRAP, block=BLOCK_H):
    n = len(resid)
    if n < block * 2:
        return np.nan, np.nan
    n_blocks = int(np.ceil(n / block))
    max_start = n - block
    amps = np.empty(n_boot)
    for b in range(n_boot):
        starts = RNG.integers(0, max_start + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        coef, *_ = np.linalg.lstsq(X, yhat + resid[idx], rcond=None)
        amps[b] = np.hypot(coef[1], coef[2])
    return float(np.percentile(amps, 2.5)), float(np.percentile(amps, 97.5))


def run_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Slide the window across the record; return per-window dataframe.

    df has columns t, y (RAW counts).  Saturation per window is computed on the
    raw values (>= rns.CAP); the cosinor is fit on the detrended signal.
    """
    t0 = df["t"].iloc[0]
    t_h_all = (df["t"] - t0).dt.total_seconds().to_numpy() / 3600.0
    y_raw = df["y"].to_numpy(dtype=float)
    y_dt = detrend_rolling(df["y"]).to_numpy()
    span_h = t_h_all[-1]

    win_h = WINDOW_DAYS * 24
    step_h = STEP_DAYS * 24
    need = MIN_FRAC_PRESENT * win_h

    rows = []
    start = 0.0
    while start + win_h <= span_h + step_h:
        sel = (t_h_all >= start) & (t_h_all < start + win_h)
        if sel.sum() >= need:
            fit = fit_window(t_h_all[sel], y_dt[sel])
            if fit is not None and np.isfinite(fit["amp"]):
                lo, hi = bootstrap_amp_ci(fit["_X"], fit["_yhat"], fit["_resid"])
                raw_win = y_raw[sel]
                present = np.isfinite(raw_win)
                sat = (float(np.sum(raw_win[present] >= rns.CAP) / present.sum())
                       if present.sum() else np.nan)
                rows.append({
                    "win_start_day": start / 24.0,
                    "center_t": t0 + pd.Timedelta(hours=start + win_h / 2),
                    "n": fit["n"], "mesor": fit["mesor"],
                    "amp": fit["amp"], "amp_ci_lo": lo, "amp_ci_hi": hi,
                    "acro_h": fit["acro_h"], "r2": fit["r2"], "sat_frac": sat,
                })
        start += step_h
    return pd.DataFrame(rows)


def phase_stability(win_df: pd.DataFrame) -> dict:
    if win_df.empty:
        return {"n_windows": 0, "mean_acro_h": np.nan, "resultant_R": np.nan,
                "median_amp": np.nan, "frac_amp_ci_above_0": np.nan,
                "mean_sat_frac": np.nan}
    ang = win_df["acro_h"].to_numpy() * (2 * np.pi / 24.0)
    w = win_df["amp"].to_numpy()
    w = np.where(np.isfinite(w), w, 0.0)
    if w.sum() == 0:
        w = np.ones_like(w)
    C = np.sum(w * np.cos(ang)) / w.sum()
    S = np.sum(w * np.sin(ang)) / w.sum()
    R = float(np.hypot(C, S))
    mean_acro = float((np.arctan2(S, C) % (2 * np.pi)) * 24.0 / (2 * np.pi))
    lo = win_df["amp_ci_lo"].to_numpy()
    frac = float(np.mean(lo > 0)) if np.isfinite(lo).any() else np.nan
    return {"n_windows": int(len(win_df)), "mean_acro_h": mean_acro,
            "resultant_R": R, "median_amp": float(win_df["amp"].median()),
            "frac_amp_ci_above_0": frac,
            "mean_sat_frac": float(win_df["sat_frac"].mean())}


# -------------------------- plotting -------------------------- #

def plot_series(label, sig_desc, win_df, summ, sat_frac, out_path):
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    days = win_df["win_start_day"].to_numpy() + WINDOW_DAYS / 2.0

    # (1) amplitude over time + bootstrap CI, with saturation overlay
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(days, win_df["amp"], color="C0", lw=1.2, label="24-h amplitude")
    if win_df["amp_ci_lo"].notna().any():
        ax.fill_between(days, win_df["amp_ci_lo"], win_df["amp_ci_hi"],
                        color="C0", alpha=0.25, label="95% block-bootstrap CI")
    ax.axhline(summ["median_amp"], color="gray", ls="--", lw=0.8,
               label=f"median={summ['median_amp']:.2f}")
    ax.set_xlabel("time (days from start)")
    ax.set_ylabel("amplitude")
    ax.set_title("24-h amplitude over time")
    ax.legend(fontsize=8, loc="upper left")
    if win_df["sat_frac"].notna().any() and win_df["sat_frac"].max() > 0:
        ax2 = ax.twinx()
        ax2.plot(days, win_df["sat_frac"] * 100, color="C3", lw=0.7, alpha=0.5)
        ax2.set_ylabel("% saturated (≥254)", color="C3", fontsize=9)
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis="y", labelcolor="C3")

    # (2) acrophase over time, colored by amplitude
    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(days, win_df["acro_h"], c=win_df["amp"], cmap="viridis",
                    s=14, norm=Normalize())
    ax.set_ylim(0, 24)
    ax.set_yticks(range(0, 25, 6))
    ax.axhline(summ["mean_acro_h"], color="C3", ls="--", lw=1.0,
               label=f"mean acro={summ['mean_acro_h']:.1f} h")
    ax.set_xlabel("time (days from start)")
    ax.set_ylabel("acrophase (hour of day)")
    ax.set_title("24-h acrophase over time")
    ax.legend(fontsize=8, loc="upper right")
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="amplitude")

    # (3) polar acrophase scatter
    ax = fig.add_subplot(gs[1, 0], projection="polar")
    ang = win_df["acro_h"].to_numpy() * (2 * np.pi / 24.0)
    sc2 = ax.scatter(ang, win_df["amp"], c=days, cmap="plasma", s=16, alpha=0.8)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 3)], fontsize=8)
    mean_ang = summ["mean_acro_h"] * (2 * np.pi / 24.0)
    rmax = np.nanmax(win_df["amp"]) if win_df["amp"].notna().any() else 1.0
    ax.annotate("", xy=(mean_ang, summ["resultant_R"] * rmax), xytext=(0, 0),
                arrowprops=dict(color="C3", width=2, headwidth=8))
    ax.set_title(f"Acrophase clock (radius=amp, color=time)\n"
                 f"resultant R={summ['resultant_R']:.2f}", fontsize=10)
    fig.colorbar(sc2, ax=ax, fraction=0.04, pad=0.08, label="days from start")

    # (4) R^2 over time
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(days, win_df["r2"], color="C2", lw=1.0)
    ax.set_xlabel("time (days from start)")
    ax.set_ylabel("window R^2")
    ax.set_title("Per-window fit quality (R^2)")
    ax.set_ylim(0, max(0.1, float(win_df["r2"].max()) * 1.1))

    fig.suptitle(
        f"Windowed 24-h cosinor — {label}  |  {sig_desc}  |  "
        f"{WINDOW_DAYS}-d window / {STEP_DAYS}-d step  |  "
        f"{summ['n_windows']} windows  |  phase R={summ['resultant_R']:.2f}  |  "
        f"saturated {sat_frac*100:.0f}% of hours", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_grid(entries, out_path, kind_title):
    n = len(entries)
    if n == 0:
        return
    fig = plt.figure(figsize=(14, 2.8 * n))
    gs = fig.add_gridspec(n, 2, hspace=0.55, wspace=0.25)
    for i, e in enumerate(entries):
        win_df, summ = e["win_df"], e["summ"]
        days = win_df["win_start_day"].to_numpy() + WINDOW_DAYS / 2.0

        ax_a = fig.add_subplot(gs[i, 0])
        ax_a.plot(days, win_df["amp"], color="C0", lw=1.0)
        if win_df["amp_ci_lo"].notna().any():
            ax_a.fill_between(days, win_df["amp_ci_lo"], win_df["amp_ci_hi"],
                              color="C0", alpha=0.2)
        ax_a.set_ylabel(f"{e['label']}\nsat {e['sat_frac']*100:.0f}%", fontsize=9)
        if i == 0:
            ax_a.set_title("24-h amplitude over time")
        if i == n - 1:
            ax_a.set_xlabel("time (days from start)")

        ax_p = fig.add_subplot(gs[i, 1])
        ax_p.scatter(days, win_df["acro_h"], c=win_df["amp"], cmap="viridis", s=8)
        ax_p.set_ylim(0, 24)
        ax_p.set_yticks(range(0, 25, 6))
        ax_p.axhline(summ["mean_acro_h"], color="C3", ls="--", lw=0.8)
        ax_p.text(0.98, 0.92, f"R={summ['resultant_R']:.2f}", transform=ax_p.transAxes,
                  ha="right", va="top", fontsize=8, color="C3")
        if i == 0:
            ax_p.set_title("24-h acrophase over time (color=amp)")
        if i == n - 1:
            ax_p.set_xlabel("time (days from start)")

    fig.suptitle(f"Windowed 24-h cosinor — {kind_title}", fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------- main -------------------------- #

def main():
    results = []
    for s in rns.iter_series():
        df = s["data"]
        if df["y"].notna().sum() < WINDOW_DAYS * 24:
            continue
        win_df = run_windows(df)
        if win_df.empty:
            print(f"[skip] {s['label']:14s} {s['signal_kind']:9s} no usable windows")
            continue
        summ = phase_stability(win_df)
        kind_dir = os.path.join(OUT_DIR, s["signal_kind"])
        os.makedirs(kind_dir, exist_ok=True)
        sig_desc = f"{s['signal_kind']} / {s['pattern_col']}"
        plot_series(s["label"], sig_desc, win_df, summ, s["sat_frac"],
                    os.path.join(kind_dir, f"{s['label']}_windowed.svg"))
        results.append({**s, "win_df": win_df, "summ": summ})
        print(f"[ok] {s['label']:14s} {s['signal_kind']:9s} "
              f"win={summ['n_windows']:4d}  med_amp={summ['median_amp']:6.2f}  "
              f"R={summ['resultant_R']:.2f}  acro={summ['mean_acro_h']:5.1f}h  "
              f"sat={s['sat_frac']*100:4.0f}%")

    # comparison grids per signal kind (detection: representative detector only)
    det = sorted([r for r in results
                  if r["signal_kind"] == "detection" and r["is_representative"]],
                 key=lambda r: r["patient"])
    rx = sorted([r for r in results if r["signal_kind"] == "stim_rx"],
                key=lambda r: r["patient"])
    plot_comparison_grid(det, os.path.join(OUT_DIR, "comparison_grid_detection.svg"),
                         "detections (representative detector per patient)")
    plot_comparison_grid(rx, os.path.join(OUT_DIR, "comparison_grid_stim_rx.svg"),
                         "Episode starts with RX")

    # CSVs
    all_rows = []
    summ_rows = []
    for r in results:
        wd = r["win_df"].copy()
        for k in ("patient", "signal_kind", "detector", "label"):
            wd.insert(0, k, r[k])
        all_rows.append(wd)
        summ_rows.append({"label": r["label"], "patient": r["patient"],
                          "signal_kind": r["signal_kind"], "detector": r["detector"],
                          "is_representative": r["is_representative"],
                          "sat_frac": r["sat_frac"], **r["summ"]})
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "windowed_fits.csv"), index=False)
    pd.DataFrame(summ_rows).to_csv(
        os.path.join(OUT_DIR, "phase_stability_summary.csv"), index=False)
    print(f"[ok] {len(results)} series -> {OUT_DIR}")


if __name__ == "__main__":
    main()
