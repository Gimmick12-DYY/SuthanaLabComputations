"""
Follow-up periodicity diagnostics for TR-PTSD RNS hourly data.

Builds on `periodicity_baseline.py` and targets three questions raised by
that first pass:

  1. Does removing a slow trend reveal a 24-h rhythm in patients whose
     baseline ACF was monotonic (e.g. RNS_D_*)?  ->  rolling-mean detrend,
     then re-run ACF and Lomb-Scargle.
  2. Is the rhythm present but phase-drifting (which would explain the
     flat global hour-of-day polar plots)?  ->  weekly hour-of-day heatmap.
  3. Are there sub-day periodicities (12 h, 8 h, etc.) hidden under the
     low-frequency baseline?  ->  log-log Lomb-Scargle.

Outputs per-patient figures and a comparison grid in
plots/periodicity_followup/.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.signal import lombscargle
from statsmodels.tsa.stattools import acf

import rns_signals as rns

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "plots", "periodicity_followup")
os.makedirs(OUT_DIR, exist_ok=True)

DETREND_WINDOW_H = 24 * 7              # 7-day rolling mean for detrending
ACF_MAX_LAG_HOURS = 24 * 14            # 14-day ACF
LS_PERIOD_MIN_H = 2.0
LS_PERIOD_MAX_H = 24 * 14
LS_N_FREQS = 4000
HEATMAP_BIN_DAYS = 7                   # one row per 7-day chunk


def detrend_rolling(y: pd.Series, window_h: int = DETREND_WINDOW_H) -> pd.Series:
    """Subtract a centered rolling mean. NaN-tolerant."""
    trend = y.rolling(window=window_h, min_periods=window_h // 4, center=True).mean()
    return y - trend


def compute_acf(y: np.ndarray, nlags: int) -> np.ndarray:
    y = pd.Series(y).interpolate(limit_direction="both").to_numpy()
    return acf(y, nlags=nlags, fft=True, missing="conservative")


def lomb_scargle_periodogram(t_hours: np.ndarray, y: np.ndarray):
    mask = ~np.isnan(y)
    t = t_hours[mask].astype(float)
    yv = y[mask].astype(float)
    yv = yv - yv.mean()
    periods = np.geomspace(LS_PERIOD_MIN_H, LS_PERIOD_MAX_H, LS_N_FREQS)
    angular_freqs = 2.0 * np.pi / periods
    power = lombscargle(t, yv, angular_freqs, normalize=True)
    return periods, power


def weekly_hour_heatmap(df: pd.DataFrame, col: str, bin_days: int = HEATMAP_BIN_DAYS):
    """Return (matrix [n_bins x 24], bin_start_dates) of mean signal."""
    df = df.copy()
    t0 = df["t"].iloc[0].normalize()
    df["bin"] = ((df["t"] - t0).dt.total_seconds() // (bin_days * 86400)).astype(int)
    df["hour"] = df["t"].dt.hour
    pivot = (
        df.groupby(["bin", "hour"])[col]
        .mean()
        .unstack("hour")
        .reindex(columns=range(24))
    )
    bin_starts = [t0 + pd.Timedelta(days=bin_days * b) for b in pivot.index]
    return pivot.to_numpy(), bin_starts


def _vlines_periods(ax, periods_days):
    for d in periods_days:
        ax.axvline(d, color="red", ls="--", lw=0.7, alpha=0.6)


def plot_patient(name: str, df: pd.DataFrame, sig_col: str, out_path: str):
    df = df.copy()
    df["y_log"] = np.log1p(df["y"])
    df["y_dt"] = detrend_rolling(df["y"])
    df["y_log_dt"] = detrend_rolling(df["y_log"])

    t_h = (df["t"] - df["t"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          width_ratios=[1.2, 1, 1])

    # --- Row 1: raw / log1p detrended diagnostics ----------------------------
    ax_hm = fig.add_subplot(gs[0, 0])
    mat, bin_starts = weekly_hour_heatmap(df, "y")
    im = ax_hm.imshow(
        mat, aspect="auto", origin="lower", interpolation="nearest",
        extent=[-0.5, 23.5, 0, len(bin_starts)],
        cmap="viridis",
    )
    ax_hm.set_xlabel("hour of day")
    ax_hm.set_ylabel(f"{HEATMAP_BIN_DAYS}-day window (oldest → newest)")
    ax_hm.set_xticks(range(0, 24, 3))
    ytick_idx = np.linspace(0, len(bin_starts) - 1, min(8, len(bin_starts))).astype(int)
    ax_hm.set_yticks(ytick_idx + 0.5)
    ax_hm.set_yticklabels([bin_starts[i].strftime("%Y-%m-%d") for i in ytick_idx],
                          fontsize=7)
    ax_hm.set_title("Hour-of-day, weekly bins (raw)")
    fig.colorbar(im, ax=ax_hm, fraction=0.04, pad=0.02)

    ax_acf = fig.add_subplot(gs[0, 1])
    a_raw = compute_acf(df["y"].to_numpy(), ACF_MAX_LAG_HOURS)
    a_dt = compute_acf(df["y_dt"].to_numpy(), ACF_MAX_LAG_HOURS)
    lags = np.arange(len(a_raw)) / 24.0
    ax_acf.plot(lags, a_raw, lw=0.7, alpha=0.5, label="raw")
    ax_acf.plot(lags, a_dt, lw=0.9, color="C3", label="detrended")
    _vlines_periods(ax_acf, [1, 7])
    ax_acf.axhline(0, color="black", lw=0.5)
    ax_acf.set_xlabel("lag (days)")
    ax_acf.set_ylabel("ACF")
    ax_acf.set_title("Autocorrelation: raw vs detrended")
    ax_acf.legend(fontsize=8, loc="upper right")

    ax_ls = fig.add_subplot(gs[0, 2])
    periods_r, pwr_r = lomb_scargle_periodogram(t_h, df["y"].to_numpy())
    periods_d, pwr_d = lomb_scargle_periodogram(t_h, df["y_dt"].to_numpy())
    ax_ls.loglog(periods_r / 24.0, pwr_r + 1e-8, lw=0.7, alpha=0.5, label="raw")
    ax_ls.loglog(periods_d / 24.0, pwr_d + 1e-8, lw=0.9, color="C3", label="detrended")
    _vlines_periods(ax_ls, [0.25, 0.5, 1, 7])
    ax_ls.set_xlabel("period (days)")
    ax_ls.set_ylabel("LS power")
    ax_ls.set_title("Lomb-Scargle (log-log)")
    ax_ls.legend(fontsize=8, loc="lower right")

    # --- Row 2: log1p versions -----------------------------------------------
    ax_hm2 = fig.add_subplot(gs[1, 0])
    mat_l, bin_starts_l = weekly_hour_heatmap(df, "y_log")
    im2 = ax_hm2.imshow(
        mat_l, aspect="auto", origin="lower", interpolation="nearest",
        extent=[-0.5, 23.5, 0, len(bin_starts_l)],
        cmap="viridis",
    )
    ax_hm2.set_xlabel("hour of day")
    ax_hm2.set_ylabel(f"{HEATMAP_BIN_DAYS}-day window")
    ax_hm2.set_xticks(range(0, 24, 3))
    ytick_idx = np.linspace(0, len(bin_starts_l) - 1, min(8, len(bin_starts_l))).astype(int)
    ax_hm2.set_yticks(ytick_idx + 0.5)
    ax_hm2.set_yticklabels([bin_starts_l[i].strftime("%Y-%m-%d") for i in ytick_idx],
                           fontsize=7)
    ax_hm2.set_title("Hour-of-day, weekly bins (log1p)")
    fig.colorbar(im2, ax=ax_hm2, fraction=0.04, pad=0.02)

    ax_acf2 = fig.add_subplot(gs[1, 1])
    a_raw_l = compute_acf(df["y_log"].to_numpy(), ACF_MAX_LAG_HOURS)
    a_dt_l = compute_acf(df["y_log_dt"].to_numpy(), ACF_MAX_LAG_HOURS)
    ax_acf2.plot(lags, a_raw_l, lw=0.7, alpha=0.5, label="log1p")
    ax_acf2.plot(lags, a_dt_l, lw=0.9, color="C3", label="log1p detrended")
    _vlines_periods(ax_acf2, [1, 7])
    ax_acf2.axhline(0, color="black", lw=0.5)
    ax_acf2.set_xlabel("lag (days)")
    ax_acf2.set_ylabel("ACF")
    ax_acf2.set_title("Autocorrelation (log1p)")
    ax_acf2.legend(fontsize=8, loc="upper right")

    ax_ls2 = fig.add_subplot(gs[1, 2])
    periods_rl, pwr_rl = lomb_scargle_periodogram(t_h, df["y_log"].to_numpy())
    periods_dl, pwr_dl = lomb_scargle_periodogram(t_h, df["y_log_dt"].to_numpy())
    ax_ls2.loglog(periods_rl / 24.0, pwr_rl + 1e-8, lw=0.7, alpha=0.5, label="log1p")
    ax_ls2.loglog(periods_dl / 24.0, pwr_dl + 1e-8, lw=0.9, color="C3",
                  label="log1p detrended")
    _vlines_periods(ax_ls2, [0.25, 0.5, 1, 7])
    ax_ls2.set_xlabel("period (days)")
    ax_ls2.set_ylabel("LS power")
    ax_ls2.set_title("Lomb-Scargle log1p (log-log)")
    ax_ls2.legend(fontsize=8, loc="lower right")

    n_h = df["y"].notna().sum()
    span_days = (df["t"].iloc[-1] - df["t"].iloc[0]).total_seconds() / 86400.0
    fig.suptitle(
        f"{name}  |  {sig_col}  |  n={n_h} hours  |  span={span_days:.0f} d  "
        f"|  detrend window={DETREND_WINDOW_H//24} d",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_grid(patient_data, out_path: str, transform: str):
    """Rows = patients, cols = [weekly heatmap, detrended ACF, detrended LS log-log]."""
    n = len(patient_data)
    fig = plt.figure(figsize=(15, 3.3 * n))
    gs = fig.add_gridspec(n, 3, hspace=0.55, wspace=0.4,
                          width_ratios=[1.3, 1, 1])

    base_col = "y" if transform == "raw" else "y_log"

    for i, (name, df, sig_col) in enumerate(patient_data):
        df = df.copy()
        if transform == "log1p":
            df["y_log"] = np.log1p(df["y"])
        df["y_dt"] = detrend_rolling(df[base_col])

        ax_hm = fig.add_subplot(gs[i, 0])
        mat, bin_starts = weekly_hour_heatmap(df, base_col)
        im = ax_hm.imshow(
            mat, aspect="auto", origin="lower", interpolation="nearest",
            extent=[-0.5, 23.5, 0, len(bin_starts)],
            cmap="viridis",
        )
        ax_hm.set_xticks(range(0, 24, 6))
        ytick_idx = np.linspace(0, len(bin_starts) - 1,
                                min(4, len(bin_starts))).astype(int)
        ax_hm.set_yticks(ytick_idx + 0.5)
        ax_hm.set_yticklabels([bin_starts[k].strftime("%Y-%m") for k in ytick_idx],
                              fontsize=7)
        ax_hm.set_ylabel(name, fontsize=10)
        fig.colorbar(im, ax=ax_hm, fraction=0.04, pad=0.02)

        ax_acf = fig.add_subplot(gs[i, 1])
        a_dt = compute_acf(df["y_dt"].to_numpy(), ACF_MAX_LAG_HOURS)
        ax_acf.plot(np.arange(len(a_dt)) / 24.0, a_dt, lw=0.9, color="C3")
        _vlines_periods(ax_acf, [1, 7])
        ax_acf.axhline(0, color="black", lw=0.5)
        ax_acf.set_xlabel("lag (days)")
        ax_acf.set_ylabel("ACF (detrended)")

        ax_ls = fig.add_subplot(gs[i, 2])
        t_h = (df["t"] - df["t"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
        periods, pwr = lomb_scargle_periodogram(t_h, df["y_dt"].to_numpy())
        ax_ls.loglog(periods / 24.0, pwr + 1e-8, lw=0.9, color="C3")
        _vlines_periods(ax_ls, [0.25, 0.5, 1, 7])
        ax_ls.set_xlabel("period (days)")
        ax_ls.set_ylabel("LS power")

        if i == 0:
            ax_hm.set_title("Hour-of-day, weekly bins")
            ax_acf.set_title("ACF (detrended)")
            ax_ls.set_title("Lomb-Scargle (log-log, detrended)")

    fig.suptitle(f"Periodicity follow-up — all patients ({transform})", fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    by_kind = {}
    for s in rns.iter_series():
        df = s["data"].copy()
        if df["y"].notna().sum() < 48:
            continue
        kind_dir = os.path.join(OUT_DIR, s["signal_kind"])
        os.makedirs(kind_dir, exist_ok=True)
        sig_desc = f"{s['signal_kind']} / {s['pattern_col']}  (sat {s['sat_frac']*100:.0f}%)"
        out_path = os.path.join(kind_dir, f"{s['label']}_followup.svg")
        plot_patient(s["label"], df, sig_desc, out_path)
        by_kind.setdefault(s["signal_kind"], []).append((s, df))
        print(f"[ok] {s['label']:14s} {s['signal_kind']:9s} -> {out_path}")

    for kind, entries in by_kind.items():
        if kind == "detection":
            entries = [(s, df) for (s, df) in entries if s["is_representative"]]
        entries = sorted(entries, key=lambda e: e[0]["patient"])
        patient_data = [(s["label"], df, s["pattern_col"]) for (s, df) in entries]
        for transform in ("raw", "log1p"):
            grid = os.path.join(OUT_DIR, f"comparison_grid_{kind}_{transform}.svg")
            plot_comparison_grid(patient_data, grid, transform)
        print(f"[ok] {kind} comparison grids -> {OUT_DIR}")


if __name__ == "__main__":
    main()
