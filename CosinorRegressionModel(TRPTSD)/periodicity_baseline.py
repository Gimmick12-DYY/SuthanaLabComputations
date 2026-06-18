"""
Simple non-parametric periodicity diagnostics for TR-PTSD RNS hourly data.

For each patient CSV in data/Data_Cosinor_09.15.25/, computes and plots:
  1. Hour-of-day mean (polar)
  2. Day-of-week mean (polar)
  3. Autocorrelation up to 14 days
  4. Lomb-Scargle periodogram over 2 h – 14 d

Each diagnostic is computed on raw counts and on log1p-transformed counts.
Outputs one figure per patient plus a comparison grid across patients.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lombscargle
from statsmodels.tsa.stattools import acf

import rns_signals as rns

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "plots", "periodicity_baseline")
os.makedirs(OUT_DIR, exist_ok=True)

ACF_MAX_LAG_HOURS = 24 * 14            # 14 days
LS_PERIOD_MIN_H = 2.0
LS_PERIOD_MAX_H = 24 * 14
LS_N_FREQS = 4000


def hour_of_day_profile(df: pd.DataFrame, col: str = "y") -> pd.DataFrame:
    g = df.groupby(df["t"].dt.hour)[col]
    return pd.DataFrame({"mean": g.mean(), "sem": g.sem()}).reindex(range(24))


def dow_profile(df: pd.DataFrame, col: str = "y") -> pd.DataFrame:
    g = df.groupby(df["t"].dt.dayofweek)[col]
    return pd.DataFrame({"mean": g.mean(), "sem": g.sem()}).reindex(range(7))


def compute_acf(y: np.ndarray, nlags: int) -> np.ndarray:
    y = pd.Series(y).interpolate(limit_direction="both").to_numpy()
    return acf(y, nlags=nlags, fft=True, missing="conservative")


def lomb_scargle_periodogram(t_hours: np.ndarray, y: np.ndarray):
    """Return (periods_hours, power). Uses normalized Lomb-Scargle."""
    mask = ~np.isnan(y)
    t = t_hours[mask].astype(float)
    yv = y[mask].astype(float)
    yv = yv - yv.mean()
    periods = np.geomspace(LS_PERIOD_MIN_H, LS_PERIOD_MAX_H, LS_N_FREQS)
    angular_freqs = 2.0 * np.pi / periods
    power = lombscargle(t, yv, angular_freqs, normalize=True)
    return periods, power


def _polar_bar(ax, values, sem, n_spokes, title):
    theta = np.linspace(0.0, 2 * np.pi, n_spokes, endpoint=False)
    width = 2 * np.pi / n_spokes
    vals = np.nan_to_num(values, nan=0.0)
    ax.bar(theta, vals, width=width, bottom=0.0, alpha=0.75,
           edgecolor="black", linewidth=0.5)
    if sem is not None:
        ax.errorbar(theta, vals, yerr=np.nan_to_num(sem, nan=0.0),
                    fmt="none", ecolor="black", capsize=2, linewidth=0.8)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title, fontsize=10)


def _label_hours(ax):
    ticks = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)


def _label_dow(ax):
    ticks = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=8)


def plot_patient(name: str, df: pd.DataFrame, sig_col: str, out_path: str):
    df = df.copy()
    df["y_log"] = np.log1p(df["y"])

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.45)

    # Row 1: raw
    ax1 = fig.add_subplot(gs[0, 0], projection="polar")
    hp_raw = hour_of_day_profile(df, "y")
    _polar_bar(ax1, hp_raw["mean"].values, hp_raw["sem"].values, 24, "Hour-of-day (raw)")
    _label_hours(ax1)

    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    dp_raw = dow_profile(df, "y")
    _polar_bar(ax2, dp_raw["mean"].values, dp_raw["sem"].values, 7, "Day-of-week (raw)")
    _label_dow(ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    a_raw = compute_acf(df["y"].to_numpy(), ACF_MAX_LAG_HOURS)
    lags = np.arange(len(a_raw)) / 24.0
    ax3.plot(lags, a_raw, lw=0.8)
    for d in (1, 7):
        ax3.axvline(d, color="red", ls="--", lw=0.7, alpha=0.6)
    ax3.axhline(0, color="black", lw=0.5)
    ax3.set_xlabel("lag (days)")
    ax3.set_ylabel("ACF")
    ax3.set_title("Autocorrelation (raw)")

    ax4 = fig.add_subplot(gs[0, 3])
    t_h = (df["t"] - df["t"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
    periods, pwr = lomb_scargle_periodogram(t_h, df["y"].to_numpy())
    ax4.semilogx(periods / 24.0, pwr, lw=0.8)
    for d in (0.5, 1, 7):
        ax4.axvline(d, color="red", ls="--", lw=0.7, alpha=0.6)
    ax4.set_xlabel("period (days)")
    ax4.set_ylabel("LS power")
    ax4.set_title("Lomb-Scargle (raw)")

    # Row 2: log1p
    ax5 = fig.add_subplot(gs[1, 0], projection="polar")
    hp_log = hour_of_day_profile(df, "y_log")
    _polar_bar(ax5, hp_log["mean"].values, hp_log["sem"].values, 24, "Hour-of-day (log1p)")
    _label_hours(ax5)

    ax6 = fig.add_subplot(gs[1, 1], projection="polar")
    dp_log = dow_profile(df, "y_log")
    _polar_bar(ax6, dp_log["mean"].values, dp_log["sem"].values, 7, "Day-of-week (log1p)")
    _label_dow(ax6)

    ax7 = fig.add_subplot(gs[1, 2])
    a_log = compute_acf(df["y_log"].to_numpy(), ACF_MAX_LAG_HOURS)
    ax7.plot(np.arange(len(a_log)) / 24.0, a_log, lw=0.8, color="C1")
    for d in (1, 7):
        ax7.axvline(d, color="red", ls="--", lw=0.7, alpha=0.6)
    ax7.axhline(0, color="black", lw=0.5)
    ax7.set_xlabel("lag (days)")
    ax7.set_ylabel("ACF")
    ax7.set_title("Autocorrelation (log1p)")

    ax8 = fig.add_subplot(gs[1, 3])
    periods_l, pwr_l = lomb_scargle_periodogram(t_h, df["y_log"].to_numpy())
    ax8.semilogx(periods_l / 24.0, pwr_l, lw=0.8, color="C1")
    for d in (0.5, 1, 7):
        ax8.axvline(d, color="red", ls="--", lw=0.7, alpha=0.6)
    ax8.set_xlabel("period (days)")
    ax8.set_ylabel("LS power")
    ax8.set_title("Lomb-Scargle (log1p)")

    n_h = df["y"].notna().sum()
    span_days = (df["t"].iloc[-1] - df["t"].iloc[0]).total_seconds() / 86400.0
    fig.suptitle(
        f"{name}  |  {sig_col}  |  n={n_h} hourly samples  |  span={span_days:.0f} d",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_grid(patient_data, out_path: str, transform: str):
    """Side-by-side comparison: rows = patients, cols = [hour polar, dow polar, ACF, LS]."""
    n = len(patient_data)
    fig = plt.figure(figsize=(16, 3.2 * n))
    gs = fig.add_gridspec(n, 4, hspace=0.55, wspace=0.4)

    col = "y" if transform == "raw" else "y_log"
    color = "C0" if transform == "raw" else "C1"

    for i, (name, df, sig_col) in enumerate(patient_data):
        ax_h = fig.add_subplot(gs[i, 0], projection="polar")
        hp = hour_of_day_profile(df, col)
        _polar_bar(ax_h, hp["mean"].values, hp["sem"].values, 24, "")
        _label_hours(ax_h)
        ax_h.set_ylabel(name, labelpad=25, fontsize=10, rotation=0, ha="right")

        ax_d = fig.add_subplot(gs[i, 1], projection="polar")
        dp = dow_profile(df, col)
        _polar_bar(ax_d, dp["mean"].values, dp["sem"].values, 7, "")
        _label_dow(ax_d)

        ax_a = fig.add_subplot(gs[i, 2])
        a = compute_acf(df[col].to_numpy(), ACF_MAX_LAG_HOURS)
        ax_a.plot(np.arange(len(a)) / 24.0, a, lw=0.8, color=color)
        for d in (1, 7):
            ax_a.axvline(d, color="red", ls="--", lw=0.6, alpha=0.6)
        ax_a.axhline(0, color="black", lw=0.5)
        ax_a.set_xlabel("lag (days)")
        ax_a.set_ylabel("ACF")

        ax_l = fig.add_subplot(gs[i, 3])
        t_h = (df["t"] - df["t"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
        periods, pwr = lomb_scargle_periodogram(t_h, df[col].to_numpy())
        ax_l.semilogx(periods / 24.0, pwr, lw=0.8, color=color)
        for d in (0.5, 1, 7):
            ax_l.axvline(d, color="red", ls="--", lw=0.6, alpha=0.6)
        ax_l.set_xlabel("period (days)")
        ax_l.set_ylabel("LS power")

        if i == 0:
            ax_h.set_title("Hour-of-day")
            ax_d.set_title("Day-of-week")
            ax_a.set_title("Autocorrelation")
            ax_l.set_title("Lomb-Scargle")

    fig.suptitle(f"Periodicity baseline — all patients ({transform})", fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    by_kind = {}
    for s in rns.iter_series():
        df = s["data"].copy()
        if df["y"].notna().sum() < 24:
            continue
        df["y_log"] = np.log1p(df["y"].clip(lower=0))
        kind_dir = os.path.join(OUT_DIR, s["signal_kind"])
        os.makedirs(kind_dir, exist_ok=True)
        sig_desc = f"{s['signal_kind']} / {s['pattern_col']}  (sat {s['sat_frac']*100:.0f}%)"
        out_path = os.path.join(kind_dir, f"{s['label']}_periodicity.png")
        plot_patient(s["label"], df, sig_desc, out_path)
        by_kind.setdefault(s["signal_kind"], []).append((s, df))
        print(f"[ok] {s['label']:14s} {s['signal_kind']:9s} -> {out_path}")

    # comparison grids per signal kind (detection: representative detector only)
    for kind, entries in by_kind.items():
        if kind == "detection":
            entries = [(s, df) for (s, df) in entries if s["is_representative"]]
        entries = sorted(entries, key=lambda e: e[0]["patient"])
        patient_data = [(s["label"], df, s["pattern_col"]) for (s, df) in entries]
        for transform in ("raw", "log1p"):
            grid = os.path.join(OUT_DIR, f"comparison_grid_{kind}_{transform}.png")
            plot_comparison_grid(patient_data, grid, transform)
        print(f"[ok] {kind} comparison grids -> {OUT_DIR}")


if __name__ == "__main__":
    main()
