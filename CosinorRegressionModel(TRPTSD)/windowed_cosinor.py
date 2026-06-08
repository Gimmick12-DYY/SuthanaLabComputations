"""
Time-resolved (sliding-window) 24-h cosinor, per patient.

Motivation
----------
The static two-harmonic cosinor (two_harmonic_cosinor.py) fits a single
fixed-phase rhythm across each patient's entire record.  For RNS_A_B2 and
RNS_G_A2 that worked (R^2 ~ 0.2), but RNS_B_B2 / RNS_D_* / RNS_F_B1 came out
nearly flat (R^2 < 0.02).  A static fit cannot distinguish two very different
situations:

  (a) genuinely arrhythmic signal, vs.
  (b) a real 24-h rhythm whose *phase drifts* over weeks/months, so a single
      fixed-phase cosine averages it away.

This script fits an independent single-harmonic 24-h cosinor inside a short
sliding window (default 3 days, stepped 1 day) and tracks how amplitude and
acrophase move over time.  A drifting acrophase with locally-high amplitude is
the signature of case (b); scattered phase with low amplitude is case (a).

Per-window amplitude gets a moving-block-bootstrap 95% CI (block = 24 h) so the
hourly autocorrelation does not make the rhythm look more certain than it is.

Model (per window, on 7-day-detrended y):
    y(t) = M + b*cos(w24 t) + g*sin(w24 t) + eps,   w24 = 2*pi/24
    A      = sqrt(b^2 + g^2)
    acro_h = ((-atan2(g, b)) mod 2pi) * 24 / (2pi)

Phase stability per patient is summarized by the mean resultant length R of the
amplitude-weighted per-window acrophases (R~1 => locked phase, R~0 => drifting).

Outputs (plots/windowed_cosinor/):
  - <patient>_windowed.png       4-panel: A(t)+CI, acro(t), polar acro, R^2(t)
  - comparison_grid.png          amplitude(t) and acrophase(t) for all patients
  - windowed_fits.csv            one row per (patient, window)
  - phase_stability_summary.csv  one row per patient
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "Data_Cosinor_09.15.25")
OUT_DIR = os.path.join(HERE, "plots", "windowed_cosinor")
os.makedirs(OUT_DIR, exist_ok=True)

DETREND_WINDOW_H = 24 * 7          # 7-day rolling mean detrend (matches prior scripts)
WINDOW_DAYS = 3                    # sliding-window length
STEP_DAYS = 1                      # step between windows
MIN_FRAC_PRESENT = 0.5             # require >=50% of hours present in a window
N_BOOTSTRAP = 200                  # moving-block bootstrap reps for amplitude CI
BLOCK_H = 24                       # bootstrap block length (hours)
RNG = np.random.default_rng(0)

W24 = 2 * np.pi / 24.0


# -------------------------- data + detrend -------------------------- #

def load_patient(csv_path: str):
    name = os.path.basename(csv_path).replace("_Full_output.csv", "")
    df = pd.read_csv(csv_path)
    sig_col = [c for c in df.columns if c.startswith("Pattern")][0]
    df = df[["Region start time", sig_col]].copy()
    df["t"] = pd.to_datetime(df["Region start time"])
    df = df.rename(columns={sig_col: "y"}).sort_values("t").reset_index(drop=True)
    return name, df[["t", "y"]], sig_col


def detrend_rolling(y: pd.Series, window_h: int = DETREND_WINDOW_H) -> pd.Series:
    trend = y.rolling(window=window_h, min_periods=window_h // 4, center=True).mean()
    return y - trend


# -------------------------- cosinor in a window -------------------------- #

def _design(t_hours: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.ones_like(t_hours),
        np.cos(W24 * t_hours),
        np.sin(W24 * t_hours),
    ])


def _amp_acro(b: float, g: float):
    # acrophase = clock time (h) of the rhythm's PEAK.  For
    # y = b cos(wt) + g sin(wt), the maximum is at wt = atan2(g, b).
    amp = float(np.hypot(b, g))
    acro = float((np.arctan2(g, b) % (2 * np.pi)) * 24.0 / (2 * np.pi))
    return amp, acro


def fit_window(t_hours: np.ndarray, y: np.ndarray):
    """Fit single-harmonic 24-h cosinor. Returns dict or None if too few points."""
    mask = ~np.isnan(y)
    if mask.sum() < 12:                       # need a minimum to fit 3 params
        return None
    t = t_hours[mask]
    yv = y[mask]
    X = _design(t)
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    yhat = X @ coef
    resid = yv - yhat
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    ss_res = float(np.sum(resid ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    amp, acro = _amp_acro(coef[1], coef[2])
    return {
        "n": int(mask.sum()), "mesor": float(coef[0]),
        "amp": amp, "acro_h": acro, "r2": float(r2),
        "_X": X, "_yhat": yhat, "_resid": resid,
    }


def bootstrap_amp_ci(X, yhat, resid, n_boot=N_BOOTSTRAP, block=BLOCK_H):
    """Moving-block bootstrap of residuals -> 95% CI on amplitude."""
    n = len(resid)
    if n < block * 2:
        return np.nan, np.nan
    n_blocks = int(np.ceil(n / block))
    max_start = n - block
    amps = np.empty(n_boot)
    for b in range(n_boot):
        starts = RNG.integers(0, max_start + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        y_boot = yhat + resid[idx]
        coef, *_ = np.linalg.lstsq(X, y_boot, rcond=None)
        amps[b] = np.hypot(coef[1], coef[2])
    return float(np.percentile(amps, 2.5)), float(np.percentile(amps, 97.5))


def run_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Slide the window across the record; return per-window dataframe."""
    t0 = df["t"].iloc[0]
    t_h_all = (df["t"] - t0).dt.total_seconds().to_numpy() / 3600.0
    y_dt = detrend_rolling(df["y"]).to_numpy()
    span_h = t_h_all[-1]

    win_h = WINDOW_DAYS * 24
    step_h = STEP_DAYS * 24
    need = MIN_FRAC_PRESENT * win_h

    rows = []
    start = 0.0
    while start + win_h <= span_h + step_h:
        end = start + win_h
        sel = (t_h_all >= start) & (t_h_all < end)
        if sel.sum() >= need:
            tw = t_h_all[sel]
            yw = y_dt[sel]
            fit = fit_window(tw, yw)
            if fit is not None and np.isfinite(fit["amp"]):
                lo, hi = bootstrap_amp_ci(fit["_X"], fit["_yhat"], fit["_resid"])
                rows.append({
                    "win_start_day": start / 24.0,
                    "center_t": t0 + pd.Timedelta(hours=start + win_h / 2),
                    "n": fit["n"], "mesor": fit["mesor"],
                    "amp": fit["amp"], "amp_ci_lo": lo, "amp_ci_hi": hi,
                    "acro_h": fit["acro_h"], "r2": fit["r2"],
                })
        start += step_h
    return pd.DataFrame(rows)


# -------------------------- phase stability summary -------------------------- #

def phase_stability(win_df: pd.DataFrame) -> dict:
    """Amplitude-weighted circular concentration of per-window acrophases."""
    if win_df.empty:
        return {"n_windows": 0, "mean_acro_h": np.nan, "resultant_R": np.nan,
                "median_amp": np.nan, "frac_amp_ci_above_0": np.nan}
    ang = win_df["acro_h"].to_numpy() * (2 * np.pi / 24.0)
    w = win_df["amp"].to_numpy()
    w = np.where(np.isfinite(w), w, 0.0)
    if w.sum() == 0:
        w = np.ones_like(w)
    C = np.sum(w * np.cos(ang)) / w.sum()
    S = np.sum(w * np.sin(ang)) / w.sum()
    R = float(np.hypot(C, S))
    mean_ang = np.arctan2(S, C) % (2 * np.pi)
    mean_acro = float(mean_ang * 24.0 / (2 * np.pi))
    # fraction of windows whose amplitude CI lower bound is > 0 (locally rhythmic)
    lo = win_df["amp_ci_lo"].to_numpy()
    frac = float(np.mean(lo > 0)) if np.isfinite(lo).any() else np.nan
    return {
        "n_windows": int(len(win_df)),
        "mean_acro_h": mean_acro,
        "resultant_R": R,
        "median_amp": float(win_df["amp"].median()),
        "frac_amp_ci_above_0": frac,
    }


# -------------------------- plotting -------------------------- #

def plot_patient(name: str, sig_col: str, win_df: pd.DataFrame, summ: dict,
                 out_path: str):
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    days = win_df["win_start_day"].to_numpy() + WINDOW_DAYS / 2.0

    # (1) amplitude over time with bootstrap CI band
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
    ax.legend(fontsize=8, loc="upper right")

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

    # (3) polar acrophase scatter, radius=amplitude, color=time
    ax = fig.add_subplot(gs[1, 0], projection="polar")
    ang = win_df["acro_h"].to_numpy() * (2 * np.pi / 24.0)
    sc2 = ax.scatter(ang, win_df["amp"], c=days, cmap="plasma", s=16, alpha=0.8)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 3)], fontsize=8)
    # mean resultant vector
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
        f"Windowed 24-h cosinor — {name}  |  {sig_col}  |  "
        f"{WINDOW_DAYS}-d window / {STEP_DAYS}-d step  |  "
        f"{summ['n_windows']} windows  |  phase R={summ['resultant_R']:.2f}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_grid(results: list[tuple[str, pd.DataFrame, dict]], out_path: str):
    n = len(results)
    fig = plt.figure(figsize=(14, 2.8 * n))
    gs = fig.add_gridspec(n, 2, hspace=0.55, wspace=0.25)
    for i, (name, win_df, summ) in enumerate(results):
        days = win_df["win_start_day"].to_numpy() + WINDOW_DAYS / 2.0

        ax_a = fig.add_subplot(gs[i, 0])
        ax_a.plot(days, win_df["amp"], color="C0", lw=1.0)
        if win_df["amp_ci_lo"].notna().any():
            ax_a.fill_between(days, win_df["amp_ci_lo"], win_df["amp_ci_hi"],
                              color="C0", alpha=0.2)
        ax_a.set_ylabel(name, fontsize=10)
        if i == 0:
            ax_a.set_title("24-h amplitude over time")
        if i == n - 1:
            ax_a.set_xlabel("time (days from start)")

        ax_p = fig.add_subplot(gs[i, 1])
        sc = ax_p.scatter(days, win_df["acro_h"], c=win_df["amp"],
                          cmap="viridis", s=8)
        ax_p.set_ylim(0, 24)
        ax_p.set_yticks(range(0, 25, 6))
        ax_p.axhline(summ["mean_acro_h"], color="C3", ls="--", lw=0.8)
        ax_p.text(0.98, 0.92, f"R={summ['resultant_R']:.2f}", transform=ax_p.transAxes,
                  ha="right", va="top", fontsize=8, color="C3")
        if i == 0:
            ax_p.set_title("24-h acrophase over time (color=amp)")
        if i == n - 1:
            ax_p.set_xlabel("time (days from start)")

    fig.suptitle("Windowed 24-h cosinor — all patients", fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------- main -------------------------- #

def main():
    csvs = sorted(glob.glob(os.path.join(DATA_DIR, "*_Full_output.csv")))
    if not csvs:
        raise SystemExit(f"No CSVs found in {DATA_DIR}")

    results = []
    all_win_rows = []
    summ_rows = []
    for path in csvs:
        name, df, sig_col = load_patient(path)
        win_df = run_windows(df)
        summ = phase_stability(win_df)
        results.append((name, win_df, summ))

        wd = win_df.copy()
        wd.insert(0, "patient", name)
        all_win_rows.append(wd)
        summ_rows.append({"patient": name, "signal_col": sig_col, **summ})

        out_path = os.path.join(OUT_DIR, f"{name}_windowed.png")
        plot_patient(name, sig_col, win_df, summ, out_path)
        print(f"[ok] {name}  windows={summ['n_windows']:4d}  "
              f"median_amp={summ['median_amp']:.2f}  "
              f"phase_R={summ['resultant_R']:.2f}  "
              f"mean_acro={summ['mean_acro_h']:.1f}h  "
              f"frac_rhythmic={summ['frac_amp_ci_above_0']:.2f}")

    grid = os.path.join(OUT_DIR, "comparison_grid.png")
    plot_comparison_grid(results, grid)

    win_csv = os.path.join(OUT_DIR, "windowed_fits.csv")
    pd.concat(all_win_rows, ignore_index=True).drop(
        columns=[c for c in ["_X", "_yhat", "_resid"] if c in []], errors="ignore"
    ).to_csv(win_csv, index=False)

    summ_csv = os.path.join(OUT_DIR, "phase_stability_summary.csv")
    pd.DataFrame(summ_rows).to_csv(summ_csv, index=False)

    print(f"[ok] comparison grid -> {grid}")
    print(f"[ok] per-window csv  -> {win_csv}")
    print(f"[ok] summary csv     -> {summ_csv}")


if __name__ == "__main__":
    main()
