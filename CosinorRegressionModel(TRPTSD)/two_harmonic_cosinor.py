"""
Two-harmonic (24 h + 12 h) cosinor fit, per patient, on detrended residuals.

Motivation (from periodicity_baseline.py / periodicity_diagnostics.py):
  - Several patients showed clear 12-h harmonic peaks in Lomb-Scargle
    alongside the 24-h fundamental (most clearly RNS_A_B2, RNS_G_A2).
  - Detrending stabilizes the mesor so the harmonic structure is what
    actually gets fit, instead of a slow drift dominating R^2.

Model (per patient, on detrended y):
    y(t) = M + b1 cos(w24 t) + g1 sin(w24 t)
              + b2 cos(w12 t) + g2 sin(w12 t) + eps
    A_k = sqrt(b_k^2 + g_k^2)
    phi_k = atan2(g_k, b_k)
    acrophase_hours_k = ((-phi_k) mod 2pi) * (period_k / 2pi)

Compares a one-harmonic (24 h only) fit against the two-harmonic fit
via a nested F-test on the 12-h component.

Outputs:
  - plots/two_harmonic_cosinor/<patient>_fit.svg  (4-panel diagnostic)
  - plots/two_harmonic_cosinor/comparison_grid.svg
  - plots/two_harmonic_cosinor/fit_summary.csv
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import acf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "Data_Cosinor_09.15.25")
OUT_DIR = os.path.join(HERE, "plots", "two_harmonic_cosinor")
os.makedirs(OUT_DIR, exist_ok=True)

DETREND_WINDOW_H = 24 * 7
ACF_LAGS_H = 24 * 7


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


# -------------------------- cosinor fitting -------------------------- #

@dataclass
class CosinorFit:
    name: str
    n_obs: int
    mesor: float
    amp_24h: float
    acrophase_h_24h: float
    amp_12h: float
    acrophase_h_12h: float
    r2_one: float
    r2_two: float
    f_stat_12h: float
    p_value_12h: float
    coefs_one: np.ndarray
    coefs_two: np.ndarray
    t_hours: np.ndarray
    y_detrended: np.ndarray
    y_hat_one: np.ndarray
    y_hat_two: np.ndarray


def _design_matrix(t_hours: np.ndarray, two_harmonic: bool) -> np.ndarray:
    w24 = 2 * np.pi / 24.0
    w12 = 2 * np.pi / 12.0
    cols = [np.ones_like(t_hours), np.cos(w24 * t_hours), np.sin(w24 * t_hours)]
    if two_harmonic:
        cols += [np.cos(w12 * t_hours), np.sin(w12 * t_hours)]
    return np.column_stack(cols)


def _amp_acrophase(b_cos: float, b_sin: float, period_h: float):
    # acrophase = clock time (h) of the PEAK.  max of b cos(wt)+g sin(wt) is at
    # wt = atan2(g, b), so peak time = atan2(b_sin, b_cos) * period / (2 pi).
    amp = float(np.hypot(b_cos, b_sin))
    phi = np.arctan2(b_sin, b_cos)                      # angle of the peak
    acrophase = (phi % (2 * np.pi)) * period_h / (2 * np.pi)
    return amp, float(acrophase)


def fit_cosinor(name: str, t_hours: np.ndarray, y: np.ndarray) -> CosinorFit:
    mask = ~np.isnan(y)
    t = t_hours[mask]
    yv = y[mask]

    X1 = _design_matrix(t, two_harmonic=False)
    X2 = _design_matrix(t, two_harmonic=True)

    c1, *_ = np.linalg.lstsq(X1, yv, rcond=None)
    c2, *_ = np.linalg.lstsq(X2, yv, rcond=None)

    yhat1 = X1 @ c1
    yhat2 = X2 @ c2
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    ss_res1 = float(np.sum((yv - yhat1) ** 2))
    ss_res2 = float(np.sum((yv - yhat2) ** 2))
    r2_one = 1 - ss_res1 / ss_tot if ss_tot > 0 else np.nan
    r2_two = 1 - ss_res2 / ss_tot if ss_tot > 0 else np.nan

    # Nested F-test: does adding the 12 h component significantly reduce SSR?
    n = len(yv)
    p1, p2 = X1.shape[1], X2.shape[1]
    df_num = p2 - p1
    df_den = n - p2
    f_stat = ((ss_res1 - ss_res2) / df_num) / (ss_res2 / df_den)
    p_val = float(1.0 - stats.f.cdf(f_stat, df_num, df_den)) if df_den > 0 else np.nan

    amp24, acro24 = _amp_acrophase(c2[1], c2[2], 24.0)
    amp12, acro12 = _amp_acrophase(c2[3], c2[4], 12.0)

    # Recompute predictions on the full (non-masked) timebase for plotting.
    Xfull1 = _design_matrix(t_hours, two_harmonic=False)
    Xfull2 = _design_matrix(t_hours, two_harmonic=True)
    yhat1_full = Xfull1 @ c1
    yhat2_full = Xfull2 @ c2

    return CosinorFit(
        name=name,
        n_obs=int(n),
        mesor=float(c2[0]),
        amp_24h=amp24, acrophase_h_24h=acro24,
        amp_12h=amp12, acrophase_h_12h=acro12,
        r2_one=float(r2_one), r2_two=float(r2_two),
        f_stat_12h=float(f_stat), p_value_12h=p_val,
        coefs_one=c1, coefs_two=c2,
        t_hours=t_hours, y_detrended=y,
        y_hat_one=yhat1_full, y_hat_two=yhat2_full,
    )


# -------------------------- plotting -------------------------- #

def _profile_24h(t_hours: np.ndarray, y: np.ndarray):
    """Empirical hour-of-day mean and SEM of a series."""
    hours = (t_hours % 24).astype(int)
    df = pd.DataFrame({"h": hours, "y": y}).dropna()
    g = df.groupby("h")["y"]
    return g.mean().reindex(range(24)).to_numpy(), g.sem().reindex(range(24)).to_numpy()


def _predicted_24h_shape(fit: CosinorFit):
    """One-day predicted shape from the fitted coefficients (without mesor)."""
    hh = np.arange(0, 24.0, 0.25)
    w24, w12 = 2 * np.pi / 24.0, 2 * np.pi / 12.0
    c1 = fit.coefs_one
    c2 = fit.coefs_two
    shape_one = c1[0] + c1[1] * np.cos(w24 * hh) + c1[2] * np.sin(w24 * hh)
    shape_two = (
        c2[0]
        + c2[1] * np.cos(w24 * hh) + c2[2] * np.sin(w24 * hh)
        + c2[3] * np.cos(w12 * hh) + c2[4] * np.sin(w12 * hh)
    )
    return hh, shape_one, shape_two


def plot_patient_fit(fit: CosinorFit, sig_col: str, out_path: str):
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # (1) 24-h shape: empirical vs one-harmonic vs two-harmonic
    ax = fig.add_subplot(gs[0, 0])
    mean_h, sem_h = _profile_24h(fit.t_hours, fit.y_detrended)
    ax.errorbar(np.arange(24), mean_h, yerr=sem_h, fmt="o", ms=4,
                color="black", label="empirical (detrended)")
    hh, sh1, sh2 = _predicted_24h_shape(fit)
    ax.plot(hh, sh1, color="C0", lw=2, label=f"1-harm (R²={fit.r2_one:.3f})")
    ax.plot(hh, sh2, color="C3", lw=2, label=f"2-harm (R²={fit.r2_two:.3f})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("detrended signal")
    ax.set_title("24-h shape: data vs fits")
    ax.legend(fontsize=8, loc="best")
    ax.set_xticks(range(0, 24, 3))

    # (2) Residual ACF for two-harmonic fit
    ax = fig.add_subplot(gs[0, 1])
    resid = fit.y_detrended - fit.y_hat_two
    a = acf(pd.Series(resid).interpolate(limit_direction="both"),
            nlags=ACF_LAGS_H, fft=True)
    ax.plot(np.arange(len(a)) / 24.0, a, lw=0.9, color="C3")
    for d in (1, 2, 0.5):
        ax.axvline(d, color="red", ls="--", lw=0.6, alpha=0.5)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("lag (days)")
    ax.set_ylabel("residual ACF")
    ax.set_title("Residuals: ACF (2-harm fit)")

    # (3) Coefficient / fit summary text
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    txt = (
        f"Patient:           {fit.name}\n"
        f"Signal column:     {sig_col}\n"
        f"n (used in fit):   {fit.n_obs}\n"
        f"Mesor:             {fit.mesor:+.3f}\n"
        f"\n24-h component\n"
        f"  amplitude:       {fit.amp_24h:.3f}\n"
        f"  acrophase:       {fit.acrophase_h_24h:5.2f} h\n"
        f"\n12-h component\n"
        f"  amplitude:       {fit.amp_12h:.3f}\n"
        f"  acrophase:       {fit.acrophase_h_12h:5.2f} h\n"
        f"\nR^2 (1-harm):      {fit.r2_one:.4f}\n"
        f"R^2 (2-harm):      {fit.r2_two:.4f}\n"
        f"ΔR^2 (12-h adds):  {fit.r2_two - fit.r2_one:.4f}\n"
        f"\nNested F-test for 12-h component\n"
        f"  F-stat:          {fit.f_stat_12h:.2f}\n"
        f"  p-value:         {fit.p_value_12h:.2e}\n"
    )
    ax.text(0.0, 1.0, txt, family="monospace", va="top", ha="left", fontsize=10)
    ax.set_title("Fit summary")

    # (4) Time-series slice (first 14 days) with both fits overlaid
    ax = fig.add_subplot(gs[1, :])
    days_to_show = 14
    n_show = min(days_to_show * 24, len(fit.y_detrended))
    ax.plot(fit.t_hours[:n_show] / 24.0, fit.y_detrended[:n_show],
            color="black", lw=0.6, alpha=0.7, label="detrended y")
    ax.plot(fit.t_hours[:n_show] / 24.0, fit.y_hat_one[:n_show],
            color="C0", lw=1.0, label="1-harm fit")
    ax.plot(fit.t_hours[:n_show] / 24.0, fit.y_hat_two[:n_show],
            color="C3", lw=1.0, label="2-harm fit")
    ax.axhline(0, color="gray", lw=0.4)
    ax.set_xlabel("time (days from start)")
    ax.set_ylabel("detrended signal")
    ax.set_title(f"First {days_to_show} days: detrended series with fitted cosinors")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Two-harmonic cosinor — {fit.name}", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_grid(fits: list[CosinorFit], out_path: str):
    n = len(fits)
    fig = plt.figure(figsize=(13, 2.8 * n))
    gs = fig.add_gridspec(n, 3, hspace=0.6, wspace=0.4,
                          width_ratios=[1, 1, 1])

    for i, fit in enumerate(fits):
        ax_s = fig.add_subplot(gs[i, 0])
        mean_h, sem_h = _profile_24h(fit.t_hours, fit.y_detrended)
        ax_s.errorbar(np.arange(24), mean_h, yerr=sem_h, fmt="o", ms=3,
                      color="black", label="empirical")
        hh, sh1, sh2 = _predicted_24h_shape(fit)
        ax_s.plot(hh, sh1, color="C0", lw=1.8, label="1-harm")
        ax_s.plot(hh, sh2, color="C3", lw=1.8, label="2-harm")
        ax_s.axhline(0, color="gray", lw=0.4)
        ax_s.set_xticks(range(0, 24, 6))
        ax_s.set_ylabel(fit.name, fontsize=10)
        if i == 0:
            ax_s.set_title("24-h shape: data vs fits")
            ax_s.legend(fontsize=7, loc="best")
        if i == n - 1:
            ax_s.set_xlabel("hour of day")

        ax_r = fig.add_subplot(gs[i, 1])
        resid = fit.y_detrended - fit.y_hat_two
        a = acf(pd.Series(resid).interpolate(limit_direction="both"),
                nlags=ACF_LAGS_H, fft=True)
        ax_r.plot(np.arange(len(a)) / 24.0, a, lw=0.8, color="C3")
        for d in (1, 0.5):
            ax_r.axvline(d, color="red", ls="--", lw=0.5, alpha=0.5)
        ax_r.axhline(0, color="black", lw=0.4)
        ax_r.set_ylim(-0.5, 1.05)
        if i == 0:
            ax_r.set_title("Residual ACF (2-harm)")
        if i == n - 1:
            ax_r.set_xlabel("lag (days)")

        ax_t = fig.add_subplot(gs[i, 2])
        ax_t.axis("off")
        msg = (
            f"R² 1-harm: {fit.r2_one:.3f}\n"
            f"R² 2-harm: {fit.r2_two:.3f}   (ΔR²={fit.r2_two - fit.r2_one:.3f})\n"
            f"A24={fit.amp_24h:.2f}  acro24={fit.acrophase_h_24h:.1f} h\n"
            f"A12={fit.amp_12h:.2f}  acro12={fit.acrophase_h_12h:.1f} h\n"
            f"12-h F-test  p={fit.p_value_12h:.2e}"
        )
        ax_t.text(0.0, 0.5, msg, family="monospace", fontsize=9, va="center")

    fig.suptitle("Two-harmonic cosinor — comparison across patients", fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary_csv(fits: list[CosinorFit], out_path: str):
    rows = []
    for f in fits:
        rows.append({
            "patient": f.name,
            "n_obs": f.n_obs,
            "mesor": f.mesor,
            "amp_24h": f.amp_24h,
            "acrophase_h_24h": f.acrophase_h_24h,
            "amp_12h": f.amp_12h,
            "acrophase_h_12h": f.acrophase_h_12h,
            "r2_one_harmonic": f.r2_one,
            "r2_two_harmonic": f.r2_two,
            "delta_r2": f.r2_two - f.r2_one,
            "f_stat_12h": f.f_stat_12h,
            "p_value_12h": f.p_value_12h,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


# -------------------------- main -------------------------- #

def main():
    csvs = sorted(glob.glob(os.path.join(DATA_DIR, "*_Full_output.csv")))
    if not csvs:
        raise SystemExit(f"No CSVs found in {DATA_DIR}")

    fits = []
    for path in csvs:
        name, df, sig_col = load_patient(path)
        t_h = (df["t"] - df["t"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
        y_dt = detrend_rolling(df["y"]).to_numpy()
        fit = fit_cosinor(name, t_h, y_dt)
        fits.append(fit)

        out_path = os.path.join(OUT_DIR, f"{name}_fit.svg")
        plot_patient_fit(fit, sig_col, out_path)
        print(f"[ok] {name}  R²(1)={fit.r2_one:.3f}  R²(2)={fit.r2_two:.3f}  "
              f"ΔR²={fit.r2_two - fit.r2_one:.3f}  "
              f"p(12h)={fit.p_value_12h:.2e}  -> {out_path}")

    grid_path = os.path.join(OUT_DIR, "comparison_grid.svg")
    plot_comparison_grid(fits, grid_path)
    csv_path = os.path.join(OUT_DIR, "fit_summary.csv")
    save_summary_csv(fits, csv_path)
    print(f"[ok] comparison grid -> {grid_path}")
    print(f"[ok] summary csv     -> {csv_path}")


if __name__ == "__main__":
    main()
