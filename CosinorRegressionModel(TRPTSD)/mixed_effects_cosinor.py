"""
Mixed-effects (population) cosinor across patients + cross-patient summary.

This is the cross-patient layer.  Layers 1-2 worked one patient at a time:
  - windowed_cosinor.py        : per-patient phase stability (resultant R)
  - rhythmicity_test.py        : per-patient AR(1) significance + phase-locking
                                 index (PLI = static_A24 / windowed_median_A24)

Here we fit ONE linearized two-harmonic cosinor across all patients at once
with patient as a random effect, to ask: is there a shared population-level
circadian rhythm, and how much does each patient deviate from it?

Model (statsmodels MixedLM), on 7-day-detrended hourly y:
    y_ij = (M + u0_i)
         + (B24 + u_c_i) cos(w24 t) + (G24 + u_s_i) sin(w24 t)   <- random slopes
         +  B12        cos(w12 t) +  G12        sin(w12 t)        <- fixed only
         + eps
    population 24-h amp/acro from (B24, G24);
    patient i amp/acro from (B24+u_c_i, G24+u_s_i)  -> partially-pooled (shrunk).

IMPORTANT caveats (stated, not hidden):
  - Only 6 patients => random-effect (co)variance is imprecise.
  - Hourly residuals are autocorrelated, so MixedLM standard errors / Wald
    p-values are anticonservative.  Use this model for *structure* (population
    rhythm, per-patient deviation, variance partition); rely on
    rhythmicity_test.py for significance.
  - B/D/F drift in phase (low resultant R), so their "fixed-phase" patient
    amplitude is small; the shared rhythm is dominated by the phase-locked
    patients (A_B2, G_A2).  The BLUP deviations + PLI make this explicit.

Outputs (plots/mixed_effects_cosinor/):
  - population_waveform.png   pop 1-/2-harmonic 24-h shape + per-patient curves
  - acrophase_clock.png       per-patient 24-h acrophase vectors + population
  - amplitude_forest.png      static A24 (data, Layer 2 CI) vs shrunken BLUP
  - mixedlm_summary.txt       full model summary
  - patient_effects.csv       per-patient population+random -> amp/acro, joined
                              with Layer 1/2 metrics
  - population_summary.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

import rns_signals as rns

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "plots", "mixed_effects_cosinor")
L1_CSV = os.path.join(HERE, "plots", "windowed_cosinor", "phase_stability_summary.csv")
L2_CSV = os.path.join(HERE, "plots", "rhythmicity_test", "rhythmicity_summary.csv")
os.makedirs(OUT_DIR, exist_ok=True)

DETREND_WINDOW_H = 24 * 7
W24 = 2 * np.pi / 24.0
W12 = 2 * np.pi / 12.0


def detrend_rolling(y: pd.Series, window_h: int = DETREND_WINDOW_H) -> pd.Series:
    trend = y.rolling(window=window_h, min_periods=window_h // 4, center=True).mean()
    return y - trend


def amp_acro(b, g, period_h):
    # acrophase = clock time (h) of the PEAK: max of b cos(wt)+g sin(wt) is at
    # wt = atan2(g, b).
    amp = float(np.hypot(b, g))
    acro = float((np.arctan2(g, b) % (2 * np.pi)) * period_h / (2 * np.pi))
    return amp, acro


def build_long_df(signal_kind: str):
    """Pool all patients for one signal kind.

    For 'detection' we take each patient's representative detector (one series
    per patient); for 'stim_rx' the single RX series.  Returns (data, label_map)
    where label_map[patient] is the series label used.
    """
    frames = []
    label_map = {}
    for s in rns.iter_series(signal_set=(signal_kind,)):
        if signal_kind == "detection" and not s["is_representative"]:
            continue
        df = s["data"]
        th = (df["t"] - df["t"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
        y_dt = detrend_rolling(df["y"]).to_numpy()
        sub = pd.DataFrame({
            "patient": s["patient"], "y": y_dt,
            "cos24": np.cos(W24 * th), "sin24": np.sin(W24 * th),
            "cos12": np.cos(W12 * th), "sin12": np.sin(W12 * th),
        }).dropna(subset=["y"])
        frames.append(sub)
        label_map[s["patient"]] = s["label"]
    return pd.concat(frames, ignore_index=True), label_map


def fit_mixed(data):
    # Diagonal variance components: independent random 24-h cos/sin slopes per
    # patient, no random intercept (the 7-day detrend already centers each
    # patient's mesor at ~0).  With only 6 groups an unstructured 2x2 RE
    # covariance sits on the boundary / non-PD Hessian; the diagonal VC form
    # is identifiable and converges cleanly.
    vc = {"re_cos24": "0 + cos24", "re_sin24": "0 + sin24"}
    md = smf.mixedlm(
        "y ~ cos24 + sin24 + cos12 + sin12",
        data, groups=data["patient"],
        vc_formula=vc,
    )
    return md.fit(method=["lbfgs"], maxiter=2000)


# -------------------------- extract effects -------------------------- #

def patient_effects(mdf, patients):
    fe = mdf.fe_params
    re = mdf.random_effects                      # dict: patient -> Series
    rows = []
    pop_b24, pop_g24 = fe["cos24"], fe["sin24"]
    def _re_value(series, term):
        # VC random effects are indexed like 're_cos24[cos24]'; match by term.
        for idx, val in series.items():
            if term in str(idx):
                return float(val)
        return 0.0

    for p in patients:
        r = re[p]
        u_c = _re_value(r, "cos24")
        u_s = _re_value(r, "sin24")
        b_i, g_i = pop_b24 + u_c, pop_g24 + u_s
        a_i, acro_i = amp_acro(b_i, g_i, 24.0)
        rows.append({"patient": p, "blup_A24": a_i, "blup_acro24_h": acro_i,
                     "re_cos24": u_c, "re_sin24": u_s})
    return pd.DataFrame(rows)


def population_summary(mdf):
    fe = mdf.fe_params
    A24, acro24 = amp_acro(fe["cos24"], fe["sin24"], 24.0)
    A12, acro12 = amp_acro(fe["cos12"], fe["sin12"], 12.0)
    # variance components (diagonal VC): one variance per vc key, in dict order
    vc_names = list(getattr(mdf.model, "exog_vc").names) \
        if hasattr(getattr(mdf.model, "exog_vc", None), "names") else ["re_cos24", "re_sin24"]
    vcomp = np.asarray(mdf.vcomp, dtype=float)
    vc_var = {name: float(vcomp[i]) for i, name in enumerate(vc_names)}
    out = {
        "pop_mesor": float(fe["Intercept"]),
        "pop_A24": A24, "pop_acro24_h": acro24,
        "pop_A12": A12, "pop_acro12_h": acro12,
        "resid_var": float(mdf.scale),
        "re_var_cos24": vc_var.get("re_cos24", np.nan),
        "re_var_sin24": vc_var.get("re_sin24", np.nan),
    }
    return out


# -------------------------- plotting -------------------------- #

def plot_population_waveform(mdf, pat_eff, out_path):
    fe = mdf.fe_params
    hh = np.arange(0, 24.01, 0.25)
    c24, s24 = np.cos(W24 * hh), np.sin(W24 * hh)
    c12, s12 = np.cos(W12 * hh), np.sin(W12 * hh)
    pop1 = fe["cos24"] * c24 + fe["sin24"] * s24
    pop2 = pop1 + fe["cos12"] * c12 + fe["sin12"] * s12

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for _, r in pat_eff.iterrows():
        curve = r["blup_A24_cos"] * c24 + r["blup_A24_sin"] * s24
        ax.plot(hh, curve, lw=1.0, alpha=0.55, label=r["patient"])
    ax.plot(hh, pop1, color="black", lw=3, label="population 24-h")
    ax.plot(hh, pop2, color="C3", lw=2.5, ls="--", label="population 24+12-h")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("hour of day")
    ax.set_ylabel("detrended signal")
    ax.set_title("Mixed-effects cosinor: population rhythm + per-patient (BLUP) 24-h curves")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_acrophase_clock(pat_eff, pop, out_path):
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 3)])

    amax = pat_eff["blup_A24"].max()
    for _, r in pat_eff.iterrows():
        ang = r["blup_acro24_h"] * (2 * np.pi / 24.0)
        # radius = phase stability (Layer 1 resultant R) if present, else 1
        rad = r["resultant_R"] if np.isfinite(r.get("resultant_R", np.nan)) else 1.0
        ax.annotate("", xy=(ang, rad), xytext=(0, 0),
                    arrowprops=dict(width=1.5, headwidth=7, alpha=0.8))
        ax.text(ang, rad + 0.04, f"{r['patient']}\nA={r['blup_A24']:.0f}",
                fontsize=7, ha="center")
    pop_ang = pop["pop_acro24_h"] * (2 * np.pi / 24.0)
    ax.annotate("", xy=(pop_ang, 1.0), xytext=(0, 0),
                arrowprops=dict(width=3, headwidth=11, color="C3"))
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-patient 24-h acrophase (arrow length = phase stability R)\n"
                 f"red = population acrophase ({pop['pop_acro24_h']:.1f} h)", fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_amplitude_forest(pat_eff, pop, out_path):
    pe = pat_eff.sort_values("static_A24")
    y = np.arange(len(pe))
    fig, ax = plt.subplots(figsize=(9, 0.8 * len(pe) + 2))
    # data estimate (Layer 2 static A24 + bootstrap CI)
    xerr = np.clip(np.vstack([
        (pe["static_A24"] - pe["static_A24_ci_lo"]).to_numpy(),
        (pe["static_A24_ci_hi"] - pe["static_A24"]).to_numpy(),
    ]), 0, None)
    ax.errorbar(pe["static_A24"], y - 0.12, xerr=xerr, fmt="o", color="C0",
                capsize=3, label="static A24 (data ± block-bootstrap 95% CI)")
    # shrunken mixed-model BLUP
    ax.scatter(pe["blup_A24"], y + 0.12, color="C3", marker="s",
               label="mixed-model BLUP A24 (partial pooling)")
    ax.axvline(pop["pop_A24"], color="black", ls="--", lw=1.2,
               label=f"population A24 = {pop['pop_A24']:.1f}")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p}\nPLI={pli:.2f}, R={r:.2f}"
                        for p, pli, r in zip(pe["patient"], pe["phase_locking_index"],
                                             pe["resultant_R"])])
    ax.set_xlabel("24-h amplitude")
    ax.set_title("Per-patient 24-h amplitude: data vs partially-pooled estimate")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------- main -------------------------- #

def _join_layers(pat_eff, signal_kind):
    """Merge Layer 1 (phase stability) and Layer 2 (AR(1) test) for this kind."""
    for col in ["resultant_R", "mean_acro_h", "median_amp",
                "static_A24", "static_A24_ci_lo", "static_A24_ci_hi",
                "phase_locking_index", "p_static_A24", "p_windowed_A24"]:
        pat_eff[col] = np.nan
    if os.path.exists(L1_CSV):
        l1 = pd.read_csv(L1_CSV)
        l1 = l1[l1["signal_kind"] == signal_kind]
        if signal_kind == "detection":
            l1 = l1[l1["is_representative"]]
        pat_eff = pat_eff.drop(columns=["resultant_R", "mean_acro_h", "median_amp"]).merge(
            l1[["patient", "resultant_R", "mean_acro_h", "median_amp"]],
            on="patient", how="left")
    if os.path.exists(L2_CSV):
        l2 = pd.read_csv(L2_CSV)
        l2 = l2[l2["signal_kind"] == signal_kind]
        if signal_kind == "detection":
            l2 = l2[l2["is_representative"]]
        pat_eff = pat_eff.drop(columns=[
            "static_A24", "static_A24_ci_lo", "static_A24_ci_hi",
            "phase_locking_index", "p_static_A24", "p_windowed_A24"]).merge(
            l2[["patient", "static_A24", "static_A24_ci_lo", "static_A24_ci_hi",
                "phase_locking_index", "p_static_A24", "p_windowed_A24"]],
            on="patient", how="left")
    return pat_eff


def run_signal_kind(signal_kind: str):
    data, label_map = build_long_df(signal_kind)
    patients = sorted(data["patient"].unique())
    out_dir = os.path.join(OUT_DIR, signal_kind)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[info] {signal_kind}: {len(data)} rows, {len(patients)} patients "
          f"({', '.join(label_map[p] for p in patients)})")

    mdf = fit_mixed(data)
    with open(os.path.join(out_dir, "mixedlm_summary.txt"), "w") as fh:
        fh.write(mdf.summary().as_text())

    pop = population_summary(mdf)
    pat_eff = patient_effects(mdf, patients)
    pat_eff["label"] = pat_eff["patient"].map(label_map)
    fe = mdf.fe_params
    pat_eff["blup_A24_cos"] = fe["cos24"] + pat_eff["re_cos24"]
    pat_eff["blup_A24_sin"] = fe["sin24"] + pat_eff["re_sin24"]
    pat_eff = _join_layers(pat_eff, signal_kind)

    kind_title = ("detections (representative detector)" if signal_kind == "detection"
                  else "Episode starts with RX")
    plot_population_waveform(mdf, pat_eff,
                             os.path.join(out_dir, "population_waveform.png"))
    plot_acrophase_clock(pat_eff, pop, os.path.join(out_dir, "acrophase_clock.png"))
    if pat_eff["static_A24"].notna().any():
        plot_amplitude_forest(pat_eff, pop, os.path.join(out_dir, "amplitude_forest.png"))

    pat_eff.to_csv(os.path.join(out_dir, "patient_effects.csv"), index=False)
    pd.DataFrame([{"signal_kind": signal_kind, **pop}]).to_csv(
        os.path.join(out_dir, "population_summary.csv"), index=False)

    print(f"[ok] {signal_kind} ({kind_title}) population 24-h: A={pop['pop_A24']:.2f}  "
          f"acro={pop['pop_acro24_h']:.1f} h  |  12-h: A={pop['pop_A12']:.2f}  "
          f"acro={pop['pop_acro12_h']:.1f} h  -> {out_dir}")


def main():
    for signal_kind in ("detection", "stim_rx"):
        run_signal_kind(signal_kind)


if __name__ == "__main__":
    main()
