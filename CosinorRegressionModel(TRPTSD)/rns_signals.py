"""
Shared data access for the RNS rhythmicity pipeline (Data_06.09.26).

Centralizes everything that varies per (patient, signal) so the analysis
scripts (windowed_cosinor, rhythmicity_test, mixed_effects_cosinor, ...) can
iterate over a uniform stream of series instead of hard-coding one column.

Signals exposed:
  - "detection"  : the Pattern {A,B} Channel {1,2} hourly detection counts.
                   Per patient we analyze every *active* pattern (>= MIN_NONZERO
                   nonzero hours); the one with the most nonzero hours is the
                   "representative" detector used for cross-patient summaries.
  - "stim_rx"    : "Episode starts with RX" — actual stimulation delivered.

Data notes baked in here (from M. Vallejo Martelo, PTSD project email):
  - Values are hourly counts on an 8-bit register: 254 is the ceiling
    (CAP), so value >= 254 is right-censored / saturated.  We do NOT discard
    saturated samples; callers get a per-series / per-window saturation
    fraction to report alongside fits.
  - RNS_H correction: replace "Pattern A Channel 1" with "Pattern B Channel 1"
    for all timestamps before 2025-10-24.
"""

from __future__ import annotations

import os
import glob

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "Data_06.09.26")

CAP = 254                      # >= CAP is saturated / right-censored
MIN_NONZERO = 500              # a detection pattern is "active" above this

TIME_COL = "Region start time"
RX_COL = "Episode starts with RX"
DETECTION_PATTERNS = [
    "Pattern A Channel 1", "Pattern B Channel 1",
    "Pattern A Channel 2", "Pattern B Channel 2",
]

# RNS_H: A1 values before this date are replaced with B1 values.
RNS_H_SWAP_BEFORE = pd.Timestamp("2025-10-24")


def detector_label(pattern_col: str) -> str:
    """'Pattern B Channel 2' -> 'B2'."""
    parts = pattern_col.split()
    return f"{parts[1]}{parts[3]}"


def patient_id(path: str) -> str:
    """'RNS_A_B2_Complete.csv' -> 'RNS_A' (detector suffix is unreliable now)."""
    toks = os.path.basename(path).split("_")
    return f"{toks[0]}_{toks[1]}"


def load_complete(path: str):
    """Return (patient_id, dataframe) with parsed 't' and the RNS_H fix applied."""
    df = pd.read_csv(path, low_memory=False)
    df["t"] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)

    pid = patient_id(path)
    if pid == "RNS_H":
        a1, b1 = "Pattern A Channel 1", "Pattern B Channel 1"
        mask = df["t"] < RNS_H_SWAP_BEFORE
        if mask.any() and a1 in df.columns and b1 in df.columns:
            df.loc[mask, a1] = df.loc[mask, b1].to_numpy()
    return pid, df


def active_detectors(df: pd.DataFrame, min_nonzero: int = MIN_NONZERO):
    out = []
    for c in DETECTION_PATTERNS:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").fillna(0)
        if int((s != 0).sum()) >= min_nonzero:
            out.append(c)
    return out


def representative_detector(df: pd.DataFrame):
    """Active detection pattern with the most nonzero hours (cross-patient proxy)."""
    best, best_nz = None, -1
    for c in DETECTION_PATTERNS:
        if c not in df.columns:
            continue
        nz = int((pd.to_numeric(df[c], errors="coerce").fillna(0) != 0).sum())
        if nz > best_nz:
            best, best_nz = c, nz
    return best


def saturation_fraction(y, cap: int = CAP) -> float:
    """Fraction of present samples at or above the register ceiling."""
    y = np.asarray(y, dtype=float)
    n = int(np.isfinite(y).sum())
    if n == 0:
        return np.nan
    return float(np.sum(y >= cap) / n)


def iter_series(signal_set=("detection", "stim_rx"), detection_scope: str = "active"):
    """Yield one dict per (patient, signal) series.

    Keys: patient, signal_kind ('detection'|'stim_rx'), detector ('B2'|'RX'|...),
    pattern_col, label ('RNS_B_B2'|'RNS_B_RX'), data (DataFrame[t, y]),
    sat_frac, is_representative.
    """
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_Complete.csv"))):
        pid, df = load_complete(path)
        rep = representative_detector(df)

        if "detection" in signal_set:
            cols = (active_detectors(df) if detection_scope == "active"
                    else [c for c in DETECTION_PATTERNS if c in df.columns])
            for c in cols:
                y = pd.to_numeric(df[c], errors="coerce")
                lab = detector_label(c)
                yield {
                    "patient": pid, "signal_kind": "detection",
                    "detector": lab, "pattern_col": c,
                    "label": f"{pid}_{lab}",
                    "data": pd.DataFrame({"t": df["t"], "y": y}),
                    "sat_frac": saturation_fraction(y),
                    "is_representative": (c == rep),
                }

        if "stim_rx" in signal_set and RX_COL in df.columns:
            y = pd.to_numeric(df[RX_COL], errors="coerce")
            yield {
                "patient": pid, "signal_kind": "stim_rx",
                "detector": "RX", "pattern_col": RX_COL,
                "label": f"{pid}_RX",
                "data": pd.DataFrame({"t": df["t"], "y": y}),
                "sat_frac": saturation_fraction(y),
                "is_representative": True,
            }


if __name__ == "__main__":
    # quick inventory
    print(f"DATA_DIR = {DATA_DIR}")
    for s in iter_series():
        d = s["data"]
        rep = "*" if s["is_representative"] else " "
        print(f"  {rep} {s['label']:14s} kind={s['signal_kind']:9s} "
              f"n={len(d):6d}  nonnull={int(d['y'].notna().sum()):6d}  "
              f"sat={s['sat_frac']*100:5.1f}%  "
              f"span={d['t'].min().date()}->{d['t'].max().date()}")
