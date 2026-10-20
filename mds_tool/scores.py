"""Reference prognostic scores: IPSS-R (Greenberg 2012) and IPSS-M (Bernard 2022).

IPSS-M weights, centring values and cut-points are copied from the official
implementation (github.com/papaemmelab/ipssm, R/IPSSMmain.R). Missing inputs
are handled as in that package: best / mean / worst scenarios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- IPSS-R
IPSSR_CATS = ["Very low", "Low", "Intermediate", "High", "Very high"]
IPSSR_CUTS = [1.5, 3.0, 4.5, 6.0]


def _ipssr_points(blast, hb, plt, anc, cyto):
    b = np.select([blast <= 2, blast < 5, blast <= 10], [0, 1, 2], 3).astype(float)
    h = np.select([hb >= 10, hb >= 8], [0, 1], 1.5).astype(float)
    p = np.select([plt >= 100, plt >= 50], [0, 0.5], 1).astype(float)
    a = np.where(anc >= 0.8, 0, 0.5).astype(float)
    return cyto + b + h + p + a


def ipssr(df: pd.DataFrame) -> pd.DataFrame:
    """IPSS-R score. Missing ANC is scored 0 (best case) and flagged; missing
    karyotype gives a best-worst range and no point estimate."""
    blast, hb, plt = df["BM_BLAST"], df["HB"], df["PLT"]
    anc = df["ANC"].fillna(99)
    cyto = df["CYTOVEC"]
    score = _ipssr_points(blast, hb, plt, anc, cyto.fillna(np.nan))
    best = _ipssr_points(blast, hb, plt, anc, cyto.fillna(0))
    worst = _ipssr_points(blast, hb, plt, df["ANC"].fillna(0), cyto.fillna(4))
    core_missing = df[["BM_BLAST", "HB", "PLT"]].isna().any(axis=1)
    score = pd.Series(np.where(core_missing | cyto.isna(), np.nan, score), index=df.index)
    out = pd.DataFrame({"IPSSR_score": score,
                        "IPSSR_best": np.where(core_missing, np.nan, best),
                        "IPSSR_worst": np.where(core_missing, np.nan, worst)}, index=df.index)
    out["IPSSR_cat"] = pd.cut(out["IPSSR_score"], [-np.inf] + IPSSR_CUTS + [np.inf],
                              labels=IPSSR_CATS)
    return out


# ---------------------------------------------------------------- IPSS-M
BETA = {"HB1": -0.171, "TRANSF_PLT100": -0.222, "BLAST5": 0.352, "CYTOVEC": 0.287,
        "TP53multi": 1.18, "FLT3": 0.798, "MLL_PTD": 0.798, "SF3B1_5q": 0.504, "NPM1": 0.430,
        "RUNX1": 0.423, "NRAS": 0.417, "ETV6": 0.391, "IDH2": 0.379, "CBL": 0.295,
        "EZH2": 0.270, "U2AF1": 0.247, "SRSF2": 0.239, "DNMT3A": 0.221, "ASXL1": 0.213,
        "KRAS": 0.202, "SF3B1_alpha": -0.0794, "nRes2": 0.231}
MEAN = {"HB1": 9.87, "TRANSF_PLT100": 1.41, "BLAST5": 0.922, "CYTOVEC": 1.39,
        "TP53multi": 0.071, "FLT3": 0.0108, "MLL_PTD": 0.0247, "SF3B1_5q": 0.0166,
        "NPM1": 0.0112, "RUNX1": 0.126, "NRAS": 0.0362, "ETV6": 0.0216, "IDH2": 0.0429,
        "CBL": 0.0473, "EZH2": 0.0588, "U2AF1": 0.0866, "SRSF2": 0.158, "DNMT3A": 0.161,
        "ASXL1": 0.252, "KRAS": 0.0271, "SF3B1_alpha": 0.186, "nRes2": 0.388}
BEST = {**{k: 0.0 for k in BETA}, "HB1": 20.0, "TRANSF_PLT100": 2.5, "SF3B1_alpha": 1.0}
WORST = {**{k: 1.0 for k in BETA}, "HB1": 4.0, "TRANSF_PLT100": 0.0, "BLAST5": 4.0,
         "CYTOVEC": 4.0, "SF3B1_alpha": 0.0, "nRes2": 2.0}
# Plain-language name and unit of each IPSS-M variable, in the order of the report table.
FEATURES = {
    "HB1": ("Haemoglobin", "per g/dL"),
    "TRANSF_PLT100": ("Platelets", "per 100 x10^9/L, capped at 250"),
    "BLAST5": ("Marrow blasts", "per 5 %, capped at 20 %"),
    "CYTOVEC": ("Karyotype risk group", "per IPSS-R step, very good (0) to very poor (4)"),
    "TP53multi": ("TP53 multi-hit", "2+ TP53 mutations, or 1 with VAF > 55 % or 17p loss"),
    "FLT3": ("FLT3 mutation", "yes / no"),
    "MLL_PTD": ("KMT2A (MLL) partial tandem duplication", "yes / no"),
    "SF3B1_5q": ("SF3B1 with del(5q)", "SF3B1 mutated with isolated del(5q)"),
    "NPM1": ("NPM1", "yes / no"),
    "RUNX1": ("RUNX1", "yes / no"),
    "NRAS": ("NRAS", "yes / no"),
    "ETV6": ("ETV6", "yes / no"),
    "IDH2": ("IDH2", "yes / no"),
    "CBL": ("CBL", "yes / no"),
    "EZH2": ("EZH2", "yes / no"),
    "U2AF1": ("U2AF1", "yes / no"),
    "SRSF2": ("SRSF2", "yes / no"),
    "DNMT3A": ("DNMT3A", "yes / no"),
    "ASXL1": ("ASXL1", "yes / no"),
    "KRAS": ("KRAS", "yes / no"),
    "SF3B1_alpha": ("SF3B1, favourable form",
                    "SF3B1 mutated, no del(5q), none of SRSF2, STAG2, BCOR, BCORL1, RUNX1, NRAS"),
    "nRes2": ("Other driver genes",
              "count of BCOR, BCORL1, CEBPA, ETNK1, GATA2, GNB1, IDH1, NF1, PHF6, PPM1D, PRPF8, "
              "PTPN11, SETBP1, STAG2, WT1, capped at 2"),
}


def feature_table(local_slope: float | None = None) -> pd.DataFrame:
    """Published hazard ratio per unit, and the same effect after local recalibration
    (score slope `local_slope` per IPSS-M point; the published scale is ln 2 per point)."""
    rows = []
    for k, (name, unit) in FEATURES.items():
        r = dict(variable=k, feature=name, unit=unit, published_hazard_ratio=round(float(np.exp(BETA[k])), 2))
        if local_slope is not None:
            r["hazard_ratio_this_cohort"] = round(float(np.exp(BETA[k] * local_slope / np.log(2))), 2)
        rows.append(r)
    return pd.DataFrame(rows)


IPSSM_CATS = ["Very Low", "Low", "Moderate Low", "Moderate High", "High", "Very High"]
IPSSM_CUTS = [-1.5, -0.5, 0.0, 0.5, 1.5]
# Tentori et al., JCO 2024: immediate transplant prolongs life expectancy from
# Moderate High upwards; delaying is better for Low / Moderate Low.
IPSSM_TRANSPLANT_FROM = "Moderate High"


def _ipssm_score(x: pd.DataFrame, fill: dict) -> pd.Series:
    s = 0.0
    for k, b in BETA.items():
        s = s + (x[k].fillna(fill[k]) - MEAN[k]) * b
    return (s / np.log(2)).round(2)


def ipssm(df: pd.DataFrame) -> pd.DataFrame:
    x = df[list(BETA)].astype(float)
    out = pd.DataFrame(index=df.index)
    out["IPSSM_score"] = _ipssm_score(x, MEAN)
    out["IPSSM_best"] = _ipssm_score(x, BEST)
    out["IPSSM_worst"] = _ipssm_score(x, WORST)
    cut = [-np.inf] + IPSSM_CUTS + [np.inf]
    out["IPSSM_cat"] = pd.cut(out["IPSSM_score"], cut, labels=IPSSM_CATS)
    same = (pd.cut(out["IPSSM_best"], cut, labels=False)
            == pd.cut(out["IPSSM_worst"], cut, labels=False))
    out["IPSSM_uncertain"] = (out["IPSSM_worst"] - out["IPSSM_best"] > 1) & ~same
    out["IPSSM_transplant"] = out["IPSSM_score"] > 0.0   # Moderate High or above
    return out


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    return df.join(ipssr(df)).join(ipssm(df))
