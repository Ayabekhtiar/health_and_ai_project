"""Build one row per patient from the clinical and mutation tables.

Only information available at the diagnostic work-up is used: blood counts,
marrow blast percentage, karyotype and the targeted sequencing panel.
CENTER is kept for validation and monitoring only; it is never a model input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cytogenetics as cg

CLINICAL_COLS = ["ID", "CENTER", "BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT", "CYTOGENETICS"]
OUTCOME_COLS = ["OS_YEARS", "OS_STATUS"]

# Variant classes that do not alter the protein: not counted as driver mutations.
NON_DRIVER_EFFECTS = {"synonymous_codon", "3_prime_UTR_variant", "2KB_upstream_variant",
                      "stop_retained_variant"}

IPSSM_MAIN_GENES = ["NPM1", "RUNX1", "NRAS", "ETV6", "IDH2", "CBL", "EZH2", "U2AF1",
                    "SRSF2", "DNMT3A", "ASXL1", "KRAS"]
IPSSM_RES_GENES = ["BCOR", "BCORL1", "CEBPA", "ETNK1", "GATA2", "GNB1", "IDH1", "NF1",
                   "PHF6", "PPM1D", "PRPF8", "PTPN11", "SETBP1", "STAG2", "WT1"]
# Other recurrent genes (>= 2% of patients) offered to the refitted model.
EXTRA_GENES = ["TET2", "ZRSR2", "DDX41", "CUX1", "JAK2", "MPL", "SH2B3"]


def load(clinical_path: str, mutation_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    clin = pd.read_csv(clinical_path, sep="\t")
    muts = pd.read_csv(mutation_path, sep="\t")
    missing = [c for c in CLINICAL_COLS if c not in clin.columns]
    if missing:
        raise ValueError(f"clinical file lacks columns: {missing}")
    return clin, muts


def mutation_features(muts: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    m = muts[~muts["EFFECT"].isin(NON_DRIVER_EFFECTS)].copy()
    m["GENE"] = m["GENE"].replace({"KMT2A": "MLL"})
    out = pd.DataFrame(index=pd.Index(ids, name="ID"))

    genes = set(IPSSM_MAIN_GENES + IPSSM_RES_GENES + EXTRA_GENES + ["SF3B1", "TP53", "FLT3"])
    present = m.groupby("GENE")["ID"].apply(set)
    for g in sorted(genes):
        out[g] = out.index.isin(present.get(g, set())).astype(float)

    ptd = m[(m["GENE"] == "MLL") & (m["EFFECT"] == "PTD")]["ID"]
    out["MLL_PTD"] = out.index.isin(set(ptd)).astype(float)

    tp53 = m[m["GENE"] == "TP53"].groupby("ID").agg(n=("GENE", "size"), vaf=("VAF", "max"))
    out["TP53_n"] = tp53["n"].reindex(out.index).fillna(0)
    out["TP53_maxvaf"] = tp53["vaf"].reindex(out.index).fillna(0)

    n_driver = m.groupby("ID").size()
    out["N_MUT"] = n_driver.reindex(out.index).fillna(0)
    out["MAX_VAF"] = m.groupby("ID")["VAF"].max().reindex(out.index).fillna(0)
    out["NO_MUT_RECORDED"] = (out["N_MUT"] == 0).astype(float)
    listed = m.assign(G=np.where((m["GENE"] == "MLL") & (m["EFFECT"] == "PTD"), "KMT2A-PTD", m["GENE"]))
    out["GENES_MUTATED"] = listed.groupby("ID")["G"].apply(lambda g: ", ".join(sorted(set(g)))) \
        .reindex(out.index).fillna("")
    return out


def build(clin: pd.DataFrame, muts: pd.DataFrame) -> pd.DataFrame:
    """Patient-level table with raw inputs, derived variables and scope flags."""
    df = clin.copy().set_index("ID")
    for col in ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    kts = [cg.parse(x) for x in df["CYTOGENETICS"]]
    df["CYTO_CAT"] = [k.category for k in kts]
    df["CYTOVEC"] = df["CYTO_CAT"].map(cg.CYTO_CODE).astype(float)
    df["CYTO_N_ABN"] = [k.n_abn if k.parsed else np.nan for k in kts]
    for flag in ["del5q", "del7_7q", "minus_7", "del17_17p", "complex", "plus_8", "abn3q"]:
        df["CYTO_" + flag] = [float(k.flags[flag]) if k.parsed else np.nan for k in kts]

    df = df.join(mutation_features(muts, df.index))

    # IPSS-M derived molecular variables (definitions from the official ipssm package)
    del5q = df["CYTO_del5q"]
    isolated = (df["CYTO_del7_7q"] == 0) & (df["CYTO_complex"] == 0)
    df["SF3B1_5q"] = np.where(df["SF3B1"] == 0, 0.0,
                     np.where(del5q.isna(), np.nan, ((del5q == 1) & isolated).astype(float)))
    co = df[["SRSF2", "STAG2", "BCOR", "BCORL1", "RUNX1", "NRAS"]].max(axis=1)
    df["SF3B1_alpha"] = np.where(df["SF3B1"] == 0, 0.0,
                        np.where(df["SF3B1_5q"].isna(), np.nan,
                                 ((df["SF3B1_5q"] == 0) & (co == 0)).astype(float)))
    loh = (df["TP53_maxvaf"] > 0.55) | (df["CYTO_del17_17p"] == 1)
    loh_unknown = df["CYTO_del17_17p"].isna() & (df["TP53_maxvaf"] <= 0.55)
    tp53multi = np.where(df["TP53_n"] >= 2, 1.0,
                np.where(df["TP53_n"] == 0, 0.0,
                np.where(loh, 1.0, np.where(loh_unknown, np.nan, 0.0))))
    df["TP53multi"] = tp53multi
    df["nRes2"] = df[IPSSM_RES_GENES].sum(axis=1).clip(upper=2)

    # transformed clinical variables (IPSS-M conventions)
    df["HB1"] = df["HB"].clip(4, 20)
    df["BLAST5"] = df["BM_BLAST"].clip(upper=20) / 5
    df["TRANSF_PLT100"] = df["PLT"].clip(upper=250) / 100
    df["LOG_ANC"] = np.log1p(df["ANC"].clip(upper=20))
    df["LOG_WBC"] = np.log(df["WBC"].clip(0.5, 50))
    df["LOG_MONO"] = np.log1p(df["MONOCYTES"].clip(upper=10))

    df = df.join(scope_flags(df))
    return df


def scope_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Intended-use checks. A patient outside scope still gets a score, but the
    output says so and the score must not be used for the transplant decision."""
    f = pd.DataFrame(index=df.index)
    f["OUT_BLAST_GE20"] = df["BM_BLAST"] >= 20          # AML by WHO blast threshold
    f["OUT_WBC_GE13"] = df["WBC"] >= 13                  # proliferative MDS/MPN
    f["MISSING_CORE"] = df[["BM_BLAST", "HB", "PLT"]].isna().any(axis=1)
    f["MISSING_KARYOTYPE"] = df["CYTO_CAT"].isna()
    f["CMML_LIKE"] = (df["MONOCYTES"] >= 1) & (df["MONOCYTES"] >= 0.1 * df["WBC"])
    f["IN_SCOPE"] = ~(f["OUT_BLAST_GE20"] | f["OUT_WBC_GE13"] | f["MISSING_CORE"])
    return f
