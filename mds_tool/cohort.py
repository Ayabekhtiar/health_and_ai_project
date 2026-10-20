"""Cohort definitions. Every exclusion is counted so the flow can be reported."""
from __future__ import annotations

import pandas as pd

from . import features, scores

DATA_CLIN = "df_clinical_train.tsv"
DATA_MUT = "mut_clinical_train.tsv"


def load_all(root: str = ".") -> pd.DataFrame:
    clin, muts = features.load(f"{root}/{DATA_CLIN}", f"{root}/{DATA_MUT}")
    df = scores.add_scores(features.build(clin, muts))
    return df


def flow(df: pd.DataFrame) -> tuple[list[tuple[str, int]], dict[str, pd.Series]]:
    steps, masks = [], {}
    m = pd.Series(True, index=df.index)
    steps.append(("Patients in the file", int(m.sum())))
    m &= df["OS_YEARS"].notna() & df["OS_STATUS"].notna()
    steps.append(("with a recorded outcome", int(m.sum())))
    m &= df["OS_YEARS"] > 0
    steps.append(("with follow-up > 0 (53 censored at day 0 removed)", int(m.sum())))
    masks["has_outcome"] = m.copy()
    m &= ~df["OUT_BLAST_GE20"]
    steps.append(("marrow blasts < 20 % (AML excluded)", int(m.sum())))
    m &= ~df["OUT_WBC_GE13"]
    steps.append(("WBC < 13 x10^9/L (proliferative MDS/MPN excluded)", int(m.sum())))
    m &= ~df["MISSING_CORE"]
    steps.append(("blasts, Hb and platelets recorded  = INTENDED-USE COHORT", int(m.sum())))
    masks["in_scope"] = m.copy()
    m &= df["IPSSR_score"].notna() & df["ANC"].notna()
    steps.append(("karyotype readable and ANC recorded = HEAD-TO-HEAD COHORT (IPSS-R computable)",
                  int(m.sum())))
    masks["head_to_head"] = m.copy()
    masks["out_of_scope"] = masks["has_outcome"] & ~masks["in_scope"]
    return steps, masks
