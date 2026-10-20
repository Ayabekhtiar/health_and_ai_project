"""Check the IPSS-M port against the official package's own test patients
(papaemmelab/ipssm, tests/testthat/test_ipssm.R) and IPSS-R against hand scoring."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mds_tool import scores  # noqa: E402
from mds_tool.features import IPSSM_RES_GENES  # noqa: E402

# Inputs copied from inst/extdata/IPSSMexample.csv (pp347, pp564)
PP = pd.DataFrame([
    dict(ID="pp347", HB=9.6, PLT=281, BM_BLAST=9, del5q=0, del7_7q=0, complex=0, CYTOVEC=1,
         TP53multi=0, MLL_PTD=0, FLT3=0, ASXL1=1, BCOR=0, BCORL1=0, CBL=0, CEBPA=0, DNMT3A=0,
         ETV6=0, EZH2=0, IDH1=0, IDH2=0, KRAS=0, NF1=0, NPM1=0, NRAS=0, RUNX1=0, SETBP1=0,
         SF3B1=0, SRSF2=1, STAG2=1, U2AF1=0, ETNK1=np.nan, GATA2=0, GNB1=np.nan, PHF6=0,
         PPM1D=0, PRPF8=np.nan, PTPN11=0, WT1=0),
    dict(ID="pp564", HB=7.5, PLT=23, BM_BLAST=16, del5q=1, del7_7q=1, complex=1, CYTOVEC=4,
         TP53multi=1, MLL_PTD=0, FLT3=0, ASXL1=0, BCOR=0, BCORL1=0, CBL=0, CEBPA=0, DNMT3A=0,
         ETV6=0, EZH2=0, IDH1=0, IDH2=0, KRAS=0, NF1=1, NPM1=0, NRAS=0, RUNX1=0, SETBP1=0,
         SF3B1=0, SRSF2=0, STAG2=0, U2AF1=0, ETNK1=0, GATA2=0, GNB1=0, PHF6=0, PPM1D=0,
         PRPF8=0, PTPN11=0, WT1=0),
]).set_index("ID")


def _nres2(row, scenario):
    r = row[IPSSM_RES_GENES]
    m, s, ns = r.isna().sum(), r.notna().sum(), r.sum()
    return {"best": min(ns, 2), "worst": min(ns + m, 2),
            "mean": min(ns, 2) + max((2 - ns) / 2 * m / (s + m) * 0.388, 0)}[scenario]


def test_ipssm_official_examples():
    x = PP.copy()
    x["HB1"] = x["HB"].clip(4, 20)
    x["BLAST5"] = x["BM_BLAST"].clip(upper=20) / 5
    x["TRANSF_PLT100"] = x["PLT"].clip(upper=250) / 100
    x["SF3B1_5q"] = 0.0
    x["SF3B1_alpha"] = 0.0
    expected = {"pp347": {"mean": 0.31, "best": 0.30, "worst": 0.63},
                "pp564": {"mean": 4.55, "best": 4.55, "worst": 4.55}}
    fills = {"mean": scores.MEAN, "best": scores.BEST, "worst": scores.WORST}
    for pid, exp in expected.items():
        for scen, val in exp.items():
            row = x.loc[[pid]].copy()
            row["nRes2"] = _nres2(x.loc[pid], scen)
            got = float(scores._ipssm_score(row, fills[scen]).iloc[0])
            assert abs(got - val) < 0.011, f"{pid} {scen}: expected {val}, got {got}"


def test_ipssr_hand_scored():
    # blasts 4% (1) + Hb 9 (1) + PLT 40 (1) + ANC 0.5 (0.5) + Poor karyotype (3) = 6.5 -> Very high
    df = pd.DataFrame(dict(BM_BLAST=[4, 1], HB=[9, 12], PLT=[40, 200], ANC=[0.5, 2.0],
                           CYTOVEC=[3, 1]))
    r = scores.ipssr(df)
    assert list(r["IPSSR_score"]) == [6.5, 1.0]
    assert list(r["IPSSR_cat"].astype(str)) == ["Very high", "Very low"]


if __name__ == "__main__":
    test_ipssm_official_examples(); test_ipssr_hand_scored()
    print("OK: IPSS-M matches official test values; IPSS-R matches hand scoring")
