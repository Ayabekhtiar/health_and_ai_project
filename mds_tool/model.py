"""Candidate models, all returning a risk score and a 2-year death probability.

The deployable model is a (penalised) Cox regression stored as plain numbers in
JSON: one weight per feature, a centring value, and the baseline 2-year
survival. Its prediction can be recomputed by hand, and each feature's
contribution is weight x (value - average), as for the scores already in use.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.util import Surv

from . import HORIZON_YEARS
from .features import EXTRA_GENES
from .metrics import harrell_c
from .scores import BETA

IPSSM_VARS = list(BETA)  # the 22 IPSS-M model variables
EXTENDED_VARS = IPSSM_VARS + ["LOG_ANC", "LOG_WBC", "LOG_MONO", "CYTO_N_ABN", "N_MUT",
                              "CYTO_MISSING"] + EXTRA_GENES

LABELS = {
    "HB1": "Haemoglobin (g/dL)", "TRANSF_PLT100": "Platelets (/100, capped 250)",
    "BLAST5": "Marrow blasts (/5 %, capped 20)", "CYTOVEC": "IPSS-R karyotype group (0-4)",
    "TP53multi": "TP53 multi-hit", "FLT3": "FLT3", "MLL_PTD": "KMT2A (MLL) PTD",
    "SF3B1_5q": "SF3B1 with del(5q)", "SF3B1_alpha": "SF3B1 (isolated, favourable)",
    "nRes2": "Other driver genes (0-2)", "LOG_ANC": "Neutrophils (log)",
    "LOG_WBC": "White cells (log)", "LOG_MONO": "Monocytes (log)",
    "CYTO_N_ABN": "Number of karyotype abnormalities", "N_MUT": "Number of driver mutations",
    "CYTO_MISSING": "Karyotype missing",
}


def surv_y(df):
    return Surv.from_arrays(df["OS_STATUS"].astype(bool).values, df["OS_YEARS"].astype(float).values)


def design(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = df.reindex(columns=[c for c in cols if c != "CYTO_MISSING"]).astype(float)
    if "CYTO_MISSING" in cols:
        x["CYTO_MISSING"] = df["CYTOVEC"].isna().astype(float)
    if "CYTO_N_ABN" in x:
        x["CYTO_N_ABN"] = x["CYTO_N_ABN"].clip(upper=5)
    if "N_MUT" in x:
        x["N_MUT"] = x["N_MUT"].clip(upper=8)
    return x[cols]


@dataclass
class LinearCox:
    """Cox model with fixed preprocessing, serialisable to JSON."""
    cols: list[str]
    l1_ratio: float = 0.5
    alpha: float | None = None
    impute: dict = field(default_factory=dict)
    mean: dict = field(default_factory=dict)
    scale: dict = field(default_factory=dict)
    coef: dict = field(default_factory=dict)
    lp_center: float = 0.0
    s0_horizon: float = np.nan
    name: str = "cox"

    # -------------------------------------------------------------- fit
    def _prep(self, df, fit=False):
        x = design(df, self.cols)
        if fit:
            self.impute = x.median().fillna(0).to_dict()
            for c in ["TP53multi", "SF3B1_5q", "SF3B1_alpha", "CYTOVEC"]:
                if c in x:
                    self.impute[c] = float(x[c].mean())
        x = x.fillna(self.impute)
        if fit:
            self.mean = x.mean().to_dict()
            self.scale = x.std().replace(0, 1).fillna(1).to_dict()
        return (x - pd.Series(self.mean)) / pd.Series(self.scale)

    def fit(self, df, groups=None, inner_folds=5):
        z = self._prep(df, fit=True)
        y = surv_y(df)
        if self.alpha is None:
            self.alpha = self._choose_alpha(z, y, df, groups, inner_folds)
        if self.alpha == 0:
            m = CoxPHSurvivalAnalysis(alpha=1e-4).fit(z.values, y)
            b = m.coef_
        else:
            m = CoxnetSurvivalAnalysis(l1_ratio=self.l1_ratio, alphas=[self.alpha],
                                       fit_baseline_model=False, max_iter=200000).fit(z.values, y)
            b = m.coef_[:, 0]
        self.coef = dict(zip(self.cols, map(float, b)))
        lp = z.values @ b
        self.lp_center = float(lp.mean())
        self.s0_horizon = _breslow_s0(df["OS_YEARS"].values, df["OS_STATUS"].values,
                                      lp - self.lp_center, HORIZON_YEARS)
        return self

    def _choose_alpha(self, z, y, df, groups, k):
        path = CoxnetSurvivalAnalysis(l1_ratio=self.l1_ratio, n_alphas=25, alpha_min_ratio=0.005,
                                      max_iter=200000).fit(z.values, y)
        alphas = path.alphas_
        groups = groups if groups is not None else np.arange(len(z)) % k
        cv = GroupKFold(n_splits=min(k, len(np.unique(groups))))
        scores = np.zeros(len(alphas))
        for tr, te in cv.split(z, groups=groups):
            m = CoxnetSurvivalAnalysis(l1_ratio=self.l1_ratio, alphas=alphas,
                                       max_iter=200000).fit(z.values[tr], y[tr])
            for j in range(len(alphas)):
                if j >= m.coef_.shape[1]:   # path stopped early: score as unusable
                    scores[j] = -np.inf
                    continue
                lp = z.values[te] @ m.coef_[:, j]
                scores[j] += harrell_c(df["OS_YEARS"].values[te], df["OS_STATUS"].values[te], lp)
        # one-standard-error spirit: largest penalty within 0.002 of the best mean C
        best = scores.max()
        ok = np.where(scores >= best - 0.002 * cv.get_n_splits())[0]
        return float(alphas[ok.min()])

    # ----------------------------------------------------------- predict
    def linear_predictor(self, df) -> np.ndarray:
        z = self._prep(df)
        b = np.array([self.coef[c] for c in self.cols])
        return z.values @ b - self.lp_center

    def risk_at_horizon(self, df) -> np.ndarray:
        return 1 - self.s0_horizon ** np.exp(self.linear_predictor(df))

    def contributions(self, df) -> pd.DataFrame:
        """Per-patient contribution of each feature to the log-hazard, relative to
        the average training patient (weight x standardised deviation)."""
        z = self._prep(df)
        return z * pd.Series(self.coef)

    # ------------------------------------------------------------- io
    def to_json(self, path, extra: dict | None = None):
        d = {k: getattr(self, k) for k in ["name", "cols", "l1_ratio", "alpha", "impute", "mean",
                                            "scale", "coef", "lp_center", "s0_horizon"]}
        d["horizon_years"] = HORIZON_YEARS
        d.update(extra or {})
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        keep = {k: d[k] for k in ["name", "cols", "l1_ratio", "alpha", "impute", "mean", "scale",
                                  "coef", "lp_center", "s0_horizon"]}
        m = cls(**keep)
        m.meta = {k: v for k, v in d.items() if k not in keep}
        return m


def _breslow_s0(time, event, lp, t) -> float:
    """Breslow baseline survival at t for centred linear predictor lp."""
    order = np.argsort(time)
    time, event, risk = time[order], event[order], np.exp(lp[order])
    rev_cum = np.cumsum(risk[::-1])[::-1]
    h = 0.0
    for i in np.where((event == 1) & (time <= t))[0]:
        h += 1.0 / rev_cum[np.searchsorted(time, time[i], side="left")]
    return float(np.exp(-h))


@dataclass
class ScoreRecalibrated:
    """Published score kept as is; a one-variable Cox model maps it to a 2-year risk."""
    score_col: str
    fill: float | None = None
    name: str = "score"

    def fit(self, df, groups=None):
        s = df[self.score_col].astype(float)
        self.fill = float(s.mean()) if self.fill is None else self.fill
        s = s.fillna(self.fill).values
        self.beta = float(CoxPHSurvivalAnalysis().fit(s[:, None], surv_y(df)).coef_[0])
        self.center = float(s.mean())
        lp = self.beta * (s - self.center)
        self.s0_horizon = _breslow_s0(df["OS_YEARS"].values, df["OS_STATUS"].values, lp,
                                      HORIZON_YEARS)
        return self

    def linear_predictor(self, df):
        return self.beta * (df[self.score_col].astype(float).fillna(self.fill).values - self.center)

    def risk_at_horizon(self, df):
        return 1 - self.s0_horizon ** np.exp(self.linear_predictor(df))


@dataclass
class Boosted:
    """Gradient-boosted Cox model: the 'is flexibility worth it?' check."""
    cols: list[str]
    name: str = "gbm"

    def fit(self, df, groups=None):
        x = design(df, self.cols)
        self.impute = x.median().fillna(0)
        self.m = GradientBoostingSurvivalAnalysis(n_estimators=300, learning_rate=0.03,
                                                  max_depth=2, subsample=0.8, min_samples_leaf=20,
                                                  random_state=0)
        self.m.fit(x.fillna(self.impute).values, surv_y(df))
        return self

    def _x(self, df):
        return design(df, self.cols).fillna(self.impute).values

    def linear_predictor(self, df):
        return self.m.predict(self._x(df))

    def risk_at_horizon(self, df):
        fns = self.m.predict_survival_function(self._x(df))
        return np.array([1 - f(HORIZON_YEARS) for f in fns])
