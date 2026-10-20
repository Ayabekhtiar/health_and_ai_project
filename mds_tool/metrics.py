"""Censoring-aware evaluation at a stated horizon.

Patients censored before the horizon have an unknown 2-year status (Session 1,
"when has follow-up truly ended?"). They are never dropped or counted as
survivors: every proportion below is a Kaplan-Meier estimate, and C-index /
AUC / Brier use inverse probability of censoring weights (sksurv).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sksurv.metrics import (brier_score, concordance_index_censored,
                            concordance_index_ipcw, cumulative_dynamic_auc)
from sksurv.util import Surv


def km_surv(time, event, t: float, return_se: bool = False):
    """Kaplan-Meier survival probability at time t (and Greenwood SE)."""
    time, event = np.asarray(time, float), np.asarray(event, float)
    if len(time) == 0:
        return (np.nan, np.nan) if return_se else np.nan
    ut, inv = np.unique(time, return_inverse=True)
    d = np.bincount(inv, weights=event)
    c = np.bincount(inv)
    at_risk = len(time) - np.concatenate([[0], np.cumsum(c)[:-1]])
    m = (ut <= t) & (d > 0)
    s = float(np.prod(1 - d[m] / at_risk[m]))
    if not return_se:
        return s
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.nansum(d[m] / (at_risk[m] * (at_risk[m] - d[m])))
    return s, s * np.sqrt(g)


def km_curve(time, event, grid):
    return np.array([km_surv(time, event, t) for t in grid])


def harrell_c(time, event, risk) -> float:
    return concordance_index_censored(np.asarray(event, bool), np.asarray(time, float),
                                      np.asarray(risk, float))[0]


def uno_c(time_tr, event_tr, time, event, risk, tau) -> float:
    ytr = Surv.from_arrays(np.asarray(event_tr, bool), np.asarray(time_tr, float))
    y = Surv.from_arrays(np.asarray(event, bool), np.asarray(time, float))
    return concordance_index_ipcw(ytr, y, np.asarray(risk, float), tau=tau)[0]


def td_auc(time_tr, event_tr, time, event, risk, t) -> float:
    ytr = Surv.from_arrays(np.asarray(event_tr, bool), np.asarray(time_tr, float))
    y = Surv.from_arrays(np.asarray(event, bool), np.asarray(time, float))
    return float(cumulative_dynamic_auc(ytr, y, np.asarray(risk, float), [t])[0][0])


def brier_at(time_tr, event_tr, time, event, surv_prob_t, t) -> float:
    ytr = Surv.from_arrays(np.asarray(event_tr, bool), np.asarray(time_tr, float))
    y = Surv.from_arrays(np.asarray(event, bool), np.asarray(time, float))
    return float(brier_score(ytr, y, np.asarray(surv_prob_t, float)[:, None], [t])[1][0])


def calibration_table(time, event, pred_risk_t, t, n_groups=5) -> pd.DataFrame:
    """Predicted vs Kaplan-Meier observed risk at t, by quantile group of predicted risk."""
    d = pd.DataFrame({"time": time, "event": event, "p": pred_risk_t})
    d["g"] = pd.qcut(d["p"].rank(method="first"), n_groups, labels=False)
    rows = []
    for g, sub in d.groupby("g"):
        s, se = km_surv(sub["time"], sub["event"], t, return_se=True)
        obs = 1 - s
        rows.append(dict(group=int(g) + 1, n=len(sub), predicted=sub["p"].mean(),
                         observed=obs, lo=max(obs - 1.96 * se, 0), hi=min(obs + 1.96 * se, 1)))
    return pd.DataFrame(rows)


def calibration_in_the_large(time, event, pred_risk_t, t) -> tuple[float, float]:
    """Observed (KM) vs mean predicted risk at t."""
    return 1 - km_surv(time, event, t), float(np.mean(pred_risk_t))


def operating_point(time, event, flag, t) -> dict:
    """Sensitivity, specificity, PPV, NPV for "flag = high risk" against death by t,
    all estimated with Kaplan-Meier within the flagged / unflagged groups."""
    time, event, flag = np.asarray(time), np.asarray(event), np.asarray(flag, bool)
    p_flag = flag.mean()
    risk_all = 1 - km_surv(time, event, t)
    risk_hi = 1 - km_surv(time[flag], event[flag], t) if flag.any() else np.nan
    risk_lo = 1 - km_surv(time[~flag], event[~flag], t) if (~flag).any() else np.nan
    sens = risk_hi * p_flag / risk_all if flag.any() else 0.0
    spec = (1 - risk_lo) * (1 - p_flag) / (1 - risk_all) if (~flag).any() else 0.0
    return dict(flagged=p_flag, sensitivity=sens, specificity=spec,
                ppv=risk_hi, npv=1 - risk_lo if (~flag).any() else np.nan)


def net_benefit(time, event, pred_risk_t, t, thresholds) -> pd.DataFrame:
    """Decision curve for "treat (refer for early transplant) if predicted risk >= pt".
    Survival version of Vickers' net benefit, with KM estimates of risk among the treated."""
    time, event, p = np.asarray(time), np.asarray(event), np.asarray(pred_risk_t)
    risk_all = 1 - km_surv(time, event, t)
    rows = []
    for pt in thresholds:
        w = pt / (1 - pt)
        treat = p >= pt
        if treat.any():
            r = 1 - km_surv(time[treat], event[treat], t)
            nb = r * treat.mean() - (1 - r) * treat.mean() * w
        else:
            nb = 0.0
        rows.append(dict(threshold=pt, net_benefit=nb, treat_all=risk_all - (1 - risk_all) * w,
                         treat_none=0.0))
    return pd.DataFrame(rows)


def bootstrap(fn, n: int, n_boot: int = 500, seed: int = 0):
    """Percentile CI of fn(idx) over bootstrap resamples of row indices."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(idx))
        except Exception:  # degenerate resample (e.g. no events)
            continue
    vals = np.array(vals, float)
    return np.nanpercentile(vals, [2.5, 97.5], axis=0)
