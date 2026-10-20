"""Fit the model chosen by the pre-specified rule on the whole intended-use cohort
and freeze it, with everything the bedside tool and the monitoring plan need.

Writes models/final_model.json. Run after notebooks/02_model_comparison.ipynb, which writes
outputs/validation/selection.json:
    .venv/bin/python scripts/03_train_final.py
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from mds_tool import HORIZON_YEARS, cohort, metrics  # noqa: E402
from mds_tool.model import EXTENDED_VARS, IPSSM_VARS, LinearCox  # noqa: E402

THRESHOLD = 0.30
N_BOOT = 200
os.makedirs("models", exist_ok=True)

sel = json.load(open("outputs/validation/selection.json"))
chosen = sel["chosen"]
print("selection rule chose:", chosen)

df = cohort.load_all()
steps, masks = cohort.flow(df)
train = df[masks["in_scope"]]
groups = train["CENTER"].values

# The deployable artefact is always a linear Cox model stored as numbers.
# If IPSS-M itself was chosen, it is the 22 IPSS-M variables with the published
# weights and only the baseline re-estimated locally.
from mds_tool.model import _breslow_s0, surv_y  # noqa: E402
from sksurv.linear_model import CoxPHSurvivalAnalysis  # noqa: E402

T, E = train.OS_YEARS.values, train.OS_STATUS.values
rng = np.random.default_rng(42)
boot = []
if chosen == "IPSS-M":
    # IPSS-M with its published weights, kept fixed. Only two numbers are estimated
    # locally: the slope linking the score to the hazard and the 2-year baseline survival.
    from mds_tool.scores import BETA, MEAN
    model = LinearCox(IPSSM_VARS, alpha=0.0, name="IPSS-M, recalibrated to 2-year risk")
    model._prep(train, fit=True)
    model.impute = {k: MEAN[k] for k in IPSSM_VARS}        # missing inputs as in the official calculator
    z = model._prep(train)
    unit = np.array([BETA[k] * model.scale[k] / np.log(2) for k in IPSSM_VARS])  # z . unit = IPSS-M score + const

    def recal(zz, t, e):
        s = zz.values @ unit
        beta = float(CoxPHSurvivalAnalysis().fit(s[:, None], surv_y(pd.DataFrame({"OS_YEARS": t, "OS_STATUS": e}))).coef_[0])
        lp = beta * s
        c = float(lp.mean())
        return beta, c, _breslow_s0(t, e, lp - c, HORIZON_YEARS)

    beta, model.lp_center, model.s0_horizon = recal(z, T, E)
    model.coef = dict(zip(IPSSM_VARS, map(float, beta * unit)))
    model.recalibration_slope = beta
    for b in range(N_BOOT):
        idx = rng.integers(0, len(train), len(train))
        bb, cb, sb = recal(z.iloc[idx], T[idx], E[idx])
        boot.append(dict(coef=dict(zip(IPSSM_VARS, map(float, bb * unit))), mean=model.mean, scale=model.scale,
                         impute=model.impute, lp_center=cb, s0=sb))
else:
    cols = IPSSM_VARS if chosen == "Cox (IPSS-M variables, refit)" else EXTENDED_VARS
    model = LinearCox(cols, l1_ratio=0.5, name=chosen).fit(train, groups=groups)
    for b in range(N_BOOT):
        idx = rng.integers(0, len(train), len(train))
        mb = LinearCox(model.cols, l1_ratio=model.l1_ratio, alpha=model.alpha).fit(train.iloc[idx])
        boot.append(dict(coef=mb.coef, mean=mb.mean, scale=mb.scale, impute=mb.impute,
                         lp_center=mb.lp_center, s0=mb.s0_horizon))

# Between-centre heterogeneity of the baseline (transportability). For each centre:
# delta_c = log(observed / expected deaths within 2 years); tau^2 by DerSimonian-Laird.
lp_all = model.linear_predictor(train)
order = np.argsort(T)
ts, es, rs = T[order], E[order], np.exp(lp_all[order])
rev = np.cumsum(rs[::-1])[::-1]
ev_t = np.unique(ts[(es == 1) & (ts <= HORIZON_YEARS)])
dH = np.array([((ts == u) & (es == 1)).sum() / rev[np.searchsorted(ts, u)] for u in ev_t])
H0 = lambda t: dH[ev_t <= t].sum()  # noqa: E731
rows = []
for c, x in train.groupby("CENTER"):
    o = int(((x.OS_YEARS <= HORIZON_YEARS) & (x.OS_STATUS == 1)).sum())
    if o < 10:
        continue
    e_ = float(sum(H0(min(t, HORIZON_YEARS)) * np.exp(l) for t, l in zip(x.OS_YEARS, lp_all[(train.CENTER == c).values])))
    rows.append(dict(center=c, observed=o, expected=e_, log_oe=np.log(o / e_), var=1 / o))
het = pd.DataFrame(rows)
w = 1 / het["var"]
mu = float((w * het.log_oe).sum() / w.sum())
q = float((w * (het.log_oe - mu) ** 2).sum())
tau2 = max(0.0, (q - (len(het) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
het.to_csv("outputs/tables/centre_heterogeneity_oe.csv", index=False)
print(f"between-centre tau = {np.sqrt(tau2):.3f} on the log-hazard scale ({len(het)} centres)")

# What the model says on its own training data: baseline for drift monitoring
p = model.risk_at_horizon(train)
obs = 1 - metrics.km_surv(train.OS_YEARS, train.OS_STATUS, HORIZON_YEARS)
monitor = dict(
    mean_predicted_2y=float(p.mean()), observed_2y_km=float(obs),
    share_flagged=float((p >= THRESHOLD).mean()),
    predicted_quantiles={q: float(np.quantile(p, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
    input_means={c: float(train[c].mean()) for c in ["HB", "PLT", "BM_BLAST", "ANC", "WBC", "CYTOVEC", "N_MUT"]},
    missing_karyotype_share=float(train.CYTO_CAT.isna().mean()),
)

# 2-year mortality of the reference categories, for display next to the model output
ref = {}
for col in ["IPSSR_cat", "IPSSM_cat"]:
    ref[col] = {str(k): float(1 - metrics.km_surv(x.OS_YEARS, x.OS_STATUS, HORIZON_YEARS))
                for k, x in train.groupby(col, observed=True)}

from mds_tool.model import design  # noqa: E402
xr = design(train, model.cols)
feature_range = {c: [float(xr[c].quantile(0.01)), float(xr[c].quantile(0.99))] for c in model.cols}
for c, (lo_v, hi_v) in {"CYTOVEC": (0, 4), "TP53multi": (0, 1), "SF3B1_5q": (0, 1), "SF3B1_alpha": (0, 1)}.items():
    if c in feature_range:
        feature_range[c] = [lo_v, hi_v]

model.to_json("models/final_model.json", extra=dict(
    feature_range=feature_range,
    between_centre_tau=float(np.sqrt(tau2)),
    recalibration_slope=getattr(model, "recalibration_slope", None),
    version="1.0.0",
    trained_on=dt.date.today().isoformat(),
    group="Aya Bekhtiar, Apolline Hadjal",
    threshold_2y=THRESHOLD,
    selection=sel,
    training=dict(n=int(len(train)), deaths=int(train.OS_STATUS.sum()),
                  centres=int(train.CENTER.nunique()), cohort_flow=steps),
    bootstrap=boot,
    monitoring_baseline=monitor,
    reference_category_2y_mortality=ref,
))

coef = pd.Series(model.coef).sort_values(key=abs, ascending=False)
pd.DataFrame({"weight_per_SD": coef, "hazard_ratio_per_SD": np.exp(coef),
              "SD": pd.Series(model.scale)[coef.index], "mean": pd.Series(model.mean)[coef.index]}
             ).to_csv("outputs/tables/final_model_weights.csv")
if chosen == "IPSS-M":
    from mds_tool.scores import feature_table  # noqa: E402
    feature_table(model.recalibration_slope).to_csv("outputs/tables/model_features.csv", index=False)
print(f"saved models/final_model.json  (alpha={model.alpha}, s0(2y)={model.s0_horizon:.3f})")
print(coef.round(3).to_string())
print(json.dumps(monitor, indent=1))
