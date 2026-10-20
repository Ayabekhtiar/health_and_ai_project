"""The bedside tool: raw diagnostic files in, one interpretable output per patient out.

Output per patient
  * 2-year mortality risk under current standard care, with an uncertainty range
    (model uncertainty from bootstrap refits, widened by best/worst values of any
    missing input);
  * a decision band relative to the 30 % early-transplant threshold:
      HIGHER RISK  - whole range >= 30 %: early transplant evaluation supported
      LOWER RISK   - whole range <  30 %: surveillance / delayed strategy supported
      BORDERLINE   - range crosses 30 %: no recommendation from the tool
      OUT OF INTENDED USE - blasts >= 20 %, WBC >= 13, or core labs missing
  * IPSS-R and IPSS-M side by side, the drivers of the estimate, input warnings.
"""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

from . import HORIZON_YEARS, features, scores
from .model import LABELS, LinearCox, design

BANDS = {
    "HIGHER": ("Higher risk", "Early transplant evaluation supported (if fit for transplant)", "#d03b3b", "▲"),
    "LOWER": ("Lower risk", "Surveillance / delayed strategy supported", "#0ca30c", "▼"),
    "BORDERLINE": ("Borderline", "Range crosses the threshold: no recommendation from the tool", "#fab219", "◆"),
    "OUT": ("Out of intended use", "Do not use this output for the transplant decision", "#898781", "✕"),
}


def load_model(path="models/final_model.json") -> LinearCox:
    return LinearCox.from_json(path)


def _risk_with(m: LinearCox, z: pd.DataFrame, coef, s0, lp_center) -> np.ndarray:
    b = np.array([coef[c] for c in m.cols])
    return 1 - s0 ** np.exp(z.values @ b - lp_center)


def _standardise(x, mean, scale):
    return (x - pd.Series(mean)) / pd.Series(scale)


def predict(clin: pd.DataFrame, muts: pd.DataFrame, m: LinearCox) -> pd.DataFrame:
    """One row per patient (see module docstring)."""
    return predict_full(clin, muts, m)[0]


def predict_full(clin: pd.DataFrame, muts: pd.DataFrame, m: LinearCox):
    """Returns (predictions, patient feature table, per-feature contributions)."""
    df = scores.add_scores(features.build(clin, muts))
    meta = getattr(m, "meta", {})
    thr = meta.get("threshold_2y", 0.30)

    x_raw = design(df, m.cols)
    x = x_raw.fillna(m.impute)
    z = _standardise(x, m.mean, m.scale)
    risk = _risk_with(m, z, m.coef, m.s0_horizon, m.lp_center)
    lp = z.values @ np.array([m.coef[c] for c in m.cols]) - m.lp_center

    # model uncertainty: bootstrap refits
    boots = []
    for b in meta.get("bootstrap", []):
        if "coef" in b:
            zb = _standardise(x_raw.fillna(b["impute"]), b["mean"], b["scale"])
            boots.append(_risk_with(m, zb, b["coef"], b["s0"], b["lp_center"]))
        else:  # recalibrated IPSS-M: bootstrap of the slope and baseline
            coef = {k: v * b["scale"] for k, v in m.coef.items()}
            boots.append(_risk_with(m, z, coef, b["s0"], m.lp_center * b["scale"]))
    if boots:
        B = np.vstack(boots)
        lo, hi = np.percentile(B, 5, axis=0), np.percentile(B, 95, axis=0)
    else:
        lo = hi = risk

    # missing-data range: each missing input set to its least / most harmful plausible value
    rng = meta.get("feature_range", {})
    best_x, worst_x = x.copy(), x.copy()
    for c in m.cols:
        miss = x_raw[c].isna()
        if not miss.any() or c not in rng:
            continue
        low, high = rng[c]
        harmful_high = m.coef[c] > 0
        best_x.loc[miss, c] = low if harmful_high else high
        worst_x.loc[miss, c] = high if harmful_high else low
    r_best = _risk_with(m, _standardise(best_x, m.mean, m.scale), m.coef, m.s0_horizon, m.lp_center)
    r_worst = _risk_with(m, _standardise(worst_x, m.mean, m.scale), m.coef, m.s0_horizon, m.lp_center)
    range_lo, range_hi = np.minimum(lo, r_best), np.maximum(hi, r_worst)

    band = np.where(range_lo >= thr, "HIGHER", np.where(range_hi < thr, "LOWER", "BORDERLINE"))
    band = np.where(df["IN_SCOPE"].values, band, "OUT")

    out = pd.DataFrame(index=df.index)
    out["CENTER"] = df["CENTER"]
    out["risk_score"] = lp                      # linear predictor: higher = worse (for C-index)
    out["risk_2y"] = risk
    out["risk_2y_low"] = range_lo
    out["risk_2y_high"] = range_hi
    # 80 % range of the risk at a centre other than the average one (between-centre tau)
    tau = meta.get("between_centre_tau", 0.0)
    out["risk_2y_centre_low"] = 1 - m.s0_horizon ** np.exp(lp - 1.2816 * tau)
    out["risk_2y_centre_high"] = 1 - m.s0_horizon ** np.exp(lp + 1.2816 * tau)
    out["band"] = band
    out["IPSSR_score"] = df["IPSSR_score"]
    out["IPSSR_cat"] = df["IPSSR_cat"].astype(str).replace("nan", "")
    out["IPSSM_score"] = df["IPSSM_score"]
    out["IPSSM_range"] = df["IPSSM_best"].round(2).astype(str) + " to " + df["IPSSM_worst"].round(2).astype(str)
    out["IPSSM_cat"] = df["IPSSM_cat"].astype(str)
    out["n_inputs_missing"] = x_raw.isna().sum(axis=1)
    out["warnings"] = [" | ".join(w) for w in _warnings(df)]
    return out, df, z * pd.Series(m.coef)


def _warnings(df):
    res = []
    for _, r in df.iterrows():
        w = []
        if r["OUT_BLAST_GE20"]:
            w.append("Marrow blasts >= 20 %: AML range, outside intended use")
        if r["OUT_WBC_GE13"]:
            w.append("WBC >= 13: proliferative MDS/MPN, outside intended use")
        if r["MISSING_CORE"]:
            w.append("Blasts, haemoglobin or platelets missing: outside intended use")
        if r["MISSING_KARYOTYPE"]:
            w.append("Karyotype missing or unreadable: risk range widened; obtain karyotype")
        if pd.isna(r["ANC"]):
            w.append("Neutrophil count missing")
        if r["CMML_LIKE"]:
            w.append("Monocytosis: check CMML criteria (performance lower in this group)")
        if pd.isna(r["TP53multi"]):
            w.append("TP53 allelic state unknown (no karyotype): TP53 multi-hit cannot be excluded")
        if r["NO_MUT_RECORDED"] == 1:
            w.append("No mutation recorded: confirm the panel was run (treated as wild type)")
        res.append(w)
    return res


# ---------------------------------------------------------------------- HTML card
CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--surface:#fcfcfb;--page:#f9f9f7;
--up:#d03b3b;--down:#2a78d6;--accent:#2a78d6}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.card{max-width:860px;margin:24px auto;background:var(--surface);border:1px solid var(--grid);border-radius:12px;padding:24px 28px}
h1{font-size:18px;margin:0}h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink2);margin:22px 0 8px}
.sub{color:var(--muted);font-size:12px}.row{display:flex;gap:16px;flex-wrap:wrap}
.big{font-size:40px;font-weight:700;line-height:1}.band{display:inline-flex;gap:8px;align-items:center;padding:6px 12px;
border-radius:999px;font-weight:600;color:#fff}.tile{flex:1 1 220px;border:1px solid var(--grid);border-radius:10px;padding:12px 14px}
.tile .k{color:var(--ink2);font-size:12px}.tile .v{font-size:18px;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:4px 6px;border-bottom:1px solid var(--grid);text-align:left}
th{color:var(--ink2);font-weight:600}.miss{color:var(--up);font-weight:600}
ul.warn{margin:0;padding-left:18px}ul.warn li{margin:2px 0}.note{background:var(--page);border-radius:8px;padding:10px 14px;color:var(--ink2);font-size:12.5px}
@media print{body{background:#fff}.card{border:none;margin:0}}
"""


def _bar_svg(risk, lo, hi, thr):
    W, X0, X1 = 640, 20, 620
    sx = lambda v: X0 + (X1 - X0) * v  # noqa: E731
    ticks = "".join(f'<line x1="{sx(t)}" y1="38" x2="{sx(t)}" y2="44" stroke="#c3c2b7"/>'
                    f'<text x="{sx(t)}" y="58" font-size="11" text-anchor="middle" fill="#898781">{int(t*100)}%</text>'
                    for t in np.linspace(0, 1, 6))
    return f"""<svg viewBox="0 0 {W} 64" width="100%" role="img" aria-label="2-year risk {risk:.0%}, range {lo:.0%} to {hi:.0%}, threshold {thr:.0%}">
<rect x="{X0}" y="20" width="{X1-X0}" height="12" rx="6" fill="#e1e0d9"/>
<rect x="{sx(thr)}" y="20" width="{X1-sx(thr)}" height="12" rx="0" fill="#f3d6d6"/>
<rect x="{sx(lo)}" y="17" width="{max(sx(hi)-sx(lo),2)}" height="18" rx="4" fill="#2a78d6" opacity=".35"/>
<circle cx="{sx(risk)}" cy="26" r="8" fill="#2a78d6" stroke="#fcfcfb" stroke-width="2"/>
<line x1="{sx(thr)}" y1="10" x2="{sx(thr)}" y2="40" stroke="#0b0b0b" stroke-width="1.5" stroke-dasharray="3 2"/>
<text x="{sx(thr)+4}" y="12" font-size="11" fill="#0b0b0b">early-transplant threshold {thr:.0%}</text>{ticks}</svg>"""


def _contrib_svg(contrib: pd.Series, k=8):
    c = contrib[contrib.abs() > 1e-3].sort_values(key=abs, ascending=False).head(k)
    if c.empty:
        return "<p class='sub'>All inputs close to the average patient.</p>"
    W, mid, rowh = 640, 380, 24
    m = max(c.abs().max(), 0.3)
    sc = (W - mid - 60) / m
    rows = []
    for i, (f, v) in enumerate(c.items()):
        y = 6 + i * rowh
        w = abs(v) * sc
        x = mid if v > 0 else mid - w
        col = "#d03b3b" if v > 0 else "#2a78d6"
        lab = html.escape(LABELS.get(f, f))
        tx = mid + w + 6 if v > 0 else mid - w - 6
        anchor = "start" if v > 0 else "end"
        rows.append(f'<text x="10" y="{y+13}" font-size="12" fill="#52514e">{lab}</text>'
                    f'<rect x="{x}" y="{y+2}" width="{max(w,1.5)}" height="14" rx="3" fill="{col}"/>'
                    f'<text x="{tx}" y="{y+13}" font-size="11" text-anchor="{anchor}" fill="#52514e">'
                    f'{"+" if v > 0 else "−"}{abs(v):.2f}</text>')
    h = 12 + len(c) * rowh
    return (f'<svg viewBox="0 0 {W} {h}" width="100%" role="img" aria-label="Contributions to the estimate">'
            f'<line x1="{mid}" y1="0" x2="{mid}" y2="{h}" stroke="#c3c2b7"/>{"".join(rows)}</svg>'
            '<p class="sub">Bars: change in log-hazard relative to the average patient of the training cohort '
            '(red raises, blue lowers risk). They describe the model, not the biology of this patient.</p>')


def card_html(pid: str, pred: pd.DataFrame, feats: pd.DataFrame, contrib_all: pd.DataFrame, m: LinearCox) -> str:
    r = pred.loc[pid]
    f = feats.loc[pid]
    contrib = contrib_all.loc[pid]
    meta = getattr(m, "meta", {})
    thr = meta.get("threshold_2y", 0.30)
    ref = meta.get("reference_category_2y_mortality", {})
    label, action, color, icon = BANDS[r["band"]]

    def val(v, fmt="{:.1f}"):
        return "<span class='miss'>missing</span>" if pd.isna(v) else fmt.format(v)

    inputs = [
        ("Marrow blasts (%)", val(f["BM_BLAST"], "{:.0f}")), ("Haemoglobin (g/dL)", val(f["HB"])),
        ("Platelets (x10⁹/L)", val(f["PLT"], "{:.0f}")), ("Neutrophils (x10⁹/L)", val(f["ANC"], "{:.2f}")),
        ("White cells (x10⁹/L)", val(f["WBC"])), ("Monocytes (x10⁹/L)", val(f["MONOCYTES"], "{:.2f}")),
        ("Karyotype", html.escape(str(f["CYTOGENETICS"])) if pd.notna(f["CYTOGENETICS"]) else "<span class='miss'>missing</span>"),
        ("IPSS-R karyotype group", f["CYTO_CAT"] if pd.notna(f["CYTO_CAT"]) else "<span class='miss'>not available</span>"),
        ("Driver mutations recorded", f"{int(f['N_MUT'])}"),
        ("TP53", "multi-hit" if f["TP53multi"] == 1 else ("mutated, single hit" if f["TP53_n"] > 0 and f["TP53multi"] == 0
                  else ("mutated, allelic state unknown" if f["TP53_n"] > 0 else "not mutated"))),
    ]
    genes = [g for g in str(f.get("GENES_MUTATED", "")).split(", ") if g and g != "TP53"]
    inputs.append(("Other mutated genes", html.escape(", ".join(genes)) or "none"))
    rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in inputs)
    warns = [w for w in r["warnings"].split(" | ") if w]
    warn_html = ("<ul class='warn'>" + "".join(f"<li>{html.escape(w)}</li>" for w in warns) + "</ul>") if warns \
        else "<p class='sub'>No input warnings.</p>"
    ipssr_ref = ref.get("IPSSR_cat", {}).get(r["IPSSR_cat"])
    ipssm_ref = ref.get("IPSSM_cat", {}).get(r["IPSSM_cat"])

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MDS risk card {html.escape(pid)}</title>
<style>{CSS}</style></head><body><div class="card">
<h1>Transplant-or-Wait · prognostic summary for the MDT</h1>
<div class="sub">Patient {html.escape(pid)} · centre {html.escape(str(r['CENTER']))} · model {html.escape(m.name)} v{meta.get('version','?')} · horizon {HORIZON_YEARS:.0f} years · decision support only</div>
<h2>Estimated risk of death within {HORIZON_YEARS:.0f} years, under standard care</h2>
<div class="row" style="align-items:center;gap:24px">
<div><div class="big">{r['risk_2y']:.0%}</div><div class="sub">range {r['risk_2y_low']:.0%} – {r['risk_2y_high']:.0%}</div></div>
<div><span class="band" style="background:{color}">{icon} {label}</span><div class="sub" style="margin-top:6px">{action}</div></div>
</div>
{_bar_svg(r['risk_2y'], r['risk_2y_low'], r['risk_2y_high'], thr)}
<p class="sub">Absolute risk varies between hospitals: for a patient with this profile, 80 % of centres in the
training data would see a 2-year mortality between {r['risk_2y_centre_low']:.0%} and {r['risk_2y_centre_high']:.0%}.
The ranking of patients transfers better than the absolute number.</p>
<h2>Reference scores</h2>
<div class="row">
<div class="tile"><div class="k">IPSS-R (reference)</div><div class="v">{r['IPSSR_cat'] or 'not computable'}{'' if pd.isna(r['IPSSR_score']) else f" · {r['IPSSR_score']:.1f}"}</div>
<div class="sub">{'' if ipssr_ref is None else f"2-year mortality of this group in the training cohort: {ipssr_ref:.0%}"}</div></div>
<div class="tile"><div class="k">IPSS-M</div><div class="v">{r['IPSSM_cat']} · {r['IPSSM_score']:.2f}</div>
<div class="sub">range {r['IPSSM_range']}{'' if ipssm_ref is None else f" · group 2-year mortality {ipssm_ref:.0%}"}</div></div>
</div>
<h2>What drives this estimate</h2>{_contrib_svg(contrib)}
<h2>Inputs used</h2><table>{rows}</table>
<h2>Warnings</h2>{warn_html}
<h2>What this tool does not do</h2>
<div class="note">It does not estimate the benefit of transplantation, and it does not assess fitness for transplant:
age, performance status, comorbidity (HCT-CI) and donor availability are not in the model and remain the MDT's
judgement. The estimate is a prognosis under the care received in a retrospective 23-centre cohort in which
treatments, including transplantation, were not recorded. Threshold {thr:.0%} reflects the IPSS-M Moderate-Low/Moderate-High
boundary at which early transplant was found to prolong life expectancy (Tentori et al., JCO 2024).</div>
</div></body></html>"""


def write_cards(pred, feats, contrib, m: LinearCox, ids, outdir: str):
    import os
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for pid in ids:
        p = os.path.join(outdir, f"card_{pid}.html")
        with open(p, "w") as fh:
            fh.write(card_html(pid, pred, feats, contrib, m))
        paths.append(p)
    return paths
