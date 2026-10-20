# Transplant or Wait

Prognostic decision support for allogeneic transplant timing in myelodysplastic syndromes.
Applied Artificial Intelligence for Health 2026–2027 · **Aya Bekhtiar, Apolline Hadjal**

At the end of the diagnostic work-up, the tool estimates a patient's risk of death within
2 years under standard care, and places it against the 30 % threshold at which early
transplant has been shown to prolong life (IPSS-M Moderate-High, Tentori et al. 2024).
It is IPSS-M with its published weights, recalibrated on 2,662 patients from 23 centres,
with scope checks, uncertainty ranges and a one-page patient card. It does not estimate the
benefit of transplant: the file records no treatment, age or fitness.

## Deliverables

| Deliverable | Where |
| --- | --- |
| Notebooks, with every graph | `notebooks/01_data_audit.ipynb`, `02_model_comparison.ipynb`, `03_final_model_and_tool.ipynb` (already run: outputs visible without running) |
| Written report | `report/Transplant_or_Wait_report.pdf` (7 pages; LaTeX source `report/Transplant_or_Wait_report.tex`) |
| Working system | `scripts/predict.py` (scores patients), `outputs/cards/` (patient cards) |
| Frozen model | `models/final_model.json` |

## Read the notebooks

Start here. Notebooks 1 and 2 are where the data audit and the model comparison are done: they
produce the report's figures and result tables. Notebook 3 explains the frozen model and the tool.
All three use the code in `mds_tool/` and are saved with their outputs and graphs.

| Notebook | What it shows | Run time |
| --- | --- | --- |
| `01_data_audit.ipynb` | The file, karyotype parsing, cohort flow, censoring, missing inputs, centres, IPSS-R and IPSS-M groups, overlap with the IPSS-M cohort | under 1 min |
| `02_model_comparison.ipynb` | Leave-one-centre-out comparison of IPSS-R, IPSS-M and three refitted models; selection rule; calibration; decision curve; reclassification; failure modes | about 3 min (4 min with a fresh refit) |
| `03_final_model_and_tool.ipynb` | The two locally fitted numbers, the risk formula, features and weights, one prediction by hand, bands and uncertainty, the patient card, scoring new patients | under 1 min |

Open them in VS Code or with `.venv/bin/jupyter notebook notebooks/`.

## Run it

```bash
./run_all.sh            # about 4 min
./run_all.sh --fresh    # also refits the leave-one-centre-out models, about 5 min
```

Rebuilds everything from `df_clinical_train.tsv` and `mut_clinical_train.tsv` (creates `.venv` from
`requirements.txt` on first run): tests, notebook 1, notebook 2, freezing the model
(`scripts/03_train_final.py`), example cards, notebook 3. Notebook 2 fits its 23 held-out centres in
parallel and caches their predictions in `outputs/validation/` with a fingerprint of the data and of
`mds_tool/`; the cache is reused only while both are unchanged, and a fresh refit gives bit-identical
results.

To score new patients, for example the held-out test set:

```bash
.venv/bin/python scripts/predict.py --clinical test_clinical.tsv --mutations test_mutations.tsv --out predictions_test.csv --cards outputs/cards_test
```

`risk_score` ranks patients (higher = worse) for a C-index; `risk_2y`, its range and `band`
(HIGHER, LOWER, BORDERLINE, OUT) are the decision outputs.

## The model: features and weights

2-year risk ≈ 1 − 0.684^(e^(0.57 × IPSS-M score)). The IPSS-M score is the published weighted
sum of 22 inputs, unchanged; only the slope (0.57) and the baseline were fitted on this data.
Hazard ratio = how much the risk of death is multiplied per unit, all else equal
(source: `outputs/tables/model_features.csv`, written by `scripts/03_train_final.py`).

| Feature | Unit or definition | Published HR | In this cohort |
| --- | --- | --- | --- |
| Haemoglobin | per g/dL | 0.84 | 0.87 |
| Platelets | per 100 ×10⁹/L, capped at 250 | 0.80 | 0.83 |
| Marrow blasts | per 5 %, capped at 20 % | 1.42 | 1.34 |
| Karyotype risk group | per IPSS-R step, very good (0) to very poor (4) | 1.33 | 1.27 |
| TP53 multi-hit | 2+ TP53 mutations, or 1 with VAF > 55 % or 17p loss | 3.25 | 2.65 |
| FLT3 mutation | yes / no | 2.22 | 1.94 |
| KMT2A (MLL) partial tandem duplication | yes / no | 2.22 | 1.94 |
| SF3B1 with del(5q) | SF3B1 mutated with isolated del(5q) | 1.66 | 1.52 |
| NPM1 · RUNX1 · NRAS · ETV6 · IDH2 | yes / no | 1.54 · 1.53 · 1.52 · 1.48 · 1.46 | 1.43 · 1.42 · 1.41 · 1.38 · 1.37 |
| CBL · EZH2 · U2AF1 · SRSF2 · DNMT3A · ASXL1 · KRAS | yes / no | 1.34 · 1.31 · 1.28 · 1.27 · 1.25 · 1.24 · 1.22 | 1.28 · 1.25 · 1.23 · 1.22 · 1.20 · 1.19 · 1.18 |
| SF3B1, favourable form | SF3B1 mutated, no del(5q), none of SRSF2, STAG2, BCOR, BCORL1, RUNX1, NRAS | 0.92 | 0.94 |
| Other driver genes | count of BCOR, BCORL1, CEBPA, ETNK1, GATA2, GNB1, IDH1, NF1, PHF6, PPM1D, PRPF8, PTPN11, SETBP1, STAG2, WT1, capped at 2 | 1.26 per gene | 1.21 per gene |

## Layout

```
mds_tool/      cytogenetics.py (karyotype parser), features.py, scores.py (IPSS-R, IPSS-M),
               model.py, metrics.py (censoring-aware), tool.py (bands, cards), plotting.py
notebooks/     01_data_audit -> 02_model_comparison -> (scripts/03_train_final.py) -> 03_final_model_and_tool
scripts/       03_train_final.py (freeze the model), predict.py (score new patients)
report/        Transplant_or_Wait_report.pdf and its LaTeX source (.tex)
tests/         karyotype parser, IPSS-M vs the official package
outputs/       figures/, tables/, validation/, cards/, audit.json, logs
models/        final_model.json
```

## Headline results (each centre held out in turn, 2,382 patients)

| | Harrell C | AUC at 2 y | Observed / expected |
| --- | --- | --- | --- |
| IPSS-R (reference) | 0.707 (0.691–0.723) | 0.754 | 0.99 |
| Tool (IPSS-M, recalibrated) | 0.750 (0.735–0.766) | 0.809 | 0.98 |

Refitted models (22 or 35 variables, boosted trees) did not beat IPSS-M. Main failure mode:
absolute risk differs between centres (observed / expected 0.57 to 1.41), so the tool must be
recalibrated on local patients before use.

## Notes

- `.venv/` is about 400 MB; delete it before zipping the folder. `run_all.sh` recreates it.
- To rebuild the report PDF after editing the `.tex`: `cd report && tectonic Transplant_or_Wait_report.tex`
  (any LaTeX engine works; figures are read from `outputs/figures/`).
- The project brief in this folder is the instructor's material and is not pushed (see `.gitignore`).
