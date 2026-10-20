# Transplant or Wait

**A decision-support tool for transplant timing in myelodysplastic syndromes (MDS)**

Aya Bekhtiar, Apolline Hadjal

## 1. What does this project do?

This tool estimates a patient's risk of dying within two years under standard care. It helps identify patients for whom early allogeneic stem cell transplantation may be worth considering.

The tool uses the published IPSS-M scoring system, recalibrated using **2,662 patients from 23 centres**.

It compares the estimated two-year mortality risk with a **30% threshold**. In the IPSS-M Moderate-High group, this threshold is based on evidence that early transplantation can prolong life (Tentori et al., 2024).

The tool also provides:
- An uncertainty range around the risk estimate.
- A risk category.
- Checks to identify patients outside the model's intended scope.
- A one-page patient card.

**Important limitations:** The tool estimates mortality risk under standard care; it does not estimate the individual benefit of transplantation. The available data do not include treatment, age or fitness. The tool must be recalibrated using local patient data before clinical use.

## 2. Quick start

### What you need

- Python and the packages listed in `requirements.txt`.
- The two training files:
  - `df_clinical_train.tsv`: clinical data, one row per patient.
  - `mut_clinical_train.tsv`: gene mutation data, one row per mutation.

### Run the complete project

Open a terminal in the project folder and run:

```bash
./run_all.sh
```

The first run creates a virtual environment (`.venv`) and installs the required packages. The script then runs the tests, notebooks, model training and freezing, and patient-card generation.

**Expected runtime:** about 4 minutes.

To rerun the leave-one-centre-out model comparisons from scratch, use:

```bash
./run_all.sh --fresh
```

**Expected runtime:** about 5 minutes.

The script reuses cached validation predictions when the data and `mds_tool/` code have not changed. The cache includes a fingerprint of both. A fresh refit produces bit-identical results.

## 3. Simple project workflow

The project follows these steps:

```text
Input data
    |
    v
1. Check and understand the data
    |
    v
2. Compare and validate the models
    |
    v
3. Freeze the selected model
    |
    v
4. Score patients and generate patient cards
```

| Step | What happens | Where |
|---|---|---|
| 1. Data audit | Check the files, patient cohort, missing data, centres, karyotypes and existing risk scores. | `notebooks/01_data_audit.ipynb` |
| 2. Model comparison | Compare IPSS-R, IPSS-M and alternative models across held-out centres. Evaluate performance and select the model using a predefined rule. | `notebooks/02_model_comparison.ipynb` |
| 3. Freeze the model | Refit the selected model on all available patients and save its parameters. | `scripts/03_train_final.py` |
| 4. Use the tool | Calculate patient risks, uncertainty ranges and risk categories; generate patient cards. | `scripts/predict.py` and `notebooks/03_final_model_and_tool.ipynb` |

## 4. Read the notebooks

The notebooks contain the analyses, graphs and results. They have already been run, so their saved outputs can be viewed without executing them again.

| Notebook | What you will learn | Approximate runtime |
|---|---|---|
| `01_data_audit.ipynb` | Input files, karyotype parsing, cohort flow, censoring, missing data, centre differences, IPSS-R/IPSS-M groups and overlap with the IPSS-M cohort. | Under 1 minute |
| `02_model_comparison.ipynb` | Model comparison, model selection, calibration, decision curves, reclassification, subgroup performance and failure modes. | 3 minutes; 4 minutes with a fresh refit |
| `03_final_model_and_tool.ipynb` | The final model, its formula and weights, a worked prediction, risk categories, uncertainty ranges, patient cards and scoring new patients. | Under 1 minute |

**How to open them:**

Open the project in VS Code, or run:

```bash
.venv/bin/jupyter notebook notebooks/
```

All notebooks use the shared code in `mds_tool/`.

## 5. Score new patients

Use `scripts/predict.py` to score a new patient dataset.

Your clinical file must contain one row per patient. Your mutation file must contain one row per mutation, linked to the corresponding patient.

Example using test data:

```bash
.venv/bin/python scripts/predict.py \
  --clinical test_clinical.tsv \
  --mutations test_mutations.tsv \
  --out predictions_test.csv \
  --cards outputs/cards_test
```

Replace the example input paths with your own files. The command writes:
- `predictions_test.csv`: patient-level predictions.
- `outputs/cards_test/`: generated patient cards.

### Understanding the results

| Output | Meaning |
|---|---|
| `risk_score` | Relative risk ranking. Higher values indicate worse prognosis; used for the C-index. |
| `risk_2y` | Estimated probability of death within two years under standard care. |
| Risk range | Uncertainty range around the estimated risk. |
| `band` | Risk category: `HIGHER`, `LOWER`, `BORDERLINE` or `OUT`. |

The risk categories help interpret the estimate against the decision threshold. `OUT` indicates that the patient is outside the tool's intended scope.

**Do not interpret these categories as a personalised estimate of transplant benefit.** The tool supports risk assessment, not a treatment decision on its own.

## 6. How the model works

The tool keeps the published IPSS-M weighted score unchanged and recalibrates its relationship with mortality using the project cohort.

The two fitted values are:
- **Slope:** 0.57
- **Baseline survival at two years:** 0.684

The estimated two-year mortality risk is:

\[
\text{Risk}_{2y}
=1-0.684^{\exp(0.57\times\text{IPSS-M score})}
\]

The IPSS-M score is calculated from its published weighted sum of 22 inputs.

A hazard ratio (HR) describes how the relative risk of death changes with a feature, all else being equal. An HR above 1 indicates higher risk; below 1 indicates lower risk.

The table below compares the published hazard ratios with those observed in this cohort. The cohort values are reported in `outputs/tables/model_features.csv`, generated by `scripts/03_train_final.py`.

### Model features and hazard ratios

| Feature | Unit or definition | Published HR | Cohort HR |
|---|---|---:|---:|
| Haemoglobin | Per g/dL | 0.84 | 0.87 |
| Platelets | Per 100 × 10⁹/L, capped at 250 | 0.80 | 0.83 |
| Bone marrow blasts | Per 5%, capped at 20% | 1.42 | 1.34 |
| Karyotype risk group | Per IPSS-R step, from very good (0) to very poor (4) | 1.33 | 1.27 |
| TP53 multi-hit | 2+ TP53 mutations, or 1 mutation with VAF above 55% or 17p loss | 3.25 | 2.65 |
| FLT3 mutation | Yes/no | 2.22 | 1.94 |
| KMT2A (MLL) partial tandem duplication | Yes/no | 2.22 | 1.94 |
| SF3B1 with del(5q) | SF3B1 mutated with isolated del(5q) | 1.66 | 1.52 |
| NPM1 | Mutation present/absent | 1.54 | 1.43 |
| RUNX1 | Mutation present/absent | 1.53 | 1.42 |
| NRAS | Mutation present/absent | 1.52 | 1.41 |
| ETV6 | Mutation present/absent | 1.48 | 1.38 |
| IDH2 | Mutation present/absent | 1.46 | 1.37 |
| CBL | Mutation present/absent | 1.34 | 1.28 |
| EZH2 | Mutation present/absent | 1.31 | 1.25 |
| U2AF1 | Mutation present/absent | 1.28 | 1.23 |
| SRSF2 | Mutation present/absent | 1.27 | 1.22 |
| DNMT3A | Mutation present/absent | 1.25 | 1.20 |
| ASXL1 | Mutation present/absent | 1.24 | 1.19 |
| KRAS | Mutation present/absent | 1.22 | 1.18 |
| Favourable SF3B1 form | SF3B1 mutated, no del(5q), and none of SRSF2, STAG2, BCOR, BCORL1, RUNX1 or NRAS | 0.92 | 0.94 |
| Other driver genes | Count of BCOR, BCORL1, CEBPA, ETNK1, GATA2, GNB1, IDH1, NF1, PHF6, PPM1D, PRPF8, PTPN11, SETBP1, STAG2 and WT1; capped at 2 | 1.26 per gene | 1.21 per gene |

The complete feature definitions, published weights and fitted parameters are available in the notebooks and output tables.

## 7. Model validation and results

The model was evaluated using **leave-one-centre-out validation**: each centre was held out in turn, and predictions were evaluated on patients from that centre. The results below cover 2,382 patients.

| Model | Harrell C-index (95% CI) | AUC at 2 years | Observed / expected mortality |
|---|---:|---:|---:|
| IPSS-R (reference) | 0.707 (0.691–0.723) | 0.754 | 0.99 |
| Recalibrated IPSS-M tool | 0.750 (0.735–0.766) | 0.809 | 0.98 |

**What do these numbers mean?**

- **C-index:** Measures how well the model ranks patients by risk. Higher is better.
- **AUC at two years:** Measures how well the model distinguishes patients who die within two years from those who do not, accounting for the evaluation method. Higher is better.
- **Observed / expected mortality:** Compares actual deaths with predicted deaths. A value near 1 indicates good overall calibration.

Alternative models using 22 or 35 variables and boosted trees did not outperform IPSS-M.

**Main limitation:** Absolute mortality risk varied substantially between centres. Observed-to-expected ratios ranged from 0.57 to 1.41. Local recalibration is therefore needed before clinical use.

## 8. Where to find project files

### Main folders and files

| Location | Contents |
|---|---|
| `notebooks/` | Three notebooks covering the data audit, model comparison and final tool. |
| `scripts/predict.py` | Scores new patients and generates patient cards. |
| `scripts/03_train_final.py` | Fits the final model and saves its parameters and feature tables. |
| `mds_tool/` | Shared code used by the notebooks and prediction tool. |
| `models/final_model.json` | Saved final model parameters. |
| `tests/` | Automated tests for karyotype parsing and IPSS-M scoring. |
| `outputs/figures/` | All project figures. |
| `outputs/tables/` | Cohort, centre, missing-data, mortality, feature and weight tables. |
| `outputs/validation/` | Validation predictions, performance metrics, calibration, decision curves, subgroup and centre results. |
| `outputs/cards/` | Generated patient cards. |
| `outputs/audit.json` | Summary of the data audit. |
| `outputs/predictions_training_file.csv` | Predictions for every patient in the training file. |
| `outputs/train_log.txt` | Final model training log. |
| `report/` | Project report in PDF and LaTeX source. |
| `requirements.txt` | Pinned Python package versions. |
| `run_all.sh` | Runs the complete project workflow. |

### What is inside `mds_tool/`?

| File | Purpose |
|---|---|
| `cohort.py` | Loads the clinical and mutation files and builds the cohort flow. |
| `cytogenetics.py` | Parses karyotypes and assigns IPSS-R cytogenetic risk groups. |
| `features.py` | Creates one feature row per patient and checks model scope. |
| `scores.py` | Calculates IPSS-R and IPSS-M using published weights. |
| `model.py` | Defines candidate models. |
| `metrics.py` | Calculates the C-index, AUC, Brier score, calibration and net benefit, accounting for censoring. |
| `tool.py` | Calculates risk estimates, uncertainty ranges, categories and patient cards. |
| `plotting.py` | Generates figures. |

### Important output files

The model-freezing script (`scripts/03_train_final.py`) generates:

- `models/final_model.json`: final model parameters.
- `outputs/tables/model_features.csv`: model features and hazard ratios.
- `outputs/tables/final_model_weights.csv`: final model weights.
- `outputs/train_log.txt`: training log.

The validation folder, `outputs/validation/`, contains:

- Predictions for held-out centres.
- Performance metrics and paired C-index differences.
- Calibration deciles and decision curves.
- Decision rules and reclassification against IPSS-R.
- Subgroup and per-centre results.
- `selection.json`: the model selected by the predefined rule.
- `loco_cache.json`: cache fingerprint.

Other audit outputs include cohort flow, centre information, missing inputs, centre heterogeneity and two-year mortality in the reference groups.

## 9. Tests and reproducibility

The automated tests are:

- `tests/test_cytogenetics.py`: checks the karyotype parser against known cases.
- `tests/test_scores.py`: checks that the project's IPSS-M calculation matches the official implementation.

Run the full workflow with `./run_all.sh` to execute the tests and regenerate the analyses and outputs.

The environment uses the pinned package versions in `requirements.txt`. The `.venv/` directory is approximately 400 MB and can be deleted before sharing or zipping the project. Running `./run_all.sh` recreates it.

### Rebuild the PDF report

After editing the LaTeX source, run:

```bash
cd report
tectonic Transplant_or_Wait_report.tex
```

Any LaTeX engine can be used. Figures are loaded from `outputs/figures/`.

## 10. Project structure

```text
mds_tool/       Shared scoring, modelling, metrics and plotting code
notebooks/      Data audit -> model comparison -> final tool
scripts/        Model freezing and patient prediction
tests/          Automated tests
models/         Frozen model parameters
outputs/        Figures, tables, validation results, cards and logs
report/         PDF report and LaTeX source
```

The project brief provided by the instructor is not included in version control; it is excluded through `.gitignore`.
