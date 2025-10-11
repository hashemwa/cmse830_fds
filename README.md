# Midterm Project – UCI Heart Disease Sites

## Project Overview

This project shows why a single rule book does not work for the UCI Heart Disease collection. We combine four local hospital files (Cleveland, Hungary, Switzerland, Long Beach VA), clean them with site-aware rules, and surface the differences through an interactive Streamlit app. The same Python codebase delivers the data prep, the dashboard, and a quick generalization demo.

## Quick Start

```bash
python prepare.py        # optional: rebuild the cleaned CSVs and print a console summary
streamlit run app.py     # launch the Streamlit dashboard
```

All paths are relative. Keep the raw `.data` files in the project folder.

## Rubric Alignment

### Data Collection & Preparation (25%)
- Four distinct sources: `cleveland.data`, `hungary.data`, `switzerland.data`, `long_beach_va.data`, each tagged with an `origin` column.
- Duplicate diagnostics per site (exact matches, feature-only matches, conflicting labels) with exact duplicates removed.
- Missing data rules: Switzerland `chol==0` converted to `NaN`; categorical codes imputed with per-site `SimpleImputer(most_frequent)`; numeric fields imputed with per-site `StandardScaler` → `KNNImputer(k=5)` → inverse scaling (Switzerland `chol` excluded from KNN and left missing).
- Data types: categorical codes stored as nullable `Int64`, `origin` as categorical, binary target `target = (num > 0)` alongside the original `num`.
- Saved outputs: `heart_imputed_sitewise.csv` (primary) and `heart_imputed_simple.csv` (baseline).

### Exploratory Data Analysis & Visualization (25%)
- Three+ visualization types in the app, all tied to `target`:
  - Box plots for `oldpeak` and `thalach` vs disease status.
  - Grouped bar chart for chest pain code (`cp`) vs disease status.
  - Interaction scatter (`oldpeak` vs `thalach`) colored by disease status and marked by origin.
  - Missingness heatmap for `['chol', 'ca', 'thal', 'origin', 'target']` highlighting Switzerland’s structural gaps.
- Summary tables show mean/median/std for key vitals and class balance per origin.
- Captions and short notes explain what each chart reveals about the “one-size-fits-all” claim.

### Data Processing (15%)
- Two imputation strategies:
  - Site-wise KNN (primary) with StandardScaler and KNN (k=5) per site.
  - Simple per-site median baseline for comparison in the app.
- Labels are not imputed; Switzerland cholesterol remains missing in the KNN output so the structured gap stays visible.

### Streamlit App Development (25%)
- Functional dashboard with at least two interactive controls: imputation toggle, origin multiselect, age slider, and an optional “hide cholesterol” checkbox.
- Story tab, explorer tab, missing-data tab, data-prep tab, and model tab with plain-language documentation.
- Auto-preparation runs on app load; download buttons supply the cleaned CSVs. Deployment instructions are below.

### GitHub Repository (10%)
- Organized repo with `app.py`, `prepare.py`, `requirements.txt`, `README.md`, and the raw `.data` files.
- README covers the project overview, rubric alignment, setup steps, and deployment notes.

### Above & Beyond Opportunities (+20% potential)
- Complex missingness conveyed through the heatmap and the “hide cholesterol” option (Switzerland remains missing under site-wise KNN).
- Stability snapshot compares mean `oldpeak`, `thalach`, and `ca` across both imputers.
- Generalization demo trains a logistic regression on Cleveland only and reports accuracy/AUC on the other sites, showing the generalization gap.
- App layout uses tabs, metrics, captions, and download buttons for a polished presentation.

## File Guide

| File | Purpose |
| --- | --- |
| `app.py` | Contains the full pipeline (loading, deduping, imputing) and the Streamlit UI. Run with `streamlit run app.py`. |
| `prepare.py` | CLI helper that calls the shared pipeline, saves the CSVs, and prints a quick summary. |
| `heart_imputed_sitewise.csv` | Site-wise KNN output (created on-demand by the pipeline). |
| `heart_imputed_simple.csv` | Simple per-site median baseline (also created on-demand). |
| `requirements.txt` | Minimal dependency list: pandas, numpy, scikit-learn, streamlit, plotly. |
| `heart-disease.names` | Original UCI data dictionary for reference. |

## Deployment (Streamlit Community Cloud)

1. Push the repo to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) and select **New app**.
3. Choose your repo, main branch, and `app.py` entry point.
4. (Optional) run `python prepare.py` locally first so the CSVs are committed; otherwise, the app will generate them on first load.
5. Click **Deploy**. The URL can be shared with classmates or graders.

## Notes for Presenters and Graders

- The dashboard auto-runs the cleaning steps at startup. Use the download buttons if you want the processed CSVs.
- Switch between “Site-wise KNN” and “Simple median” to compare imputers and highlight Switzerland’s missing cholesterol.
- The Model tab reveals how a Cleveland-only model drops in accuracy and AUC on other sites, reinforcing the main story.

Enjoy presenting how local context changes the heart disease diagnosis story!*** End Patch
