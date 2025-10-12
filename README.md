# 🫀 Heart Disease by Origin — "One Size Fits All" Is False

The classic Cleveland-only approach overstates performance. When we add Hungary, Switzerland, and Long Beach VA, we see structured missingness and distribution shifts. One recipe cannot cover every origin.

## What's in this repo

-   `analysis.ipynb` — notebook that builds the combined table per dataset: adds `origin`, adds `target`, drops duplicates, checks missingness by origin, runs imputation (categoricals → mode, numerics → KNN with scaling, Switzerland `chol` left as structural `NaN`), and exports `combined_clean.csv`.
-   `app.py` — Streamlit dashboard that loads `combined_clean.csv` and tells the story with filters and interactive charts.
-   `data/` — raw processed UCI files (`cleveland.data`, `hungary.data`, `switzerland.data`, `long_beach_va.data`) where `?` marks missing values.
-   `combined_clean.csv` — final clean table the app reads (created by the notebook).
-   `features.txt`, `heart-disease.names`— extra notes and reference material.

## Install

Use Python 3.9 or newer and install the needed packages:

```         
pip install pandas numpy seaborn matplotlib scikit-learn streamlit altair plotly
```

## Run the notebook

1.  Open `analysis.ipynb` in Jupyter (VSCode Highly Recommended) and run all cells.

2.  The notebook:

    -   loads each dataset, adds the `origin`, and keeps both `num` and `target = (num > 0)`;
    -   checks missing values overall and by origin to highlight gaps;
    -   runs a simple imputer and a KNN imputer, keeping the KNN result as `df_final`;
    -   clips coded fields to valid ranges and adds human-friendly labels (`cp_label`, `num_label`, etc.);
    -   prints tables and charts that show how the datasets drift apart.

3.  Export the clean table for the app:

    ```         
    df_final.to_csv("combined_clean.csv", index=False)
    ```

    Expect about 918–920 rows depending on deduping and the exact source files. Keep `combined_clean.csv` in the project root next to `app.py`.

## Use the app

1.  Make sure `combined_clean.csv` sits next to `app.py`.

2.  Launch the dashboard:

    ```         
    streamlit run app.py
    ```

3.  Explore the tabs:

    -   **Sidebar filters:** choose origins and an age range.
    -   **Metrics:** row counts plus disease prevalence by origin.
    -   **Distributions:** smooth area density curves (KDE) for a chosen numeric variable (e.g., `trestbps`, `chol`, `thalach`, `oldpeak`) by origin, with per-origin medians shown above the chart.
    -   **Relationships:** Max heart rate (`thalach`) vs age with per-origin LOESS trend lines, highlighting that the relationship’s slope/curvature differs across datasets.
    -   **Categoricals:** stacked, normalized bar charts by origin for a selected labeled feature (e.g., `cp_label`, `restecg_label`, `slope_label`, `thal_label`, `num_label`), showing composition differences across selections.
    -   **Data & Download:** preview the filtered table and export a CSV.
    -   **Prevalence:** shows the gap across datasets. The final proof that "one size fits all" fails.

## Methods

-   **Duplicates:** drop exact duplicates per dataset. Only two rows were dropped.
-   **Missingness:** convert `?` to `NaN`; treat Switzerland `chol == 0` as structural missing (leave as `NaN`).
-   **Imputation (per dataset):**
    -   Categoricals (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `thal`) → most frequent value.
    -   Numerics (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`) → KNN (`k=5`) with `StandardScaler` (scale → KNN → inverse scale).
    -   Switzerland: exclude `chol` from the KNN distance so it stays `NaN`.
-   **Target:** `target = 1` when `num > 0`, else `0`.
-   **Encoding:** categorical codes stored as pandas `Int64`; `origin` stored as a categorical field.

## Rubric mapping

+-------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Requirement                         | Where it's covered                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
+:===================================:+:============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================+
| \                                   | **`analysis.ipynb`**: loads **4 UCI datasets** (Cleveland, Hungary, Long Beach VA, Switzerland), adds `origin`, **drops duplicates** with `.drop_duplicates()`, builds **`combined_clean.csv`**.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Data Collection & Preparation (25%) |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
+-------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| EDA & Visualization (25%)           | **`analysis.ipynb`**: summary tables (`describe`, groupby-describe), seaborn **KDE** comparisons, boxplots; **Missingness by origin × column** heatmap. **App (`app.py`)**: **Distributions (KDE areas)**, **Relationships** (per-origin LOESS line for `thalach` vs `age`), **Categoricals** (stacked % bars), **Prevalence** by origin, **Data & Download**.                                                                                                                                                                                                                                                                                                                                                                              |
+-------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Data Processing (15%)               | **Per-dataset imputation** functions in notebook: **KNN** on **continuous numeric** features (`age, trestbps, chol, thalach, oldpeak, ca`), **mode** for **categorical/ordinal/nominal** (`sex, cp, fbs, restecg, exang, slope, thal`). **`target` and `num` are excluded from KNN distance** to avoid leakage.                                                                                                                                                                                                                                                                                                                                                                                                                             |
+-------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| \                                   | **`app.py`**: interactive **origin** and **age** filters; multiple tabs (Distributions, Relationships, Categoricals, Data & Download, Prevalence); metrics (rows, prevalence); tooltips; CSV download.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Streamlit App (25%)                 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
+-------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| \                                   | Organized repo with **`analysis.ipynb`**, **`app.py`**, **`data/`**, **`combined_clean.csv`** (or instructions to regenerate), and a clear **README** (install, run notebook, export CSV, run app).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| GitHub Repository (10%)             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
+-------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Above & Beyond (+25%)               | We **impute** each dataset separately using **two methods** (Simple and KNN). We make sure categorical (nominal/ordinal) columns use only their allowed values (for example: `cp` = 1–4, `restecg` = 0–2, `slope` = 1–3, `ca` = 0–3, `thal` = 3/6/7). When using KNN, we **do not include `target` or `num`** so they don’t influence how missing values are filled. We show what’s **missing** in each dataset and why that matters. The Streamlit app is **interactive** (filters, tooltips, visuals, etc.) and a user friendly interface. We compare patterns across Cleveland, Hungary, Switzerland, and Long Beach VA; the clear shifts in prevalence, distributions, and missingness show that a single approach does not generalize. |
+-------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+