# 🫀 Heart Disease Multi-Institutional Analysis

**Interactive Streamlit Dashboard Exploring Origin-Specific Patterns in Heart Disease Data**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.46+-FF4B4B?logo=streamlit)](https://streamlit.io/) [![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://www.python.org/)

> *"One Size Fits All" Isn't The Most Effective: Why heart disease prediction requires understanding institutional differences*

------------------------------------------------------------------------

## Why This Dataset?

I chose the **UCI Heart Disease Database** because I am come from a medical background and have a strong interest in healthcare analytics. Heart disease is a leading cause of death worldwide, and accurate prediction models can save lives through early intervention. This dataset uniquely combines data from **four different medical institutions** (Cleveland Clinic, Hungarian Institute of Cardiology, University Hospital Zurich, and V.A. Medical Center Long Beach). This multi-origin structure makes it perfect for exploring a critical but often overlooked question in medical machine learning:

**Can models trained on one institution's data generalize to other populations?**

Most ML researchers focus only on the Cleveland dataset, which represents just one patient population. By analyzing all four origins together, we can investigate: - How patient characteristics vary across different origins - Whether feature relationships are universal or population-specific - Why "one-size-fits-all" models may fail when applied to new settings

This dataset reveals real-world challenges in medical AI: distribution shifts, missing data patterns, and population variability that must be addressed for fair and accurate prediction.

------------------------------------------------------------------------

## What I Learned from IDA/EDA

### Key Insights from Initial Data Analysis

1.  **Significant Prevalence Differences (45.9% Cleveland → 93.5% Switzerland)**
    -   Heart disease rates vary dramatically by origin
    -   Cleveland likely serves higher-risk or specialty populations
    -   Models trained on one origin will systematically over/under-estimate risk elsewhere
    -   This is the **base rate fallacy** in action
2.  **Origin-Specific Missing Data Patterns**
    -   Hungary & Long Beach VA: `ca` (coronary vessels) = 0 for ALL patients
    -   Switzerland: `chol` (cholesterol) = 0 for ALL patients
    -   These aren't true zeros. They most likely represent different data collection procedures/equipment
    -   Imputation must respect these institutional differences to avoid bias
3.  **Feature Distributions Vary Significantly**
    -   Blood pressure, cholesterol, and heart rate distributions differ across origins
    -   Example: `trestbps`-`age` correlation is 2x stronger in Switzerland (0.37) vs Long Beach (0.18)
    -   Different patient populations, fitness levels, or testing protocols
4.  **Relationship Patterns Are Not Universal**
    -   Age-heart rate decline varies by origin (different slopes in LOESS curves)
    -   `oldpeak` (ST depression) predicts heart disease 2x better in Cleveland (0.42) than Long Beach (0.21)
    -   Feature importance is population-dependent
5.  **Categorical Distributions Reveal Referral Bias**
    -   Different chest pain type distributions suggest varying patient pathways
    -   Some institutions may see more severe cases (specialty clinic effect)
    -   Impacts how we should interpret and model the data

### Evidence Against "One-Size-Fits-All" Models

Every analysis page reveals origin-specific patterns: - **Distributions**: Different medians and shapes across origins - **Relationships**: Varying correlation structures in heatmaps - **Categories**: Different symptom/test result distributions - **Prevalence**: Differences in heart disease rates

**Conclusion**: Models must account for these institutional differences to avoid biased predictions and ensure fairness in healthcare AI.

------------------------------------------------------------------------

## Preprocessing Steps Completed

### 1. Data Loading & Combination

-   Loaded four datasets from UCI repository (Cleveland, Hungary, Switzerland, Long Beach VA)
-   Added `origin` identifier to track institutional source
-   Removed duplicate rows (minimal: \~2 rows total)
-   Combined into single multi-origin dataset

### 2. Target Variable Creation

-   Created binary `target` variable from original `num` (diagnosis)
-   `target = 1` when `num > 0` (any heart disease present)
-   `target = 0` when `num = 0` (no disease)
-   Simplifies multi-class problem to binary classification

### 3. Missing Data Handling

-   Converted `?` symbols to `NaN` for proper missing value representation
-   Identified origin-specific missing patterns (`ca`, `chol`)
-   Preserved zeros for ca and chol columns to maintain data authenticity and reveal institutional differences
-   Missing data rates vary by origin (evidence of different protocols)
-   Hidden missing values (e.g., all zeros) identified and treated accordingly

### 4. Imputation Strategy

-   **Method**: KNN Imputation (k=5) applied **independently per origin**
-   **Why KNN?**: Preserves local structure and feature relationships better than mean/mode
-   **Why per-origin?**: Prevents data leakage and respects institutional boundaries
-   **Categorical features**: Mode imputation for truly missing values
-   **Result**: Complete dataset while maintaining origin-specific characteristics

### 5. Feature Engineering & Encoding

-   Created human-readable labels alongside numeric codes:
    -   `cp_label`: Chest pain type (typical angina, atypical angina, etc.)
    -   `restecg_label`: Resting ECG results
    -   `slope_label`: ST segment slope
    -   `thal_label`: Thalassemia status
    -   `num_label`: Original diagnosis severity
-   Maintains both formats: numeric for modeling, labels for interpretation
-   Ensured categorical codes stay within valid ranges post-imputation

### 6. Data Validation

-   Verified biological plausibility of values
-   Confirmed no remaining missing values after imputation
-   Checked distributions match expected medical ranges
-   Validated origin-specific patterns are preserved

------------------------------------------------------------------------

## What I've Built with Streamlit

### Modern Multi-Page App

Built an interactive dashboard using **Streamlit 1.50** with horizontal navigation. The app features a clean, professional design with consistent styling and Material icons throughout.

### Navigation Structure

**About** - Project overview with summary metrics. Problem statement with compelling statistics. Expandable sections for methodology and navigation guide

**Initial Data Analysis** - **Data Overview**: Individual dataset statistics, feature descriptions, and data quality notes - **Missing Data**: Comprehensive analysis with metrics, heatmaps, and missingness patterns by origin

**Data Cleaning** - **Imputation**: Side-by-side comparison of Simple vs KNN imputation methods with distribution preservation analysis - **Encoding**: Labels mapped to numeric codes for interpretability

**Exploratory Data Analysis** - **Distributions**: Interactive KDE plots comparing feature distributions across origins - **Relationships**: Age-heart rate scatter with LOESS trends and correlation heatmaps by origin - **Categories**: Stacked bar charts showing categorical feature distributions - **Prevalence**: Heart disease rates by origin revealing base rate differences

**Modeling & Thesis validation** - Global vs Stratified models (Logistic Regression and Decision Trees) with performance metrics, ROC curves, and confusion matrices broken down by origin

**Data Export** - Preview of filtered/cleaned data and CSV download functionality with applied filters

### Interactive Features

1.  **Global Filters (Sidebar)**
    -   Multi-select for origins (Cleveland, Hungary, Switzerland, Long Beach VA)
    -   Age range slider
    -   Filters apply across all analysis pages
2.  **Dynamic Visualizations**
    -   Altair charts with tooltips and interactivity
    -   Origin-specific color coding for consistency
    -   Responsive layouts that adapt to container width
3.  **Contextual Information Boxes**
    -   Every page ends with an info box explaining "Why this matters"
    -   Connects observations to real-world implications
    -   Guides users through the analytical narrative
4.  **Origin-Specific Comparisons**
    -   Tabbed views for individual dataset analysis
    -   Side-by-side correlation heatmaps revealing relationship differences
    -   Separate statistical summaries for each institution

### Technical Highlights

-   **Session state management**: Efficient data sharing across pages
-   **Proper categorical statistics**: Count/unique/top/freq instead of mean/std for encoded categories
-   **Material Design icons**: Modern look and feel
-   **Responsive metrics**: Dynamic calculations based on filtered data
-   **Error handling**: Graceful fallbacks when data is insufficient

------------------------------------------------------------------------

## Repository Structure

```         
├── app.py                        # Main Streamlit application (navigation router)
├── analysis.py                   # Data processing pipeline and visualization functions
├── pages/                        # Separate page files for each navigation item
│   ├── about.py            
│   ├── data_overview.py
│   ├── missing_data.py
│   ├── imputation.py
│   ├── encoding.py
│   ├── distributions.py
│   ├── relationships.py
│   ├── categoricals.py
│   ├── prevalence.py
│   └── data_export.py
├── data/                          # Raw UCI datasets
│   ├── cleveland.data
│   ├── hungary.data
│   ├── switzerland.data
│   ├── long_beach_va.data
│   └── documentation/             # Data documentation
│       ├── heart-disease.names    # Official UCI feature documentation
│       └── features.txt           # Additional feature notes
├── cleaned_data.csv               # Final cleaned and imputed dataset
├── notebook.ipynb                 # Extra notebook for better display
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

------------------------------------------------------------------------

## Installation & Setup

### Prerequisites

-   Python 3.9 or newer
-   pip package manager

### Install Dependencies

``` bash
pip install pandas numpy seaborn matplotlib scikit-learn streamlit altair scipy
```

Or use requirements (if available):

``` bash
pip install -r requirements.txt
```

### Required Packages

-   `pandas` - Data manipulation
-   `numpy` - Numerical operations
-   `scikit-learn` - Imputation algorithms
-   `streamlit` - Web app framework (requires 1.46+ for horizontal navigation)
-   `altair` - Interactive visualizations
-   `seaborn` - Statistical plots (for analysis.py)
-   `matplotlib` - Plotting (for analysis.py)
-   `scipy` - Statistical functions (KDE, LOESS)

------------------------------------------------------------------------

## Running the Application

### Launch the Streamlit Dashboard

``` bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

------------------------------------------------------------------------

## Data Source

**UCI Machine Learning Repository - Heart Disease Database** - Cleveland Clinic Foundation - Hungarian Institute of Cardiology, Budapest\
- University Hospital, Zurich, Switzerland - V.A. Medical Center, Long Beach, California

[UCI Repository Link](https://archive.ics.uci.edu/ml/datasets/heart+disease)

------------------------------------------------------------------------

## Data Dictionary

| Feature | Type | Description | Values/Range |
|---------|------|-------------|--------------|
| `age` | Numeric | Age in years | 29-77 |
| `sex` | Binary | Biological sex | 0=Female, 1=Male |
| `cp` | Categorical | Chest pain type | 1=Typical angina, 2=Atypical angina, 3=Non-anginal pain, 4=Asymptomatic |
| `trestbps` | Numeric | Resting blood pressure (mm Hg) | 94-200 |
| `chol` | Numeric | Serum cholesterol (mg/dl) | 126-564 (0=not collected in Switzerland) |
| `fbs` | Binary | Fasting blood sugar > 120 mg/dl | 0=False, 1=True |
| `restecg` | Categorical | Resting ECG results | 0=Normal, 1=ST-T abnormality, 2=LVH |
| `thalach` | Numeric | Maximum heart rate achieved | 71-202 |
| `exang` | Binary | Exercise-induced angina | 0=No, 1=Yes |
| `oldpeak` | Numeric | ST depression induced by exercise | 0.0-6.2 |
| `slope` | Categorical | Slope of peak exercise ST segment | 1=Upsloping, 2=Flat, 3=Downsloping |
| `ca` | Numeric | Number of major vessels colored by fluoroscopy | 0-3 |
| `thal` | Categorical | Thalassemia | 3=Normal, 6=Fixed defect, 7=Reversible defect |
| `num` | Categorical | Diagnosis of heart disease | 0=No disease, 1-4=Disease severity |
| `target` | Binary | Heart disease present (derived) | 0=No, 1=Yes |
| `origin` | Categorical | Data source institution | Cleveland, Hungary, Long Beach VA, Switzerland |

### Feature Reliability by Origin

| Feature | Cleveland | Hungary | Long Beach VA | Switzerland |
|---------|-----------|---------|---------------|-------------|
| `ca` | ✅ Complete | ❌ 99% missing | ❌ 99% missing | ❌ 96% missing |
| `thal` | ✅ Complete | ❌ 90% missing | ❌ 83% missing | ⚠️ 42% missing |
| `slope` | ✅ Complete | ❌ 65% missing | ❌ 51% missing | ⚠️ 14% missing |
| `fbs` | ✅ Complete | ✅ 3% missing | ✅ 4% missing | ❌ 61% missing |
| `chol` | ✅ Complete | ✅ 8% missing | ✅ 4% missing | ❌ All zeros |

**Modeling Note:** Features with >50% missing in any origin (`ca`, `thal`, `slope`, `fbs`, `chol`) are excluded from modeling to avoid imputation artifacts.

------------------------------------------------------------------------

## Modeling Approach

### Models Implemented

1. **Global Logistic Regression** - Linear baseline trained on all origins combined
2. **Global Decision Tree** - Non-linear model (max_depth=5) trained on all origins
3. **Stratified Logistic Regression** - Separate LR models trained per institution
4. **Stratified Decision Tree** - Separate DT models trained per institution

### Evaluation Metrics

- **Accuracy, Precision, Recall, F1-Score** - Standard classification metrics
- **ROC-AUC** - Area under the receiver operating characteristic curve
- **Per-Origin Performance Breakdown** - Tests generalization across institutions
- **Confusion Matrices** - Visual comparison of prediction errors

------------------------------------------------------------------------

## Author

**Wahid Hashem**\
CMSE 830 - Foundations of Data Science\
Michigan State University

------------------------------------------------------------------------

## License

This project uses publicly available data from the UCI Machine Learning Repository. Please cite the original data sources if using this work.
