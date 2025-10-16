# 🫀 Heart Disease Multi-Institutional Analysis

**Interactive Streamlit Dashboard Exploring Origin-Specific Patterns in Heart Disease Data**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.46+-FF4B4B?logo=streamlit)](https://streamlit.io/) [![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://www.python.org/)

> *"One Size Fits All" Is False: Why heart disease prediction requires understanding institutional differences*

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
    -   Switzerland: `chol` (cholesterol) = 0 for ALL patients\
    -   These aren't true zeros—they represent different data collection procedures/equipment
    -   Naive imputation would mask these institutional differences
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

Every analysis page reveals origin-specific patterns: - **Distributions**: Different medians and shapes across origins - **Relationships**: Varying correlation structures in heatmaps - **Categories**: Different symptom/test result distributions - **Prevalence**: Dramatic base rate differences

**Conclusion**: Models must account for these institutional differences through techniques like origin-stratified modeling, domain adaptation, or calibration.

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
-   **Preserved zeros as-is** to maintain data authenticity and reveal institutional differences
-   Missing data rates vary by origin (evidence of different protocols)

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

### Modern Multi-Page Architecture

Built an interactive dashboard using **Streamlit 1.46+** with horizontal navigation (`st.navigation(position="top")`). The app features a clean, professional design with consistent styling and Material icons throughout.

### Navigation Structure

**Home** - Project overview with summary metrics - Problem statement with compelling statistics - Expandable sections for methodology and navigation guide

**Initial Data Analysis** - **Data Overview**: Individual dataset statistics, feature descriptions, and data quality notes - **Missing Data**: Comprehensive analysis with metrics, heatmaps, and feature-level breakdowns

**Data Cleaning** - **Imputation**: Side-by-side comparison of Simple vs KNN imputation methods with distribution preservation analysis - **Encoding**: Human-readable labels mapped to numeric codes for interpretability

**Exploratory Data Analysis** - **Distributions**: Interactive KDE plots comparing feature distributions across origins - **Relationships**: Age-heart rate scatter with LOESS trends + correlation heatmaps by origin - **Categories**: Stacked bar charts showing categorical feature distributions - **Prevalence**: Heart disease rates by origin revealing base rate differences

**Data Export** - Preview of filtered/cleaned data - CSV download functionality with applied filters

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
-   **Material Design icons**: Professional visual elements throughout
-   **Responsive metrics**: Dynamic calculations based on filtered data
-   **Error handling**: Graceful fallbacks when data is insufficient

### Storytelling Through Design

Each page follows a consistent pattern: 1. **Title + Subtitle**: Clear page purpose 2. **Context**: Brief explanation of what users will see 3. **Visualization/Analysis**: Interactive charts and tables 4. **Info Box**: Connects findings to the overarching thesis about origin-specific patterns

The app successfully transforms complex multi-origin data into an accessible narrative about why institutional differences matter for medical machine learning.

------------------------------------------------------------------------

## Repository Structure

```         
├── app.py                      # Main Streamlit application (navigation router)
├── analysis.py                 # Data processing pipeline and visualization functions
├── pages/                      # Separate page files for each navigation item
│   ├── about.py               # Home/landing page
│   ├── data_overview.py       # Raw data exploration
│   ├── missing_data.py        # Missing value analysis
│   ├── imputation.py          # Imputation comparison
│   ├── encoding.py            # Categorical encoding mappings
│   ├── distributions.py       # Feature distribution analysis
│   ├── relationships.py       # Feature relationships & correlations
│   ├── categoricals.py        # Categorical variable analysis
│   ├── prevalence.py          # Disease prevalence by origin
│   └── data_export.py         # Download functionality
├── data/                       # Raw UCI datasets
│   ├── cleveland.data
│   ├── hungary.data
│   ├── switzerland.data
│   └── long_beach_va.data
├── heart-disease.names         # Official UCI feature documentation
├── features.txt                # Additional feature notes
└── README.md                   # This file
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
-   `streamlit>=1.46` - Web app framework (requires 1.46+ for horizontal navigation)
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

### Using the App

1.  **Start at Home** to understand the project context
2.  **Apply filters** in the sidebar (optional - defaults to all data)
3.  **Navigate through sections** using the horizontal menu:
    -   Begin with Data Overview and Missing Data to see raw patterns
    -   Review Imputation and Encoding to understand cleaning steps
    -   Explore Distributions, Relationships, and Categories for key findings
    -   Check Prevalence for the most dramatic evidence
    -   Download cleaned data from Data Export
4.  **Read info boxes** at the bottom of each page for interpretation
5.  **Interact with charts** - hover for details, click legends to filter

------------------------------------------------------------------------

## Methodology Summary

### Data Processing Pipeline

1.  **Load** → Four datasets from UCI repository
2.  **Combine** → Add origin identifiers, create binary target
3.  **Analyze** → Identify missing patterns and institutional differences
4.  **Impute** → KNN (k=5) independently per origin
5.  **Encode** → Add human-readable labels alongside numeric codes
6.  **Export** → Clean dataset ready for modeling or further analysis

### Key Design Decisions

-   **Per-origin imputation**: Prevents data leakage across institutions
-   **KNN over mean/mode**: Better preserves distributions and relationships
-   **Preserve zeros**: Maintains data authenticity and reveals procedural differences
-   **Dual encoding**: Supports both modeling (numeric) and interpretation (labels)
-   **Binary target**: Simplifies problem while maintaining clinical relevance

------------------------------------------------------------------------

## Key Findings

1.  **Prevalence varies 2.7x** across origins (54% → 20%)
2.  **Feature correlations differ by 2x+** between institutions
3.  **Missing data patterns** reveal different testing procedures
4.  **Distribution shifts** challenge model generalization
5.  **Relationship structures** are population-dependent

**Implication**: Origin-aware modeling is essential for medical AI fairness and accuracy.

------------------------------------------------------------------------

## Data Source

**UCI Machine Learning Repository - Heart Disease Database** - Cleveland Clinic Foundation - Hungarian Institute of Cardiology, Budapest\
- University Hospital, Zurich, Switzerland - V.A. Medical Center, Long Beach, California

[UCI Repository Link](https://archive.ics.uci.edu/ml/datasets/heart+disease)

------------------------------------------------------------------------

## Author

**Wahid Hashem**\
CMSE 830 - Foundations of Data Science\
Michigan State University

------------------------------------------------------------------------

## License

This project uses publicly available data from the UCI Machine Learning Repository. Please cite the original data sources if using this work.