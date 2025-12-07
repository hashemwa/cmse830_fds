import streamlit as st

dfv = st.session_state.dfv
df = st.session_state.df

st.title(
    "Heart Disease Analysis Across Multiple Hospitals",
)
st.markdown("*Why Hospital-Specific Models Matter*")

st.subheader("Dataset Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Patients", f"{len(dfv):,}", help="Number of patients in filtered dataset"
    )

with col2:
    if "target" in dfv.columns and dfv["target"].notna().any():
        prevalence = 100 * dfv["target"].mean()
        st.metric(
            "Overall Prevalence",
            f"{prevalence:.1f}%",
            help="Percentage with heart disease",
        )

if {"origin", "target"}.issubset(dfv.columns):
    by_origin = dfv.groupby("origin", observed=True)["target"].mean().mul(100)
    if len(by_origin):
        with col3:
            st.metric(
                "Lowest Prevalence",
                f"{by_origin.min():.1f}%",
                delta=by_origin.idxmin(),
                delta_color="off",
                help="Minimum prevalence across origins",
            )

        with col4:
            st.metric(
                "Highest Prevalence",
                f"{by_origin.max():.1f}%",
                delta=by_origin.idxmax(),
                delta_color="off",
                help="Maximum prevalence across origins",
            )

st.divider()

st.subheader("Main Problem")
st.error(
    "**Origins differ in distributions, relationships, categorical mix, and prevalence.** "
    "A model trained on a single source (e.g., Cleveland only) may not generalize well to other populations/areas.",
    icon=":material/error:",
)

with st.expander("About This Dataset", icon=":material/dataset:", expanded=True):
    st.markdown(f"""
    This interactive app explores a **combined heart disease dataset** from four medical institutions:
    
    - :material/health_cross: **Cleveland Clinic Foundation** (Cleveland, USA)
    - :material/health_cross: **Hungarian Institute of Cardiology** (Budapest, Hungary)
    - :material/health_cross: **V.A. Medical Center** (Long Beach, USA)
    - :material/health_cross: **University Hospital** (Zurich, Switzerland)

    The dataset contains **14 clinical features** from **{len(df):,} patients** and examines how
    heart disease patterns vary across different populations and healthcare settings.
    """)

with st.expander("Methodology", icon=":material/science:"):
    st.markdown("""
    ### Data Processing Pipeline
    
    **1. Missing Value Handling**
    - Features with >50% missing in any origin are **excluded** (ca, thal, slope, fbs, chol)
    - Remaining features imputed using KNN or MICE per-origin
    - Note: High missingness cannot be reliably recovered by any method
    
    **2. Data Normalization**
    - Thalassemia codes normalized to `{3, 6, 7}`
    - Binary and ordinal features clamped to valid ranges
    - Age rounded to integers
    
    **3. Feature Engineering**
    - Created `target` variable: binary indicator (0 = no disease, 1 = disease present)
    - Added descriptive labels for categorical variables
    - Generated `origin` identifier for each dataset source
    
    **4. Exploratory Data Analysis**
    - Compared feature distributions across origins using KDE plots
    - Examined correlation structures per-origin with heatmaps
    - Analyzed categorical feature distributions and prevalence rates
    
    **5. Modeling**
    - Trained global models (Logistic Regression, Decision Tree) on all data
    - Trained stratified models (one per origin) to test hospital-specific patterns
    - Evaluated with accuracy, F1, ROC-AUC, and cross-validation
    """)

with st.expander("How to Use This App", icon=":material/explore:"):
    st.markdown("""
    ### :material/navigation: Navigation
    
    Use the **top navigation bar** to explore different analysis sections:
    - **Initial Data Analysis**: Raw data overview and missing data patterns
    - **Data Cleaning**: Imputation comparison and feature engineering
    - **Exploratory Data Analysis**: Distributions, relationships, categories, and prevalence
    - **Modeling**: Model development and thesis validation
    - **Data Export**: Download filtered data for your own analysis

    ### :material/tune: Filters (Sidebar)

    - **Origin Selection**: Choose which medical centers to include
    - **Age Range**: Filter patients by age
    
    All visualizations and statistics update automatically based on your selections.
    
    *Note: Modeling pages use all data (ignoring filters) to ensure proper train/test splitting.*
    
    ### :material/lightbulb_2: Tips
    
    - Start with **Data Overview** to understand the raw data structure
    - Review **Missing Data** to see origin-specific missingness patterns
    - Explore **Distributions** to see how variables differ across origins
    - Check **Model Development** to compare global vs stratified approaches
    """)

st.divider()

st.info(
    "**Data Source:** UCI Machine Learning Repository - Heart Disease Dataset (Cleveland, Hungary, Switzerland, Long Beach VA)",
    icon=":material/data_info_alert:",
)
