import streamlit as st

# Get filtered data from session state
dfv = st.session_state.dfv

st.title("Heart Disease EDA")
st.markdown("*One Size ≠ Fits All — Multi-Source Analysis*")

# Key metrics at the top
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

with col3:
    if {"origin", "target"}.issubset(dfv.columns):
        by_origin = dfv.groupby("origin")["target"].mean().mul(100)
        if len(by_origin):
            st.metric(
                "Lowest Prevalence",
                f"{by_origin.min():.1f}%",
                delta=by_origin.idxmin(),
                delta_color="off",
                help=f"Minimum prevalence across origins",
            )

with col4:
    if {"origin", "target"}.issubset(dfv.columns):
        by_origin = dfv.groupby("origin")["target"].mean().mul(100)
        if len(by_origin):
            st.metric(
                "Highest Prevalence",
                f"{by_origin.max():.1f}%",
                delta=by_origin.idxmax(),
                delta_color="off",
                help=f"Maximum prevalence across origins",
            )

st.divider()

# Key findings
st.subheader("Key Findings")
st.info(
    "**Origins differ in distributions, relationships, categorical mix, and prevalence.** "
    "A model trained on a single source (e.g., Cleveland only) may not generalize well to other populations.",
    icon=":material/lightbulb:",
)

# About section
with st.expander("About This Dataset", icon=":material/dataset:", expanded=True):
    st.markdown("""
    This interactive app explores a **combined heart disease dataset** from four medical institutions:
    
    - 🏥 **Cleveland Clinic Foundation** (Cleveland, USA)
    - 🏥 **Hungarian Institute of Cardiology** (Budapest, Hungary)  
    - 🏥 **V.A. Medical Center** (Long Beach, USA)
    - 🏥 **University Hospital** (Zurich, Switzerland)
    
    The dataset contains **14 clinical features** from **918 patients** and examines how 
    heart disease patterns vary across different populations and healthcare settings.
    """)

# Methodology section
with st.expander("Methodology", icon=":material/science:"):
    st.markdown("""
    ### Data Processing Pipeline
    
    **1. Missing Value Imputation**
    - KNN imputation (k=5) for numerical features
    - Preserves relationships between variables
    - Compares favorably to simple median/mode imputation
    
    **2. Data Normalization**
    - Thalassemia codes normalized to `{3, 6, 7}`
    - Binary and ordinal features clamped to valid ranges
    - Age rounded to integers
    
    **3. Feature Engineering**
    - Created `target` variable: binary indicator (0 = no disease, 1 = disease present)
    - Added descriptive labels for categorical variables
    - Generated `origin` identifier for each dataset source
    
    ### Analysis Focus
    
    This EDA investigates whether patterns in heart disease indicators are **consistent across origins** 
    or if there are **significant differences** that would affect model generalization.
    """)

# How to use section
with st.expander("How to Use This App", icon=":material/explore:"):
    st.markdown("""
    ### Navigation
    
    Use the **top navigation bar** to explore different analysis sections:
    - **Initial Data Analysis**: Raw data overview and missingness patterns
    - **Data Cleaning**: Imputation strategies and feature engineering
    - **Exploratory Data Analysis**: Distributions, relationships, and prevalence
    - **Download**: Export filtered data for your own analysis
    
    ### Filters (Sidebar)
    
    - **Origin Selection**: Choose which medical centers to include
    - **Age Range**: Filter patients by age
    
    All visualizations and statistics update automatically based on your selections.
    
    ### Tips
    
    - Start with **Data Overview** to understand the raw data
    - Compare **Imputation Comparison** to see how missing values were handled
    - Explore **Distribution Analysis** to see how variables differ across origins
    - Check **Prevalence** to understand disease rates by demographic factors
    """)

st.divider()

# Footer with dataset info
st.caption(
    "📊 **Data Source**: UCI Machine Learning Repository — Heart Disease Dataset | "
    "🔗 **Original Sources**: Cleveland, Hungarian, Long Beach VA, Switzerland"
)
