import streamlit as st
from analysis import get_individual_raw_datasets

# Get filtered data from session state
df_raw_filtered = st.session_state.df_raw_filtered
origin_sel = st.session_state.origin_sel
age_range = st.session_state.age_range

st.title("Data Overview")
st.markdown(
    "*Initial Data Analysis — Exploring Individual Datasets Before Combination*"
)

# Feature descriptions expander
with st.expander("Feature Descriptions & Encoding", icon=":material/info:"):
    st.markdown("""
    ### ⚙️ Derived Variables
    
    **origin**: Dataset source identifier (added to each individual dataset before combination)
    - `Cleveland` - Cleveland Clinic Foundation
    - `Hungary` - Hungarian Institute of Cardiology, Budapest
    - `Long Beach VA` - V.A. Medical Center, Long Beach
    - `Switzerland` - University Hospital, Zurich, Switzerland
    
    **target**: Binary heart disease indicator (derived from `num` in each dataset before combination)
    - `0` = no heart disease (`num = 0`)
    - `1` = heart disease present (`num ≥ 1`)
    - Simplifies multi-class diagnosis into presence/absence for binary classification
    
    ---
    
    ### Numerical Features
    - **age**: Age in years (range: 29-77)
    - **trestbps**: Resting blood pressure on admission (mm Hg)
    - **chol**: Serum cholesterol (mg/dL)
    - **thalach**: Maximum heart rate achieved during exercise
    - **oldpeak**: ST depression induced by exercise relative to rest
    - **ca**: Number of major vessels (0-3) colored by fluoroscopy
    
    ### Categorical Features (Encoded as Numbers)
    
    **sex**: `1 = male`, `0 = female`
    
    **cp** (Chest pain type): 
    - `1 = typical angina`
    - `2 = atypical angina`
    - `3 = non-anginal pain`
    - `4 = asymptomatic`
    
    **fbs** (Fasting blood sugar > 120 mg/dL): `1 = true`, `0 = false`
    
    **restecg** (Resting electrocardiogram results):
    - `0 = normal`
    - `1 = ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)`
    - `2 = Probable or definite left ventricular hypertrophy (LVH) by Estes' criteria`
    
    **exang** (Exercise-induced angina): `1 = yes`, `0 = no`
    
    **slope** (Slope of the peak exercise ST segment):
    - `1 = upsloping`
    - `2 = flat`
    - `3 = downsloping`
    
    **thal** (Thalassemia blood disorder):
    - `3 = normal`
    - `6 = fixed defect`
    - `7 = reversible defect`
    
    **num** (Original diagnosis - angiographic disease status, 0-4 scale):
    - `0 = no disease (< 50% diameter narrowing)`
    - `1 = mild heart disease`
    - `2 = moderate heart disease`
    - `3 = severe heart disease`
    - `4 = critical heart disease`
    """)

# Get individual raw datasets
raw_datasets = get_individual_raw_datasets()

st.subheader("Individual Dataset Analysis")
st.markdown(
    "Each dataset was collected from different medical institutions with varying data collection practices."
)

# Create tabs for each dataset
tabs = st.tabs(["Cleveland", "Hungary", "Long Beach VA", "Switzerland"])

for idx, (dataset_name, tab) in enumerate(zip(raw_datasets.keys(), tabs)):
    with tab:
        dataset = raw_datasets[dataset_name]

        # Dataset metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", len(dataset))
        col2.metric("Columns", len(dataset.columns))
        missing_count = dataset.isnull().sum().sum()
        col3.metric("Missing Values", missing_count)
        if "target" in dataset.columns:
            prevalence = 100 * dataset["target"].mean()
            col4.metric("Prevalence", f"{prevalence:.1f}%")

        # Data preview
        st.markdown("**Data Preview**")
        st.dataframe(dataset.head(10), use_container_width=True)

        # Summary statistics with tabs
        st.markdown("**Summary Statistics**")

        num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
        available_num = [c for c in num_cols if c in dataset.columns]

        cat_cols = [
            "sex",
            "cp",
            "fbs",
            "restecg",
            "exang",
            "slope",
            "thal",
            "num",
            "target",
        ]
        available_cat = [c for c in cat_cols if c in dataset.columns]

        stat_tabs = st.tabs(["Numerical", "Categorical"])

        with stat_tabs[0]:
            if available_num:
                st.dataframe(
                    dataset[available_num].describe().round(2),
                    use_container_width=True,
                )
            else:
                st.info("No numerical features available")

        with stat_tabs[1]:
            if available_cat:
                cat_summary = dataset[available_cat].describe(include="all").round(2)
                st.dataframe(cat_summary, use_container_width=True)
            else:
                st.info("No categorical features available")

st.divider()

# Combined dataset analysis
st.subheader("Combined Dataset Summary")
st.markdown("After combining all four datasets, here are the overall statistics:")

# Combined dataset metrics (using raw filtered data)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Rows", f"{len(df_raw_filtered):,}")
col2.metric("Columns", len(df_raw_filtered.columns))
missing_count = df_raw_filtered.isnull().sum().sum()
col3.metric("Missing Values", missing_count)
if "target" in df_raw_filtered.columns:
    prevalence = 100 * df_raw_filtered["target"].mean()
    col4.metric("Prevalence", f"{prevalence:.1f}%")

# Additional filter info
if origin_sel or age_range:
    st.info(
        f"**Filters Applied:** Origins: {', '.join(origin_sel)}"
        + (f" | Age range: {age_range[0]}-{age_range[1]}" if age_range else "")
    )

st.markdown("**Summary Statistics (Combined Raw Data)**")

num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
available_num = [c for c in num_cols if c in df_raw_filtered.columns]

cat_cols = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
    "num",
    "target",
    "origin",
]
available_cat = [c for c in cat_cols if c in df_raw_filtered.columns]

combined_stat_tabs = st.tabs(["Numerical", "Categorical"])

with combined_stat_tabs[0]:
    if available_num:
        st.dataframe(
            df_raw_filtered[available_num].describe().round(2),
            use_container_width=True,
        )
    else:
        st.info("No numerical features available")

with combined_stat_tabs[1]:
    if available_cat:
        cat_summary = df_raw_filtered[available_cat].describe(include="all").round(2)
        st.dataframe(cat_summary, use_container_width=True)
    else:
        st.info("No categorical features available")
