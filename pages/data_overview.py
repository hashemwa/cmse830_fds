import streamlit as st
import pandas as pd
from analysis import get_individual_raw_datasets

df_raw_filtered = st.session_state.df_raw_filtered
origin_sel = st.session_state.origin_sel
age_range = st.session_state.age_range

st.title("Data Overview")
st.markdown("*Exploring Individual Datasets Before Combination*")

st.warning(
    "**Data Quality Note:** Some hospitals recorded missing values as zeros instead of leaving them blank. "
    "For example, `chol` (cholesterol) = 0 for all Switzerland patients, and some Long Beach VA patients have `chol` = 0 or `trestbps` = 0. "
    "Since these values are clinically impossible, they are treated as missing data and converted to NaN before analysis.",
    icon=":material/warning:",
)

with st.expander("Feature Descriptions & Encoding", icon=":material/info:"):
    st.markdown("""
    ### :material/settings: Derived Variables
    
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
    - **age**: Age in years
    - **trestbps**: Resting blood pressure (in mm Hg on admission to the hospital)
    - **chol**: Serum cholesterol in mg/dL
    - **thalach**: Maximum heart rate achieved
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
    
    **restecg** (Resting electrocardiographic results):
    - `0 = normal`
    - `1 = having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)`
    - `2 = showing probable or definite left ventricular hypertrophy by Estes' criteria`
    
    **exang** (Exercise induced angina): `1 = yes`, `0 = no`
    
    **slope** (The slope of the peak exercise ST segment):
    - `1 = upsloping`
    - `2 = flat`
    - `3 = downsloping`
    
    **thal** (Thalassemia): 
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

raw_datasets = get_individual_raw_datasets()

st.subheader("Individual Dataset Analysis")
st.caption(
    "Each dataset was collected from different medical institutions with varying data collection practices."
)

tabs = st.tabs(["Cleveland", "Hungary", "Long Beach VA", "Switzerland"])

for idx, (dataset_name, tab) in enumerate(zip(raw_datasets.keys(), tabs)):
    with tab:
        dataset = raw_datasets[dataset_name]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", len(dataset))
        col2.metric("Columns", len(dataset.columns))
        missing_count = dataset.isnull().sum().sum()
        col3.metric("Missing Values", missing_count)
        if "target" in dataset.columns:
            prevalence = 100 * dataset["target"].mean()
            col4.metric("Prevalence", f"{prevalence:.1f}%")

        st.markdown("**Data Preview**")
        st.dataframe(dataset.head(10), use_container_width=True)

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
                cat_stats = []
                for col in available_cat:
                    value_counts = dataset[col].value_counts()
                    cat_stats.append(
                        {
                            "count": dataset[col].notna().sum(),
                            "unique": dataset[col].nunique(),
                            "top": value_counts.index[0]
                            if len(value_counts) > 0
                            else None,
                            "freq": value_counts.iloc[0]
                            if len(value_counts) > 0
                            else None,
                        }
                    )
                cat_summary = pd.DataFrame(cat_stats, index=available_cat).T
                st.dataframe(cat_summary, use_container_width=True)
            else:
                st.info("No categorical features available")

st.divider()

st.subheader("Combined Dataset Summary")
st.caption("After combining all four datasets, here are the overall statistics:")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Rows", f"{len(df_raw_filtered):,}")
col2.metric("Columns", len(df_raw_filtered.columns))
missing_count = df_raw_filtered.isnull().sum().sum()
col3.metric("Missing Values", missing_count)
if "target" in df_raw_filtered.columns:
    prevalence = 100 * df_raw_filtered["target"].mean()
    col4.metric("Prevalence", f"{prevalence:.1f}%")

if origin_sel and len(origin_sel) < 4:
    origins_text = ", ".join(origin_sel)
    age_text = (
        f" | Age: {age_range[0]}-{age_range[1]}"
        if age_range
        and age_range != (df_raw_filtered["age"].min(), df_raw_filtered["age"].max())
        else ""
    )
    st.info(
        f"**Active Filters:** {origins_text}{age_text}", icon=":material/filter_alt:"
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
        cat_stats = []
        for col in available_cat:
            value_counts = df_raw_filtered[col].value_counts()
            cat_stats.append(
                {
                    "count": df_raw_filtered[col].notna().sum(),
                    "unique": df_raw_filtered[col].nunique(),
                    "top": value_counts.index[0] if len(value_counts) > 0 else None,
                    "freq": value_counts.iloc[0] if len(value_counts) > 0 else None,
                }
            )
        cat_summary = pd.DataFrame(cat_stats, index=available_cat).T
        st.dataframe(cat_summary, use_container_width=True)
    else:
        st.info("No categorical features available")

st.divider()

st.info(
    "**Why this matters:** We need to understand the raw data from each hospital before doing any analysis. "
    "The different sample sizes, missing values, and feature patterns show that each institution collected data differently. "
    "If we simply deleted all rows with missing values, we'd lose entire features. "
    "And if we ignored these differences between hospitals, our analysis could give us wrong answers.",
    icon=":material/info:",
)
