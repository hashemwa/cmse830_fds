import streamlit as st

dfv = st.session_state.dfv
df_raw_filtered = st.session_state.df_raw_filtered
df = st.session_state.df
origin_sel = st.session_state.origin_sel
age_range = st.session_state.age_range


st.title("Data Export")
st.markdown("*Download the cleaned, imputed dataset for your own analysis*")

st.caption(
    "Export the fully processed dataset with KNN imputation, encoded labels, and applied filters. "
    "The data includes both numeric codes (for modeling) and human-readable labels (for interpretation)."
)

st.subheader("Filtered Data Preview")

show_cols = [
    c
    for c in [
        "age",
        "origin",
        "sex",
        "cp",
        "cp_label",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "restecg_label",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "slope_label",
        "ca",
        "thal",
        "thal_label",
        "num",
        "num_label",
        "target",
    ]
    if c in dfv.columns
]

st.dataframe(
    dfv[show_cols].head(500) if show_cols else dfv.head(500), use_container_width=True
)

st.subheader("Export Summary")

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
col1.metric("Rows", len(dfv))
col2.metric("Columns", len(dfv.columns))
col3.metric("Size", f"{dfv.memory_usage(deep=True).sum() / 1024:.1f} KB")

with col4:
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        "Download CSV",
        dfv.to_csv(index=False),
        file_name="filtered_heart_disease_data.csv",
        mime="text/csv",
        icon=":material/download:",
        use_container_width=True,
        type="primary",
    )

st.divider()

st.info(
    "**What you can do with this data:** Use this cleaned dataset for your own machine learning projects, statistical analysis, "
    "or visualizations. The missing values have been filled in using KNN imputation, so it's ready to use. "
    "You'll find both numeric codes (for building models) and text labels (for understanding results). "
    "Just remember that the 'origin' column shows which hospital the data came from. This matters for accurate analysis "
    "because each hospital's data has different patterns.",
    icon=":material/info:",
)
