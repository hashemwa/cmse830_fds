import streamlit as st


# Get data from session state
dfv = st.session_state.dfv
df_raw_filtered = st.session_state.df_raw_filtered
df = st.session_state.df
origin_sel = st.session_state.origin_sel
age_range = st.session_state.age_range


st.title("Data Export")
st.markdown("*Download the cleaned, imputed dataset for your own analysis*")

st.markdown(
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
col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(dfv))
col2.metric("Columns", len(dfv.columns))
col3.metric("Size", f"{dfv.memory_usage(deep=True).sum() / 1024:.1f} KB")

st.download_button(
    "Download filtered data (CSV)",
    dfv.to_csv(index=False),
    file_name="filtered_heart_disease_data.csv",
    mime="text/csv",
    icon=":material/download:",
    use_container_width=True,
)

st.divider()

st.info(
    "**What you can do with this data:** Use the exported dataset for your own machine learning models, statistical analysis, "
    "or visualization projects. The data is clean, imputed, and ready for modeling. Both numeric codes and descriptive labels "
    "are included, so you can choose the format that works best for your use case. Remember that the origin variable captures "
    "important institutional differences that may need to be accounted for in your analysis.",
    icon=":material/info:",
)
