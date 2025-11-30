import streamlit as st
from analysis import get_combined_knn_imputed, get_combined_mice_imputed

origin_sel = st.session_state.origin_sel


st.title("Data Export")
st.markdown("*Download the cleaned, imputed dataset for your own analysis*")

# Let user choose imputation method
imputation_choice = st.radio(
    "Select Imputation Method:",
    ["KNN Imputation", "MICE Imputation"],
    horizontal=True,
    help="Choose which imputation method was used to fill in missing values",
)

if imputation_choice == "KNN Imputation":
    df_export = get_combined_knn_imputed()
    method_name = "KNN"
else:
    df_export = get_combined_mice_imputed()
    method_name = "MICE"

# Apply origin filter from sidebar (list of selected origins)
if origin_sel and "origin" in df_export.columns:
    df_export = df_export[df_export["origin"].isin(origin_sel)]

st.caption(
    f"Export the fully processed dataset with {method_name} imputation, encoded labels, and applied filters. "
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
    if c in df_export.columns
]

st.dataframe(
    df_export[show_cols].head(500) if show_cols else df_export.head(500),
    use_container_width=True,
)

st.subheader("Export Summary")

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
col1.metric("Rows", len(df_export))
col2.metric("Columns", len(df_export.columns))
col3.metric("Size", f"{df_export.memory_usage(deep=True).sum() / 1024:.1f} KB")

with col4:
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        "Download CSV",
        df_export.to_csv(index=False),
        file_name=f"heart_disease_{method_name.lower()}_imputed.csv",
        mime="text/csv",
        icon=":material/download:",
        use_container_width=True,
        type="primary",
    )

st.divider()

st.warning(
    "**Note:** Features with >50% missing data (ca, thal, slope, fbs, chol) are included in this export "
    "but should be used with caution. These features are excluded from the modeling page due to unreliable imputation.",
    icon=":material/warning:",
)

st.info(
    f"**What you can do with this data:** Use this cleaned dataset for your own machine learning projects, statistical analysis, "
    f"or visualizations. The missing values have been filled in using {method_name} imputation. "
    "You'll find both numeric codes (for building models) and text labels (for understanding results). "
    "The 'origin' column shows which hospital the data came from. This matters because each hospital's data has different patterns.",
    icon=":material/info:",
)
