import streamlit as st
from analysis import missingness_heatmap

# Get filtered data from session state
df_raw_filtered = st.session_state.df_raw_filtered

st.title("Missing Data")
st.markdown(
    "*Analysis of missing values in the **original raw data** before imputation*"
)

if df_raw_filtered.isnull().values.any():
    st.subheader("Missingness Heatmap by Origin")
    num_cols = [
        c
        for c in [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
            "num",
        ]
        if c in df_raw_filtered.columns
    ]
    chart = missingness_heatmap(df_raw_filtered, num_cols)
    st.altair_chart(chart, use_container_width=True)

    st.caption(
        "Missingness varies by origin → preprocessing must be origin-aware. "
        "A single imputation recipe can bias results."
    )
else:
    st.success("✅ No missing values in the filtered raw data!")
