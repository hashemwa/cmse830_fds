import streamlit as st
import pandas as pd
from analysis import missingness_heatmap

# Get filtered data from session state
df_raw_filtered = st.session_state.df_raw_filtered

st.title("Missing Data")
st.markdown("*Understanding missing values in the **raw data** before imputation*")

# Overall missingness metrics
total_cells = df_raw_filtered.size
missing_cells = df_raw_filtered.isnull().sum().sum()
missing_pct = 100 * missing_cells / total_cells if total_cells > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cells", f"{total_cells:,}")
col2.metric("Missing Cells", f"{missing_cells:,}")
col3.metric("Missing %", f"{missing_pct:.2f}%")
features_with_missing = (df_raw_filtered.isnull().sum() > 0).sum()
col4.metric(
    "Features Affected", f"{features_with_missing}/{len(df_raw_filtered.columns)}"
)

st.divider()

if df_raw_filtered.isnull().values.any():
    # Feature-level missingness breakdown
    st.subheader("Missing Data by Feature")

    missing_summary = pd.DataFrame(
        {
            "Missing Count": df_raw_filtered.isnull().sum(),
            "Missing %": 100 * df_raw_filtered.isnull().sum() / len(df_raw_filtered),
        }
    ).sort_values("Missing Count", ascending=False)

    # Only show features with missing data
    missing_summary = missing_summary[missing_summary["Missing Count"] > 0]

    if not missing_summary.empty:
        st.dataframe(
            missing_summary.round(2),
            use_container_width=True,
            column_config={
                "Missing Count": st.column_config.NumberColumn(format="%d"),
                "Missing %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

        st.caption(
            ":material/lightbulb: **Key observation:** Some features like `ca` and `thal` have substantial missing data."
        )

    st.divider()

    # Missingness heatmap
    st.subheader("Missingness Patterns by Origin")
    st.caption(
        "This heatmap shows the percentage of missing values for each feature across different data sources."
    )

    features = [
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

    chart = missingness_heatmap(df_raw_filtered, features)
    st.altair_chart(chart, use_container_width=True)

    st.info(
        "**Why this matters:** The missing data patterns prove that hospitals collected data in fundamentally different ways. "
        "Some hospitals are missing entire features (like `ca` and `thal`) while others have complete data. "
        "This isn't random. It tells us that hospitals used different equipment, ran different tests, and followed different protocols. "
        "If we ignore these differences and just combine everything together, we're pretending four different hospitals are the same. "
        "That's why a 'one size fits all' model trained on mixed data will fail when used at any single hospital.",
        icon=":material/info:",
    )
else:
    st.success(
        "No missing values in the filtered raw data!", icon=":material/check_circle:"
    )
