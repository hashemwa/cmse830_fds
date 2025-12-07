import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.stats import gaussian_kde
from analysis import (
    get_individual_raw_datasets,
    get_individual_knn_imputed,
    get_individual_mice_imputed,
    get_combined_knn_imputed,
    get_combined_mice_imputed,
    get_raw_data,
)

dfv = st.session_state.dfv
df_raw_filtered = st.session_state.df_raw_filtered
df = st.session_state.df
origin_sel = st.session_state.origin_sel
age_range = st.session_state.age_range


st.title("Imputation")
st.markdown("*Comparing KNN and MICE Imputation Methods*")

st.warning(
    "**Methodology:** Each dataset was imputed **independently** to avoid data leakage across origins. "
    "This preserves the unique characteristics of each medical institution's data.",
    icon=":material/science:",
)

st.error(
    "**Important Limitation:** Some features have 60-100% missing values in certain origins "
    "(e.g., `chol` in Switzerland, `ca` and `thal` in Hungary). No imputation method can reliably recover "
    "data that was never collected. These features are **excluded from modeling**. Imputation is only applied "
    "to features with moderate missingness where patterns can be reasonably estimated.",
    icon=":material/warning:",
)

raw_datasets = get_individual_raw_datasets()
knn_datasets = get_individual_knn_imputed()
mice_datasets = get_individual_mice_imputed()

st.subheader("Individual Dataset Comparison")
st.caption("Compare imputation methods for each dataset source.")

tabs = st.tabs(["Cleveland", "Hungary", "Long Beach VA", "Switzerland"])

for idx, (dataset_name, tab) in enumerate(zip(raw_datasets.keys(), tabs)):
    with tab:
        raw_data = raw_datasets[dataset_name]
        knn_data = knn_datasets[dataset_name]
        mice_data = mice_datasets[dataset_name]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Original**")
            missing_count = raw_data.isnull().sum().sum()
            st.metric("Missing Values", missing_count)

        with col2:
            st.markdown("**KNN**")
            missing_knn = knn_data.isnull().sum().sum()
            st.metric(
                "Missing Values",
                missing_knn,
            )

        with col3:
            st.markdown("**MICE**")
            missing_mice = mice_data.isnull().sum().sum()
            st.metric(
                "Missing Values",
                missing_mice,
            )

        st.markdown("**Distribution Comparison Example: Resting Blood Pressure**")
        st.caption(
            "Representative example showing how imputation methods affect distribution shape. "
            "See summary statistics below for detailed metrics across all variables."
        )

        if "trestbps" in raw_data.columns:
            original_vals = raw_data["trestbps"].dropna().values
            knn_vals = knn_data["trestbps"].dropna().values
            mice_vals = mice_data["trestbps"].dropna().values

            if len(original_vals) > 1 and original_vals.std() > 0:
                x_min = min(
                    original_vals.min(),
                    knn_vals.min(),
                    mice_vals.min(),
                )
                x_max = max(
                    original_vals.max(),
                    knn_vals.max(),
                    mice_vals.max(),
                )
                x_smooth = np.linspace(x_min, x_max, 100)

                try:
                    kde_orig = gaussian_kde(original_vals, bw_method="scott")
                    kde_knn = gaussian_kde(knn_vals, bw_method="scott")
                    kde_mice = gaussian_kde(mice_vals, bw_method="scott")

                    y_orig = kde_orig(x_smooth)
                    y_knn = kde_knn(x_smooth)
                    y_mice = kde_mice(x_smooth)

                    chart_df = pd.DataFrame(
                        {
                            "trestbps (mmHg)": np.tile(x_smooth, 3),
                            "Density": np.concatenate([y_orig, y_knn, y_mice]),
                            "Method": np.repeat(
                                [
                                    "Original",
                                    "KNN Imputation",
                                    "MICE Imputation",
                                ],
                                len(x_smooth),
                            ),
                        }
                    )

                    hover = alt.selection_point(
                        fields=["trestbps (mmHg)"],
                        nearest=True,
                        on="mouseover",
                        empty=False,
                    )

                    lines = (
                        alt.Chart(chart_df)
                        .mark_line(size=3, interpolate="natural")
                        .encode(
                            x=alt.X(
                                "trestbps (mmHg):Q",
                                title="Resting Blood Pressure (mmHg)",
                                axis=alt.Axis(grid=False),
                            ),
                            y=alt.Y(
                                "Density:Q",
                                title="Density",
                                axis=alt.Axis(gridOpacity=0.5),
                            ),
                            color=alt.Color(
                                "Method:N",
                                scale=alt.Scale(
                                    domain=[
                                        "Original",
                                        "KNN Imputation",
                                        "MICE Imputation",
                                    ],
                                    range=["#4c78a8", "#54a24b", "#e45756"],
                                ),
                                legend=alt.Legend(title="Method", orient="right"),
                            ),
                        )
                    )

                    points = (
                        lines.mark_point(size=100)
                        .encode(
                            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                            tooltip=[
                                alt.Tooltip("Method:N", title="Method"),
                                alt.Tooltip(
                                    "trestbps (mmHg):Q",
                                    title="Resting Blood Pressure (mmHg)",
                                    format=".1f",
                                ),
                                alt.Tooltip("Density:Q", title="Density", format=".4f"),
                            ],
                        )
                        .add_params(hover)
                    )

                    rule = (
                        alt.Chart(chart_df)
                        .mark_rule(color="gray", strokeWidth=1)
                        .encode(
                            x="trestbps (mmHg):Q",
                            opacity=alt.condition(hover, alt.value(0.5), alt.value(0)),
                        )
                        .transform_filter(hover)
                    )

                    chart = (lines + points + rule).properties(height=400)

                    st.altair_chart(chart, use_container_width=True)
                    st.caption(
                        "Note: Higher missingness leads to greater distribution shift. "
                        "Long Beach VA (~29% missing) shows more distortion than Cleveland (~0% missing)."
                    )
                except Exception:
                    st.warning(
                        "Unable to generate distribution comparison for this dataset."
                    )

st.divider()

st.subheader("Combined Dataset: Imputation Results")
st.caption(
    "Compare the combined raw data with the final cleaned dataset after imputation."
)

combined_raw = get_raw_data()
combined_knn = get_combined_knn_imputed()
combined_mice = get_combined_mice_imputed()

imputation_method = st.radio(
    "Select Imputation Method to Compare:",
    ["KNN Imputation", "MICE Imputation"],
    horizontal=True,
)

if imputation_method == "KNN Imputation":
    combined_imputed = combined_knn
else:
    combined_imputed = combined_mice

num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
available_num = [c for c in num_cols if c in combined_imputed.columns]

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
available_cat = [c for c in cat_cols if c in combined_imputed.columns]

with st.expander("Imputation Methods Explained", icon=":material/info:"):
    st.markdown("""
    **KNN Imputation** (k=5):
    - Uses the 5 most similar patients to estimate missing values
    - Works well when missingness is low (<20%)
    - Higher missingness leads to greater distribution distortion
    - Assumes data is Missing At Random (MAR)
    
    **MICE (Multivariate Imputation by Chained Equations)**:
    - Models each feature with missing values as a function of other features
    - Iteratively estimates missing values
    - More robust for complex relationships, but still limited by data availability
    - Also assumes data is Missing At Random (MAR)
    
    **Neither method can recover data that was never collected.**
    """)

st.markdown(f"**Summary Statistics Comparison: Raw vs {imputation_method}**")

feature_tabs = st.tabs(["Numerical", "Categorical"])

with feature_tabs[0]:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original (Raw Data)**")
        if available_num:
            st.dataframe(
                combined_raw[available_num].describe().round(2),
                use_container_width=True,
                hide_index=False,
                height=320,
            )
        else:
            st.info("No numerical features available")

    with col2:
        st.markdown(f"**After {imputation_method}**")
        if available_num:
            st.dataframe(
                combined_imputed[available_num].describe().round(2),
                use_container_width=True,
                hide_index=False,
                height=320,
            )
        else:
            st.info("No numerical features available")

with feature_tabs[1]:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original (Raw Data)**")
        if available_cat:
            cat_stats = []
            for col in available_cat:
                value_counts = combined_raw[col].value_counts()
                cat_stats.append(
                    {
                        "count": combined_raw[col].notna().sum(),
                        "unique": combined_raw[col].nunique(),
                        "top": value_counts.index[0] if len(value_counts) > 0 else None,
                        "freq": value_counts.iloc[0] if len(value_counts) > 0 else None,
                    }
                )
            cat_summary = pd.DataFrame(cat_stats, index=available_cat).T
            st.dataframe(cat_summary, use_container_width=True, height=180)
        else:
            st.info("No categorical features available")

    with col2:
        st.markdown(f"**After {imputation_method}**")
        if available_cat:
            cat_stats = []
            for col in available_cat:
                value_counts = combined_imputed[col].value_counts()
                cat_stats.append(
                    {
                        "count": combined_imputed[col].notna().sum(),
                        "unique": combined_imputed[col].nunique(),
                        "top": value_counts.index[0] if len(value_counts) > 0 else None,
                        "freq": value_counts.iloc[0] if len(value_counts) > 0 else None,
                    }
                )
            cat_summary = pd.DataFrame(cat_stats, index=available_cat).T
            st.dataframe(cat_summary, use_container_width=True, height=180)
        else:
            st.info("No categorical features available")

st.divider()

st.info(
    "**Why this matters:** Imputation works best for features with low to moderate missingness, "
    "where there's enough data to estimate patterns. For features with extreme missingness (>50%), "
    "no method can reliably recover what was never measured, so those features are excluded from modeling entirely.",
    icon=":material/info:",
)
