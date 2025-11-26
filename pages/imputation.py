import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.stats import gaussian_kde
from analysis import (
    get_individual_raw_datasets,
    get_individual_simple_imputed,
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
st.markdown("*Comparing Simple, KNN, and MICE Imputation Methods*")

st.warning(
    "**Methodology:** Each dataset was imputed **independently** to avoid data leakage across origins. "
    "This preserves the unique characteristics of each medical institution's data.",
    icon=":material/science:",
)

raw_datasets = get_individual_raw_datasets()
simple_datasets = get_individual_simple_imputed()
knn_datasets = get_individual_knn_imputed()
mice_datasets = get_individual_mice_imputed()

st.subheader("Individual Dataset Comparison")
st.caption("Compare imputation methods for each dataset source.")

tabs = st.tabs(["Cleveland", "Hungary", "Long Beach VA", "Switzerland"])

for idx, (dataset_name, tab) in enumerate(zip(raw_datasets.keys(), tabs)):
    with tab:
        raw_data = raw_datasets[dataset_name]
        simple_data = simple_datasets[dataset_name]
        knn_data = knn_datasets[dataset_name]
        mice_data = mice_datasets[dataset_name]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**Original**")
            missing_count = raw_data.isnull().sum().sum()
            st.metric("Missing Values", missing_count)

        with col2:
            st.markdown("**Simple**")
            missing_simple = simple_data.isnull().sum().sum()
            st.metric(
                "Missing Values",
                missing_simple,
            )

        with col3:
            st.markdown("**KNN**")
            missing_knn = knn_data.isnull().sum().sum()
            st.metric(
                "Missing Values",
                missing_knn,
            )

        with col4:
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
            simple_vals = simple_data["trestbps"].dropna().values
            knn_vals = knn_data["trestbps"].dropna().values
            mice_vals = mice_data["trestbps"].dropna().values

            if len(original_vals) > 1 and original_vals.std() > 0:
                x_min = min(
                    original_vals.min(),
                    simple_vals.min(),
                    knn_vals.min(),
                    mice_vals.min(),
                )
                x_max = max(
                    original_vals.max(),
                    simple_vals.max(),
                    knn_vals.max(),
                    mice_vals.max(),
                )
                x_smooth = np.linspace(x_min, x_max, 100)

                try:
                    kde_orig = gaussian_kde(original_vals, bw_method="scott")
                    kde_simple = gaussian_kde(simple_vals, bw_method="scott")
                    kde_knn = gaussian_kde(knn_vals, bw_method="scott")
                    kde_mice = gaussian_kde(mice_vals, bw_method="scott")

                    y_orig = kde_orig(x_smooth)
                    y_simple = kde_simple(x_smooth)
                    y_knn = kde_knn(x_smooth)
                    y_mice = kde_mice(x_smooth)

                    chart_df = pd.DataFrame(
                        {
                            "trestbps (mmHg)": np.tile(x_smooth, 4),
                            "Density": np.concatenate(
                                [y_orig, y_simple, y_knn, y_mice]
                            ),
                            "Method": np.repeat(
                                [
                                    "Original",
                                    "Simple Imputation",
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
                                        "Simple Imputation",
                                        "KNN Imputation",
                                        "MICE Imputation",
                                    ],
                                    range=["#4c78a8", "#f58518", "#54a24b", "#e45756"],
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
                        "KNN and MICE imputation preserve the original distribution shape better than simple mean imputation."
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
    **Simple Imputation** (mean/mode):
    - Replaces missing values with the overall mean (numerical) or mode (categorical)
    - Fast and straightforward
    - :material/cancel: Reduces variance and ignores relationships between features
    
    **KNN Imputation** (k=5):
    - Uses the 5 most similar patients to estimate missing values
    - Preserves local structure and feature relationships
    - :material/check_circle: Better maintains the original data distribution
    
    **MICE (Multivariate Imputation by Chained Equations)**:
    - Models each feature with missing values as a function of other features
    - Iteratively estimates missing values
    - :material/check_circle: Very robust, accounts for uncertainty, and handles complex relationships
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
            )
        else:
            st.info("No numerical features available")

    with col2:
        st.markdown(f"**After {imputation_method}**")
        if available_num:
            st.dataframe(
                combined_imputed[available_num].describe().round(2),
                use_container_width=True,
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
            st.dataframe(cat_summary, use_container_width=True)
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
            st.dataframe(cat_summary, use_container_width=True)
        else:
            st.info("No categorical features available")

st.divider()

st.info(
    "**Why this matters:** How we fill in missing values directly affects everything we learn from the data. "
    "KNN and MICE imputation preserve the relationships between features and create realistic values based on similar patients. "
    "Simple mean/mode imputation would make all patients look more similar than they really are, "
    "which could mislead any predictions or conclusions we make from this data.",
    icon=":material/info:",
)
