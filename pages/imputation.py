import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.stats import gaussian_kde

# Import analysis functions as needed
from analysis import (
    get_individual_raw_datasets,
    get_individual_simple_imputed,
    get_individual_knn_imputed,
    get_combined_knn_imputed,
    get_raw_data,
)

# Get data from session state
dfv = st.session_state.dfv
df_raw_filtered = st.session_state.df_raw_filtered
df = st.session_state.df
origin_sel = st.session_state.origin_sel
age_range = st.session_state.age_range


st.title("Imputation")
st.markdown("*Comparing Simple vs KNN Imputation Methods*")

# Methodology explanation
st.warning(
    "**Methodology:** Each dataset was imputed **independently** to avoid data leakage across origins. "
    "This preserves the unique characteristics of each medical institution's data.",
    icon=":material/science:",
)

with st.expander("Why KNN Imputation?", icon=":material/info:"):
    st.markdown("""
    **Simple Imputation** (mean/mode):
    - Replaces missing values with the overall mean (numerical) or mode (categorical)
    - Fast and straightforward
    - :material/cancel: Reduces variance and ignores relationships between features
    
    **KNN Imputation** (k=5):
    - Uses the 5 most similar patients to estimate missing values
    - Preserves local structure and feature relationships
    - :material/check_circle: Better maintains the original data distribution
    """)

# Get imputed datasets
raw_datasets = get_individual_raw_datasets()
simple_datasets = get_individual_simple_imputed()
knn_datasets = get_individual_knn_imputed()

st.subheader("Individual Dataset Comparison")
st.caption("Compare imputation methods for each dataset source.")

# Create tabs for each dataset
tabs = st.tabs(["Cleveland", "Hungary", "Long Beach VA", "Switzerland"])

for idx, (dataset_name, tab) in enumerate(zip(raw_datasets.keys(), tabs)):
    with tab:
        raw_data = raw_datasets[dataset_name]
        simple_data = simple_datasets[dataset_name]
        knn_data = knn_datasets[dataset_name]

        # Metrics comparison
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Original**")
            missing_count = raw_data.isnull().sum().sum()
            st.metric("Missing Values", missing_count)

        with col2:
            st.markdown("**Simple Imputation**")
            missing_simple = simple_data.isnull().sum().sum()
            st.metric(
                "Missing Values",
                missing_simple,
            )

        with col3:
            st.markdown("**KNN Imputation**")
            missing_knn = knn_data.isnull().sum().sum()
            st.metric(
                "Missing Values",
                missing_knn,
            )

        # Distribution comparison example
        st.markdown("**Distribution Comparison Example: Resting Blood Pressure**")
        st.caption(
            "Representative example showing how imputation methods affect distribution shape. "
            "See summary statistics below for detailed metrics across all variables."
        )

        if "trestbps" in raw_data.columns:
            # Prepare data for KDE
            original_vals = raw_data["trestbps"].dropna().values
            simple_vals = simple_data["trestbps"].dropna().values
            knn_vals = knn_data["trestbps"].dropna().values

            if len(original_vals) > 1 and original_vals.std() > 0:
                x_min = min(original_vals.min(), simple_vals.min(), knn_vals.min())
                x_max = max(original_vals.max(), simple_vals.max(), knn_vals.max())
                x_smooth = np.linspace(x_min, x_max, 100)  # Reduced from 200 to 100

                # Compute KDE values vectorized (much faster)
                try:
                    kde_orig = gaussian_kde(original_vals, bw_method="scott")
                    kde_simple = gaussian_kde(simple_vals, bw_method="scott")
                    kde_knn = gaussian_kde(knn_vals, bw_method="scott")

                    # Evaluate all at once
                    y_orig = kde_orig(x_smooth)
                    y_simple = kde_simple(x_smooth)
                    y_knn = kde_knn(x_smooth)

                    # Create DataFrame efficiently
                    chart_df = pd.DataFrame(
                        {
                            "trestbps (mmHg)": np.tile(x_smooth, 3),
                            "Density": np.concatenate([y_orig, y_simple, y_knn]),
                            "Method": np.repeat(
                                ["Original", "Simple Imputation", "KNN Imputation"],
                                len(x_smooth),
                            ),
                        }
                    )

                    # Use line chart for better visibility of overlapping distributions
                    chart = (
                        alt.Chart(chart_df)
                        .mark_line(size=3)
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
                                    ],
                                    range=["#4c78a8", "#f58518", "#54a24b"],
                                ),
                                legend=alt.Legend(title="Method", orient="right"),
                            ),
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
                        .properties(height=400)
                    )

                    st.altair_chart(chart, use_container_width=True)
                    st.caption(
                        "KNN imputation preserves the original distribution shape better than simple mean imputation."
                    )
                except Exception:
                    st.warning(
                        "Unable to generate distribution comparison for this dataset."
                    )

st.divider()

# Combined dataset comparison
st.subheader("Combined Dataset: Before vs After KNN Imputation")
st.caption(
    "Compare the combined raw data with the final cleaned dataset after KNN imputation."
)

combined_raw = get_raw_data()
combined_knn = get_combined_knn_imputed()

num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
available_num = [c for c in num_cols if c in combined_knn.columns]

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
available_cat = [c for c in cat_cols if c in combined_knn.columns]

# Summary statistics comparison with single tab control
st.markdown("**Summary Statistics Comparison**")

feature_tabs = st.tabs(["Numerical", "Categorical"])

with feature_tabs[0]:
    # Numerical features side-by-side
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
        st.markdown("**After KNN Imputation**")
        if available_num:
            st.dataframe(
                combined_knn[available_num].describe().round(2),
                use_container_width=True,
            )
        else:
            st.info("No numerical features available")

with feature_tabs[1]:
    # Categorical features side-by-side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original (Raw Data)**")
        if available_cat:
            # Calculate proper categorical statistics
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
        st.markdown("**After KNN Imputation**")
        if available_cat:
            # Calculate proper categorical statistics
            cat_stats = []
            for col in available_cat:
                value_counts = combined_knn[col].value_counts()
                cat_stats.append(
                    {
                        "count": combined_knn[col].notna().sum(),
                        "unique": combined_knn[col].nunique(),
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
    "KNN imputation preserves the relationships between features and creates realistic values based on similar patients. "
    "Simple mean/mode imputation would make all patients look more similar than they really are, "
    "which could mislead any predictions or conclusions we make from this data.",
    icon=":material/info:",
)
