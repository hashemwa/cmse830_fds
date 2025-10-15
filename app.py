import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.stats import gaussian_kde
from analysis import (
    get_clean_data,
    get_raw_data,
    get_individual_raw_datasets,
    get_individual_simple_imputed,
    get_individual_knn_imputed,
    get_combined_knn_imputed,
    kde_by_origin,
    thalach_vs_age_trend,
    stacked_categorical,
    prevalence_bar,
    missingness_heatmap,
)

alt.data_transformers.disable_max_rows()

st.set_page_config(page_title="Heart EDA — One Size ≠ Fits All", layout="wide")


# ---------------------- Load Data ----------------------
@st.cache_data
def load_data():
    return get_clean_data(k=5)


@st.cache_data
def load_raw_data():
    return get_raw_data()


df = load_data()
df_raw = load_raw_data()

# Stable origin ordering
if "origin" in df.columns:
    origin_order = sorted(df["origin"].dropna().unique().tolist())
    df["origin"] = pd.Categorical(df["origin"], categories=origin_order, ordered=True)

# ---------------------- Sidebar Navigation & Filters ----------------------
with st.sidebar:
    st.header("Navigation")

    # Main section selector
    section = st.selectbox(
        "Select Section:",
        [
            "Home",
            "Initial Data Analysis",
            "Data Cleaning",
            "Exploratory Data Analysis",
            "Data Export",
        ],
        index=0,
        format_func=lambda x: {
            "Home": "🏠 Home",
            "Initial Data Analysis": "🔍 Initial Data Analysis",
            "Data Cleaning": "🧹 Data Cleaning",
            "Exploratory Data Analysis": "📈 Exploratory Data Analysis",
            "Data Export": "💾 Data Export",
        }[x],
    )

    # Sub-page selector based on section
    if section == "Initial Data Analysis":
        page = st.radio(
            "Choose analysis:",
            ["Data Overview", "Missingness Analysis"],
            label_visibility="visible",
        )
    elif section == "Data Cleaning":
        page = st.radio(
            "Choose analysis:",
            ["Imputation Comparison", "Feature Engineering"],
            label_visibility="visible",
        )
    elif section == "Exploratory Data Analysis":
        page = st.radio(
            "Choose analysis:",
            [
                "Distribution Analysis",
                "Relationships",
                "Categorical Analysis",
                "Prevalence",
            ],
            label_visibility="visible",
        )
    elif section == "Data Export":
        page = "Data Export"
    else:
        page = "Home"

    st.divider()
    st.header("Filters")

    origin_opts = (
        sorted(df["origin"].dropna().unique()) if "origin" in df.columns else []
    )
    origin_sel = []
    for origin in origin_opts:
        if st.checkbox(origin, value=True, key=f"origin_{origin}"):
            origin_sel.append(origin)

    if "age" in df.columns and df["age"].notna().any():
        a_min, a_max = int(np.nanmin(df["age"])), int(np.nanmax(df["age"]))
        age_range = st.slider("Age range", a_min, a_max, (a_min, a_max))
    else:
        age_range = None

# Apply filters
mask = pd.Series(True, index=df.index)
if origin_sel and "origin" in df.columns:
    mask &= df["origin"].isin(origin_sel)
if age_range and "age" in df.columns:
    mask &= df["age"].between(*age_range)
dfv = df.loc[mask].copy()

if dfv.empty:
    st.warning("No rows match the current filters. Expand your selections.")
    st.stop()

if "origin" in dfv.columns:
    dfv["origin"] = pd.Categorical(dfv["origin"], categories=origin_order, ordered=True)

# Apply same filters to raw data
mask_raw = pd.Series(True, index=df_raw.index)
if origin_sel and "origin" in df_raw.columns:
    mask_raw &= df_raw["origin"].isin(origin_sel)
if age_range and "age" in df_raw.columns:
    mask_raw &= df_raw["age"].between(*age_range)
df_raw_filtered = df_raw.loc[mask_raw].copy()

# ---------------------- PAGE CONTENT ----------------------

if page == "Home":
    st.title("Heart Dataset — Interactive EDA (Combined Sources)")
    st.markdown("### *One Size ≠ Fits All*")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{len(dfv):,}")
    if "target" in dfv.columns and dfv["target"].notna().any():
        c2.metric("Overall Prevalence", f"{100 * dfv['target'].mean():.1f}%")
    if {"origin", "target"}.issubset(dfv.columns):
        by_origin = dfv.groupby("origin")["target"].mean().mul(100)
        if len(by_origin):
            c3.metric(
                "Lowest Prevalence",
                f"{by_origin.min():.1f}%",
                delta=by_origin.idxmin(),
                delta_color="off",
            )
            c4.metric(
                "Highest Prevalence",
                f"{by_origin.max():.1f}%",
                delta=by_origin.idxmax(),
                delta_color="off",
            )

    st.markdown("""
    ## Key Findings
    
    > **Origins differ in distributions, relationships, categorical mix, and prevalence.**  
    > A model trained on a single source (e.g., **Cleveland** only) may not generalize.
    
    ### About This Dataset
    
    This app explores a **combined** dataset from four sources:
    - **Cleveland**
    - **Hungary**
    - **Long Beach VA**
    - **Switzerland**
    
    ### Methodology
    
    **Data Processing:**
    - KNN imputation (k=5) for missing values
    - Code normalization (`thal ∈ {3,6,7}`, binary/ordinal clamps, `age` integer)
    - Label columns for interpretability
    - **`target`** = presence of heart disease (`num > 0`)
    
    **Use the sidebar to:**
    - Navigate between analysis pages
    - Filter by origin(s) and age range
    - Explore how distributions and patterns vary across sources
    """)

elif page == "Data Overview":
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
                    cat_summary = (
                        dataset[available_cat].describe(include="all").round(2)
                    )
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
            cat_summary = (
                df_raw_filtered[available_cat].describe(include="all").round(2)
            )
            st.dataframe(cat_summary, use_container_width=True)
        else:
            st.info("No categorical features available")

elif page == "Missingness Analysis":
    st.title("Missingness Analysis")
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

elif page == "Imputation Comparison":
    st.title("Imputation Comparison")
    st.markdown("*Comparing Simple vs KNN Imputation Methods*")

    # Methodology explanation
    st.info(
        "**Methodology:** Each dataset was imputed **independently** to avoid data leakage across origins. "
        "This preserves the unique characteristics of each medical institution's data."
    )

    with st.expander("Why KNN Imputation?", icon=":material/info:"):
        st.markdown("""
        **Simple Imputation** (mean/mode):
        - Replaces missing values with the overall mean (numerical) or mode (categorical)
        - Fast and straightforward
        - ❌ Reduces variance and ignores relationships between features
        
        **KNN Imputation** (k=5):
        - Uses the 5 most similar patients to estimate missing values
        - Preserves local structure and feature relationships
        - ✅ Better maintains the original data distribution
        """)

    # Get imputed datasets
    raw_datasets = get_individual_raw_datasets()
    simple_datasets = get_individual_simple_imputed()
    knn_datasets = get_individual_knn_imputed()

    st.subheader("Individual Dataset Comparison")
    st.markdown("Compare imputation methods for each dataset source.")

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
                    delta=f"{missing_simple - missing_count}",
                )

            with col3:
                st.markdown("**KNN Imputation**")
                missing_knn = knn_data.isnull().sum().sum()
                st.metric(
                    "Missing Values",
                    missing_knn,
                    delta=f"{missing_knn - missing_count}",
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
                    x_smooth = np.linspace(x_min, x_max, 200)

                    chart_data = []

                    # Original KDE
                    try:
                        kde_orig = gaussian_kde(original_vals, bw_method="scott")
                        for x_val in x_smooth:
                            chart_data.append(
                                {
                                    "trestbps (mmHg)": x_val,
                                    "Density": kde_orig(x_val)[0],
                                    "Method": "Original",
                                }
                            )
                    except Exception:
                        pass

                    # Simple KDE
                    try:
                        kde_simple = gaussian_kde(simple_vals, bw_method="scott")
                        for x_val in x_smooth:
                            chart_data.append(
                                {
                                    "trestbps (mmHg)": x_val,
                                    "Density": kde_simple(x_val)[0],
                                    "Method": "Simple Imputation",
                                }
                            )
                    except Exception:
                        pass

                    # KNN KDE
                    try:
                        kde_knn = gaussian_kde(knn_vals, bw_method="scott")
                        for x_val in x_smooth:
                            chart_data.append(
                                {
                                    "trestbps (mmHg)": x_val,
                                    "Density": kde_knn(x_val)[0],
                                    "Method": "KNN Imputation",
                                }
                            )
                    except Exception:
                        pass

                    if chart_data:
                        chart_df = pd.DataFrame(chart_data)

                        # Use line chart for better visibility of overlapping distributions
                        chart = (
                            alt.Chart(chart_df)
                            .mark_line(size=3, strokeDash=[0, 0])
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
                                    alt.Tooltip(
                                        "Density:Q", title="Density", format=".4f"
                                    ),
                                ],
                            )
                            .properties(height=400)
                        )

                        st.altair_chart(chart, use_container_width=True)
                        st.caption(
                            "KNN imputation preserves the original distribution shape better than simple mean imputation."
                        )

    st.divider()

    # Combined dataset comparison
    st.subheader("Combined Dataset: Before vs After KNN Imputation")
    st.markdown(
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
                st.dataframe(
                    combined_raw[available_cat].describe(include="all").round(2),
                    use_container_width=True,
                )
            else:
                st.info("No categorical features available")

        with col2:
            st.markdown("**After KNN Imputation**")
            if available_cat:
                st.dataframe(
                    combined_knn[available_cat].describe(include="all").round(2),
                    use_container_width=True,
                )
            else:
                st.info("No categorical features available")

    st.caption(
        "✅ **Result:** KNN imputation successfully filled all missing values while preserving "
        "the statistical properties and distributions of the original data."
    )

elif page == "Feature Engineering":
    st.title("Feature Engineering")
    st.markdown(
        "*Data Cleaning — Creating Human-Readable Labels for Categorical Variables*"
    )

    # Encoding mappings
    st.subheader("Categorical Variable Mappings")
    st.write(
        "Numeric codes were mapped to descriptive labels to improve interpretability:"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Chest Pain", "Resting ECG", "ST Slope", "Thalassemia", "Diagnosis"]
    )

    with tab1:
        cp_mapping = pd.DataFrame(
            {
                "Code": [1, 2, 3, 4],
                "Label": [
                    "typical angina",
                    "atypical angina",
                    "non-anginal pain",
                    "asymptomatic",
                ],
            }
        )
        st.dataframe(cp_mapping, use_container_width=True, hide_index=True)

    with tab2:
        restecg_mapping = pd.DataFrame(
            {
                "Code": [0, 1, 2],
                "Label": [
                    "normal",
                    "ST–T wave abnormality (>0.05 mV)",
                    "LVH by Estes",
                ],
            }
        )
        st.dataframe(restecg_mapping, use_container_width=True, hide_index=True)

    with tab3:
        slope_mapping = pd.DataFrame(
            {
                "Code": [1, 2, 3],
                "Label": ["upsloping", "flat", "downsloping"],
            }
        )
        st.dataframe(slope_mapping, use_container_width=True, hide_index=True)

    with tab4:
        thal_mapping = pd.DataFrame(
            {
                "Code": [3, 6, 7],
                "Label": ["normal", "fixed defect", "reversible defect"],
            }
        )
        st.dataframe(thal_mapping, use_container_width=True, hide_index=True)

    with tab5:
        num_mapping = pd.DataFrame(
            {
                "Code": [0, 1, 2, 3, 4],
                "Label": [
                    "no heart disease",
                    "mild heart disease",
                    "moderate heart disease",
                    "severe heart disease",
                    "critical heart disease",
                ],
            }
        )
        st.dataframe(num_mapping, use_container_width=True, hide_index=True)

    # Before/After comparison
    st.subheader("Before & After Sample")

    label_cols = ["cp_label", "restecg_label", "slope_label", "thal_label", "num_label"]
    original_cols = ["cp", "restecg", "slope", "thal", "num"]

    if all(col in df.columns for col in label_cols):
        sample_df = df[original_cols + label_cols].head(5)

        col_before, col_after = st.columns(2)

        with col_before:
            st.markdown("**Original (Numeric)**")
            st.dataframe(
                sample_df[original_cols],
                use_container_width=True,
            )

        with col_after:
            st.markdown("**Transformed (Labels)**")
            st.dataframe(
                sample_df[label_cols],
                use_container_width=True,
            )

    st.caption(
        "✅ **Result:** Categorical variables now have interpretable labels while retaining original numeric codes for modeling."
    )

elif page == "Distribution Analysis":
    st.title("Distribution Analysis")

    var_labels = {
        "trestbps": "Resting Blood Pressure",
        "chol": "Serum Cholesterol",
        "thalach": "Maximum Heart Rate",
        "oldpeak": "ST Depression",
        "age": "Age",
    }
    units = {"trestbps": "(mmHg)", "chol": "(mg/dL)", "thalach": "(bpm)", "oldpeak": ""}

    dist_candidates = [
        c for c in ["trestbps", "chol", "thalach", "oldpeak"] if c in dfv.columns
    ]

    if dist_candidates:
        dist_var = st.selectbox(
            "Choose a numerical variable:",
            options=dist_candidates,
            index=0,
        )

        var_display_name = var_labels.get(dist_var, dist_var)
        st.subheader(f"Distribution of {var_display_name} by Origin")

        if "origin" in dfv.columns:
            medians = dfv.groupby("origin")[dist_var].median().round(2)
            cols = st.columns(len(medians))
            for idx, (origin, median_val) in enumerate(medians.items()):
                cols[idx].metric(
                    f"{origin}", f"{median_val:.1f}", delta="median", delta_color="off"
                )

            chart = kde_by_origin(dfv, dist_var)
            if chart is not None:
                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning(
                    f"Not enough data variation to display distribution for {dist_var}."
                )

            st.caption(
                "Medians and shapes differ across origins → evidence against a one-source view."
            )
    else:
        st.info("No numeric variables found for distribution.")

elif page == "Relationships":
    st.title("Relationship Analysis")
    st.subheader("Max Heart Rate vs Age by Origin")

    needed_cols = {"age", "thalach", "target", "origin"}
    if needed_cols.issubset(dfv.columns):
        chart = thalach_vs_age_trend(dfv)
        st.altair_chart(chart, use_container_width=True)

        st.caption(
            "Slope/curvature varies by origin → relationships are not universal. "
            "The LOESS smoothing shows different age-related heart rate patterns."
        )
    else:
        st.info("Need columns: age, thalach, target, origin.")

elif page == "Categorical Analysis":
    st.title("Categorical Variable Analysis")
    st.subheader("Categorical Mix by Origin")

    cat_candidates = [
        c
        for c in ["cp_label", "restecg_label", "slope_label", "thal_label", "num_label"]
        if c in dfv.columns
    ]

    if cat_candidates:
        cat_var = st.selectbox(
            "Choose a categorical variable:",
            options=cat_candidates,
            index=0,
        )

        if "origin" in dfv.columns:
            chart = stacked_categorical(dfv, cat_var)
            st.altair_chart(chart, use_container_width=True)
            st.caption(
                "Legend and stacked segment order follow the intended clinical sequence. "
                "Different origins show different distributions of categorical features."
            )
    else:
        st.info("No label columns found.")

elif page == "Prevalence":
    st.title("Heart Disease Prevalence Analysis")
    st.subheader("Prevalence by Origin")

    if {"origin", "target"}.issubset(dfv.columns) and dfv["target"].notna().any():
        bar = prevalence_bar(dfv)
        st.altair_chart(bar, use_container_width=True)

        st.caption(
            "Different base rates across sources → a single-source model mis-estimates risk elsewhere."
        )

        st.subheader("Prevalence Statistics")
        prev_stats = dfv.groupby("origin")["target"].agg(
            [
                ("Count", "count"),
                ("Cases", "sum"),
                ("Prevalence %", lambda x: f"{100 * x.mean():.1f}%"),
            ]
        )
        st.dataframe(prev_stats)
    else:
        st.info("`origin` and/or `target` not available for prevalence chart.")

elif page == "Data Export":
    st.title("Data Export")
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

    st.dataframe(dfv[show_cols].head(500) if show_cols else dfv.head(500))

    st.download_button(
        "Download filtered data (CSV)",
        dfv.to_csv(index=False),
        file_name="filtered.csv",
        mime="text/csv",
        icon=":material/download:",
    )

    st.subheader("Export Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(dfv))
    col2.metric("Columns", len(dfv.columns))
    col3.metric("Size", f"{dfv.memory_usage(deep=True).sum() / 1024:.1f} KB")
