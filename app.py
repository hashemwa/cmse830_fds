# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from analysis import (
    get_clean_data,
    kde_by_origin,
    thalach_vs_age_trend,
    stacked_categorical,
    prevalence_bar,
)

# Allow large datasets
alt.data_transformers.disable_max_rows()

st.set_page_config(page_title="Heart EDA — One Size ≠ Fits All", layout="wide")

# ---------------------- Header & Docs ----------------------
st.title("Heart Dataset — Interactive EDA (Combined Sources)")

with st.expander("About this app / Methods"):
    st.markdown("""
**Narrative:** *One size fits all is false.* Using only **one** dataset (e.g., Cleveland) can bias
conclusions. This app explores a **combined** dataset from **Cleveland, Hungary, Long Beach VA,
Switzerland** to show how **prevalence** and **distributions** differ by **origin**.

**What’s in `combined_clean.csv`:**
- All 4 sources stacked with an **`origin`** column
- **Imputed** (KNN in your notebook) + **code normalization** (`thal ∈ {3,6,7}`, binary/ordinal clamps, `age` integer)
- Label columns: `cp_label`, `restecg_label`, `slope_label`, `thal_label`, `num_label`
- **`target`** = `(num > 0)`

**How to use:** Filter by **origin(s)** and **age**. Explore **distributions**, **prevalence**, **relationships**, and
**categorical mixes** to see how sites differ.
""")


# ---------------------- Load Data ----------------------
@st.cache_data
def load_data(path="combined_clean.csv"):
    """Load data from CSV (fallback) or use analysis.py's get_clean_data()"""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return get_clean_data(k=5)


try:
    df = load_data()
except Exception as e:
    st.error(
        f"Couldn't load 'combined_clean.csv'. Export your cleaned df_final to this path. Error: {e}"
    )
    st.stop()

# Ensure expected columns exist (labels may or may not be present; handle gracefully)
expected_num = ["age", "trestbps", "chol", "thalach", "oldpeak"]
needed = {"origin", "target", "num"}.union(expected_num)
missing = [c for c in needed if c not in df.columns]
if missing:
    st.warning(
        f"Missing expected columns in CSV: {missing}. The app will still run with available columns."
    )

# Stable origin ordering for cleaner charts
if "origin" in df.columns:
    origin_order = sorted(df["origin"].dropna().unique().tolist())
    df["origin"] = pd.Categorical(df["origin"], categories=origin_order, ordered=True)

# ---------------------- Sidebar Filters ----------------------
with st.sidebar:
    st.header("Filters")

    origin_opts = (
        sorted(df["origin"].dropna().unique()) if "origin" in df.columns else []
    )

    # Create checkboxes for each origin in a single column
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

# Keep categorical dtype after filtering
if "origin" in dfv.columns:
    dfv["origin"] = pd.Categorical(dfv["origin"], categories=origin_order, ordered=True)

# ---------------------- Top Metrics & Narrative Anchor ----------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (filtered)", f"{len(dfv):,}")
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

st.markdown(
    "> **Key idea:** Origins differ in *distributions*, **relationships**, **categorical mix**, and **prevalence**. "
    "A model trained on a single source (e.g., **Cleveland** only) may not generalize."
)


# ---------------------- Tabs ----------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Distributions", "Relationships", "Categoricals", "Data & Download", "Prevalence"]
)


# ---------- Tab 1: Distributions ----------
with tab1:
    # Mapping of variable names to descriptive labels
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
            key="dist_var_selector",
        )
        title_suffix = f" {units.get(dist_var, '')}"
        var_display_name = var_labels.get(dist_var, dist_var)
        st.subheader(f"Distribution of {var_display_name} by Origin")

        if "origin" in dfv.columns:
            # Display medians in a nicer format
            medians = dfv.groupby("origin")[dist_var].median().round(2)
            cols = st.columns(len(medians))
            for idx, (origin, median_val) in enumerate(medians.items()):
                cols[idx].metric(
                    f"{origin}", f"{median_val:.1f}", delta="median", delta_color="off"
                )

            # Use the plotting function from analysis.py
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
            # Fallback single distribution
            data = dfv[dist_var].dropna()
            if len(data) > 1:
                x_min = data.min()
                x_max = data.max()

                if x_max - x_min < 1e-10:
                    st.warning(
                        f"All values for {dist_var} are approximately the same. Cannot show distribution."
                    )
                else:
                    bins = np.linspace(x_min, x_max, 50)
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    counts, _ = np.histogram(data, bins=bins, density=True)

                    chart_df = pd.DataFrame({dist_var: counts}, index=bin_centers)
                    x_axis_label = (
                        f"{var_display_name} {units.get(dist_var, '')}".strip()
                    )
                    chart_df.index.name = x_axis_label

                    st.markdown(f"**Y-axis:** Density | **X-axis:** {x_axis_label}")

                    st.area_chart(chart_df, height=400, use_container_width=True)
    else:
        st.info(
            "No numeric variables found for distribution (trestbps/chol/thalach/oldpeak)."
        )


# ---------- Tab 5: Prevalence (target differences across origins) ----------
with tab5:
    st.subheader("Heart Disease Prevalence by Origin")
    if {"origin", "target"}.issubset(dfv.columns) and dfv["target"].notna().any():
        # Use the plotting function from analysis.py
        bar = prevalence_bar(dfv)
        st.altair_chart(bar, use_container_width=True)
        st.caption(
            "Different base rates across sources → a single-source model mis-estimates risk elsewhere."
        )
    else:
        st.info("`origin` and/or `target` not available for prevalence chart.")

# ---------- Tab 2: Relationships (age ↔ physiology, colored by target) ----------
with tab2:
    st.subheader("Max Heart Rate vs Age by Origin")
    needed_cols = {"age", "thalach", "target", "origin"}
    if needed_cols.issubset(dfv.columns):
        # Use the plotting function from analysis.py
        chart = thalach_vs_age_trend(dfv)
        st.altair_chart(chart, use_container_width=True)

        st.caption(
            "Slope/curvature varies by origin → relationships are not universal."
        )
    else:
        st.info("Need columns: age, thalach, target, origin.")

# ---------- Tab 3: Categoricals ----------
with tab3:
    st.subheader("Categorical feature mix by origin")
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
            key="cat_var",
        )

        if "origin" in dfv.columns:
            # Use the plotting function from analysis.py
            chart = stacked_categorical(dfv, cat_var)
            st.altair_chart(chart, use_container_width=True)
            st.caption(
                "Legend and stacked segment order follow the intended clinical sequence."
            )
        else:
            st.info("Need 'origin' for categorical comparison.")
    else:
        st.info("No label columns found (e.g., cp_label, thal_label).")


# ---------- Tab 4: Data & Download ----------
with tab4:
    st.subheader("Filtered data preview")
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
