# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

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
    return pd.read_csv(path)


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
    origin_sel = st.multiselect("Origins", options=origin_opts, default=origin_opts)

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
c1, c2, c3 = st.columns(3)
c1.metric("Rows (filtered)", f"{len(dfv):,}")
if "target" in dfv.columns and dfv["target"].notna().any():
    c2.metric("Prevalence (target=1)", f"{100 * dfv['target'].mean():.1f}%")
if {"origin", "target"}.issubset(dfv.columns):
    by_origin = dfv.groupby("origin")["target"].mean().mul(100)
    if len(by_origin):
        c3.metric(
            "Prevalence range by origin",
            f"{by_origin.min():.1f}%–{by_origin.max():.1f}%",
        )

st.markdown(
    "> **Key idea:** Origins differ in **prevalence**, **distributions**, **relationships**, and **categorical mix**. "
    "A model trained on a single source (e.g., **Cleveland** only) may not generalize."
)


# ---------------------- Tabs ----------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Distributions", "Relationships", "Categoricals", "Data & Download", "Prevalence"]
)

# Desired category orders
ORDERS = {
    "cp_label": [
        "typical angina",
        "atypical angina",
        "non-anginal pain",
        "asymptomatic",
    ],
    "restecg_label": ["normal", "ST–T wave abnormality (>0.05 mV)", "LVH by Estes"],
    "slope_label": ["upsloping", "flat", "downsloping"],
    "thal_label": ["normal", "fixed defect", "reversible defect"],
    "num_label": [
        "no heart disease",
        "mild heart disease",
        "moderate heart disease",
        "severe heart disease",
        "critical heart disease",
    ],
}


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
            "Select variable to visualize:",
            options=dist_candidates,
            index=0,
            key="dist_var_selector",
        )
        title_suffix = f" {units.get(dist_var, '')}"
        var_display_name = var_labels.get(dist_var, dist_var)
        st.subheader(f"Distribution of {var_display_name}{title_suffix} by Origin")

        if "origin" in dfv.columns:
            # Display medians in a nicer format
            medians = dfv.groupby("origin")[dist_var].median().round(2)
            cols = st.columns(len(medians))
            for idx, (origin, median_val) in enumerate(medians.items()):
                cols[idx].metric(f"{origin}", f"{median_val:.1f}")

            # Create smooth density curves for each origin using KDE
            from scipy.stats import gaussian_kde

            density_data = {}
            x_min = dfv[dist_var].min()
            x_max = dfv[dist_var].max()

            # Handle edge case where all values are the same
            if x_max - x_min < 1e-10:
                st.warning(
                    f"All values for {dist_var} are approximately the same. Cannot show distribution."
                )
            else:
                # Create 200 smooth points for x-axis
                x_smooth = np.linspace(x_min, x_max, 200)

                for origin in sorted(dfv["origin"].unique()):
                    origin_data = dfv[dfv["origin"] == origin][dist_var].dropna().values
                    if len(origin_data) > 1 and origin_data.std() > 1e-10:
                        try:
                            # Use gaussian KDE for smooth density curves
                            kde = gaussian_kde(origin_data, bw_method="scott")
                            density = kde(x_smooth)
                            density_data[origin] = density
                        except (np.linalg.LinAlgError, ValueError):
                            # Fallback to interpolated histogram if KDE fails
                            bins = np.linspace(origin_data.min(), origin_data.max(), 50)
                            counts, _ = np.histogram(
                                origin_data, bins=bins, density=True
                            )
                            bin_centers = (bins[:-1] + bins[1:]) / 2
                            density_data[origin] = np.interp(
                                x_smooth, bin_centers, counts
                            )

                # Only show chart if we have data
                if density_data:
                    # Create DataFrame for Altair chart
                    x_axis_label = (
                        f"{var_display_name} {units.get(dist_var, '')}".strip()
                    )

                    # Reshape data for Altair (long format)
                    chart_data = []
                    for origin, density in density_data.items():
                        for x_val, y_val in zip(x_smooth, density):
                            chart_data.append(
                                {
                                    x_axis_label: x_val,
                                    "Density": y_val,
                                    "Origin": origin,
                                }
                            )
                    chart_df = pd.DataFrame(chart_data)

                    # Get dynamic color mapping based on available origins
                    # Using Tableau10 colors to match the line chart
                    color_map = {
                        "Cleveland": "#4c78a8",
                        "Hungary": "#f58518",
                        "Long Beach VA": "#54a24b",
                        "Switzerland": "#e45756",
                    }

                    # Create Altair area chart
                    area_chart = (
                        alt.Chart(chart_df)
                        .mark_area(opacity=0.6)
                        .encode(
                            x=alt.X(
                                f"{x_axis_label}:Q",
                                title=x_axis_label,
                                axis=alt.Axis(grid=False),
                            ),
                            y=alt.Y(
                                "Density:Q",
                                title="Density",
                                axis=alt.Axis(gridOpacity=0.5),
                            ),
                            color=alt.Color(
                                "Origin:N",
                                legend=alt.Legend(title="Origin", orient="right"),
                                scale=alt.Scale(
                                    domain=list(color_map.keys()),
                                    range=list(color_map.values()),
                                ),
                            ),
                        )
                        .properties(height=400)
                    )

                    st.altair_chart(area_chart, use_container_width=True)
                else:
                    st.warning(
                        f"Not enough data to display distribution for {dist_var}."
                    )

            st.caption(
                "Medians and shapes differ across origins → evidence against a one-source view."
            )
            st.dataframe(dfv.groupby("origin")[dist_var].describe().round(2))
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
    st.subheader("Heart Disease Prevalence (target=1) by Origin")
    if {"origin", "target"}.issubset(dfv.columns) and dfv["target"].notna().any():
        prev = dfv.groupby("origin")["target"].agg(["mean", "count"]).reset_index()
        prev["Prevalence (%)"] = (100 * prev["mean"]).round(1)
        prev = prev.rename(columns={"origin": "Origin", "count": "n"})
        bar = (
            alt.Chart(prev)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Origin:N",
                    title="Origin",
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y(
                    "Prevalence (%):Q",
                    title="Prevalence (%)",
                    axis=alt.Axis(gridOpacity=0.5),
                ),
                tooltip=["Origin", "n", "Prevalence (%)"],
            )
        )
        st.altair_chart(bar.properties(height=450), use_container_width=True)
        st.caption(
            "Different base rates across sources → a single-source model mis-estimates risk elsewhere."
        )
    else:
        st.info("`origin` and/or `target` not available for prevalence chart.")

# ---------- Tab 2: Relationships (age ↔ physiology, colored by target) ----------
with tab2:
    st.subheader("Max Heart Rate vs Age (colored by target) + per-origin trends")
    needed_cols = {"age", "thalach", "target", "origin"}
    if needed_cols.issubset(dfv.columns):
        # Create a selection for multi-line tooltip
        hover = alt.selection_point(
            fields=["age"],
            nearest=True,
            on="mouseover",
            empty=False,
        )

        # Color mapping to match the distribution chart
        color_map = {
            "Cleveland": "#4c78a8",
            "Hungary": "#f58518",
            "Long Beach VA": "#54a24b",
            "Switzerland": "#e45756",
        }

        # Per-origin loess trend lines
        lines = (
            alt.Chart(dfv)
            .transform_loess("age", "thalach", groupby=["origin"])
            .mark_line(size=3)
            .encode(
                x=alt.X("age:Q", title="Age (years)", axis=alt.Axis(grid=False)),
                y=alt.Y(
                    "thalach:Q",
                    title="Max Heart Rate (thalach)",
                    axis=alt.Axis(gridOpacity=0.5),
                ),
                color=alt.Color(
                    "origin:N",
                    legend=alt.Legend(title="Origin"),
                    scale=alt.Scale(
                        domain=list(color_map.keys()), range=list(color_map.values())
                    ),
                ),
            )
        )

        # Points that appear on hover
        points = (
            lines.mark_point(size=100, filled=True)
            .encode(
                opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                tooltip=[
                    alt.Tooltip("age:Q", title="Age", format=".0f"),
                    alt.Tooltip("origin:N", title="Origin"),
                    alt.Tooltip("thalach:Q", title="Max Heart Rate", format=".1f"),
                ],
            )
            .add_params(hover)
        )

        # Vertical rule to show where you're hovering
        rule = (
            alt.Chart(dfv)
            .transform_loess("age", "thalach", groupby=["origin"])
            .mark_rule(color="gray", opacity=0.5)
            .encode(
                x="age:Q",
            )
            .transform_filter(hover)
        )

        # Combine all layers
        chart = (lines + points + rule).properties(height=450)
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
            # Enforce order in the dataframe (helps tooltips/tables)
            if cat_var in ORDERS:
                dfv[cat_var] = pd.Categorical(
                    dfv[cat_var], categories=ORDERS[cat_var], ordered=True
                )

            cat_df = dfv.groupby(["origin", cat_var]).size().reset_index(name="n")
            cat_df["pct"] = cat_df.groupby("origin")["n"].transform(
                lambda x: 100 * x / x.sum()
            )

            # Optional: a numeric rank for bullet-proof stack ordering
            if cat_var in ORDERS:
                cat_df["cat_rank"] = cat_df[
                    cat_var
                ].cat.codes  # 0..k-1 in your specified order

            chart = (
                alt.Chart(cat_df)
                .mark_bar()
                .encode(
                    x=alt.X("origin:N", title="Origin", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y(
                        "pct:Q",
                        title="Percent within origin",
                        stack="normalize",
                        axis=alt.Axis(gridOpacity=0.5),
                    ),
                    # Fixed domain controls legend order AND stack order
                    color=alt.Color(
                        f"{cat_var}:N",
                        legend=alt.Legend(
                            title=cat_var.replace("_", " "), symbolType="circle"
                        ),
                        scale=alt.Scale(domain=ORDERS.get(cat_var)),  # <- key line
                    ),
                    # Extra safety: force stack draw order using the rank
                    order=alt.Order("cat_rank:Q")
                    if "cat_rank" in cat_df.columns
                    else alt.Undefined,
                    tooltip=[
                        "origin",
                        alt.Tooltip(f"{cat_var}:N", title=cat_var.replace("_", " ")),
                        alt.Tooltip("n:Q", title="Count"),
                        alt.Tooltip("pct:Q", title="Percent", format=".1f"),
                    ],
                )
                .properties(height=400)
            )
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
    )
