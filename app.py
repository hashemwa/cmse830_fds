# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Heart EDA — One Size ≠ Fits All", layout="wide")

# ---------------------- Header & Docs ----------------------
st.title("Heart Dataset — Interactive EDA (Combined Sources)")

with st.expander("About this app / Methods"):
    st.markdown("""
**Narrative:** *One size fits all is false.* Using only **one** dataset (e.g., Cleveland) can bias
conclusions. This app explores a **combined** dataset from **Cleveland, Hungary, Long Beach VA,
Switzerland** to show how **prevalence** and **distributions** differ by **origin**.

**What’s in this dataset (`combined_clean.csv`):**
- All 4 sources stacked together with an **`origin`** column.
- **Missing values imputed** (you chose KNN in your notebook).
- **Code normalization**: binary clamps, ordinal range clips, `thal ∈ {3,6,7}`, `age` as whole years.
- **Readable labels**: `cp_label`, `restecg_label`, `slope_label`, `thal_label`, `num_label`.
- **`target`** = `(num > 0)`.

**How to use:** Filter by **origin(s)** and **age**; choose which variable to explore in distributions; 
examine **prevalence** and **relationships** that differ across sources.
""")


# ---------------------- Load Data ----------------------
@st.cache_data
def load_data(path="combined_clean.csv"):
    return pd.read_csv(path)


try:
    df = load_data()
except Exception as e:
    st.error(
        f"Couldn't load 'combined_clean.csv'. Please export your cleaned df_final to this path. Error: {e}"
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

# ---------------------- Sidebar Filters (interactions) ----------------------
with st.sidebar:
    st.header("Filters")
    # Origin multi-select
    origin_opts = (
        sorted(df["origin"].dropna().unique()) if "origin" in df.columns else []
    )
    origin_sel = st.multiselect("Origins", options=origin_opts, default=origin_opts)

    # Age slider
    if "age" in df.columns and df["age"].notna().any():
        a_min, a_max = int(np.nanmin(df["age"])), int(np.nanmax(df["age"]))
        age_range = st.slider("Age range", a_min, a_max, (a_min, a_max))
    else:
        age_range = None

    # Variable selector for distribution
    dist_candidates = [
        c for c in ["trestbps", "chol", "thalach", "oldpeak"] if c in df.columns
    ]
    dist_var = st.selectbox(
        "Distribution variable",
        options=dist_candidates,
        index=0 if dist_candidates else None,
    )

# Apply filters
mask = pd.Series(True, index=df.index)
if origin_sel and "origin" in df.columns:
    mask &= df["origin"].isin(origin_sel)
if age_range and "age" in df.columns:
    mask &= df["age"].between(*age_range)
dfv = df.loc[mask].copy()

# ---------------------- Top Metrics & Narrative Anchor ----------------------
c1, c2, c3 = st.columns(3)
c1.metric("Rows (filtered)", f"{len(dfv):,}")
if "target" in dfv.columns and dfv["target"].notna().any():
    c2.metric("Prevalence (target=1)", f"{100 * dfv['target'].mean():.1f}%")
if "origin" in dfv.columns and dfv["origin"].notna().any() and "target" in dfv.columns:
    by_origin = dfv.groupby("origin")["target"].mean().mul(100)
    if len(by_origin):
        c3.metric(
            "Prevalence range by origin",
            f"{by_origin.min():.1f}%–{by_origin.max():.1f}%",
        )

st.markdown(
    "> **Key idea:** Origins differ in **prevalence** and **feature distributions**. "
    "A model trained on a single source (e.g., **Cleveland** only) may not generalize."
)

# ---------------------- Tabs ----------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Distributions", "Prevalence", "Relationships", "Data & Download"]
)

# ---------- Tab 1: Distributions (useful, origin-aware) ----------
with tab1:
    st.subheader(f"Distribution of {dist_var} by Origin")
    if dist_var in dfv.columns:
        if "origin" in dfv.columns:
            base = alt.Chart(dfv).transform_density(
                dist_var, as_=[dist_var, "density"], groupby=["origin"]
            )
            dens = base.mark_area(opacity=0.35).encode(
                x=alt.X(f"{dist_var}:Q", title=dist_var),
                y="density:Q",
                color=alt.Color("origin:N", legend=alt.Legend(title="Origin")),
            )
            # Median rules by origin
            med = (
                dfv.groupby("origin")[dist_var]
                .median()
                .reset_index()
                .rename(columns={dist_var: "median"})
            )
            med_lines = (
                alt.Chart(med)
                .mark_rule(size=2)
                .encode(x="median:Q", color="origin:N", tooltip=["origin", "median"])
            )
            st.altair_chart(
                (dens + med_lines).properties(height=360), use_container_width=True
            )
            st.caption(
                "Medians and shapes differ across origins → evidence against a one-source view."
            )
            st.dataframe(dfv.groupby("origin")[dist_var].describe().round(2))
        else:
            # Fallback single-density
            dens = (
                alt.Chart(dfv)
                .transform_density(dist_var, as_=[dist_var, "density"])
                .mark_area(opacity=0.35)
                .encode(x=alt.X(f"{dist_var}:Q", title=dist_var), y="density:Q")
            )
            st.altair_chart(dens.properties(height=360), use_container_width=True)
    else:
        st.info(
            "Pick a variable available in the dataset for distribution visualization."
        )

# ---------- Tab 2: Prevalence (target differences across origins) ----------
with tab2:
    st.subheader("Heart Disease Prevalence (target=1) by Origin")
    if {"origin", "target"}.issubset(dfv.columns) and dfv["target"].notna().any():
        prev = dfv.groupby("origin")["target"].agg(["mean", "count"]).reset_index()
        prev["Prevalence (%)"] = (100 * prev["mean"]).round(1)
        prev = prev.rename(columns={"origin": "Origin", "count": "n"})
        bar = (
            alt.Chart(prev)
            .mark_bar()
            .encode(
                x=alt.X("Origin:N", title="Origin"),
                y=alt.Y("Prevalence (%):Q", title="Prevalence (%)"),
                tooltip=["Origin", "n", "Prevalence (%)"],
            )
        )
        st.altair_chart(bar.properties(height=360), use_container_width=True)
        st.caption(
            "Different base rates across sources → a single-source model mis-estimates risk elsewhere."
        )
    else:
        st.info("`origin` and/or `target` not available for prevalence chart.")

# ---------- Tab 3: Relationships (age ↔ physiology, colored by target) ----------
with tab3:
    st.subheader("Max Heart Rate vs Age (colored by target) + per-origin trends")
    needed_cols = {"age", "thalach", "target", "origin"}
    if needed_cols.issubset(dfv.columns):
        scat = (
            alt.Chart(dfv)
            .mark_circle(size=42, opacity=0.35)
            .encode(
                x=alt.X("age:Q", title="Age (years)"),
                y=alt.Y("thalach:Q", title="Max Heart Rate (thalach)"),
                color=alt.Color("target:N", legend=alt.Legend(title="Target (0/1)")),
                tooltip=[
                    "origin",
                    "age",
                    "thalach",
                    "trestbps",
                    "chol",
                    "oldpeak",
                    "target",
                ],
            )
            .interactive()
        )

        # Per-origin loess trend lines (helps see relationship differences)
        trend = (
            alt.Chart(dfv)
            .transform_loess("age", "thalach", groupby=["origin"])
            .mark_line()
            .encode(x="age:Q", y="thalach:Q", color="origin:N")
        )

        st.altair_chart(
            (scat & trend).resolve_scale(color="independent").properties(height=380),
            use_container_width=True,
        )
        st.caption(
            "Slope/curvature varies by origin → relationships are not universal."
        )
    else:
        st.info("Need columns: age, thalach, target, origin.")

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
    if show_cols:
        st.dataframe(dfv[show_cols].head(500))
    else:
        st.dataframe(dfv.head(500))

    st.download_button(
        "⬇️ Download filtered data (CSV)",
        dfv.to_csv(index=False),
        file_name="filtered.csv",
        mime="text/csv",
    )
