import streamlit as st
import pandas as pd
import numpy as np
from analysis import get_clean_data, get_raw_data

st.set_page_config(page_title="Heart Disease Analysis", layout="wide")

about_page = st.Page("pages/about.py", title="About", icon=":material/home:")

ida_overview = st.Page(
    "pages/data_overview.py", title="Data Overview", icon=":material/table_chart:"
)
ida_missing_data = st.Page(
    "pages/missing_data.py",
    title="Missing Data",
    icon=":material/feature_search:",
)

cleaning_imputation = st.Page(
    "pages/imputation.py",
    title="Imputation",
    icon=":material/cleaning_services:",
)
cleaning_encoding = st.Page(
    "pages/encoding.py",
    title="Feature Engineering",
    icon=":material/build:",
)

eda_distributions = st.Page(
    "pages/distributions.py",
    title="Distributions",
    icon=":material/bar_chart:",
)
eda_relationships = st.Page(
    "pages/relationships.py", title="Relationships", icon=":material/scatter_plot:"
)
eda_categoricals = st.Page(
    "pages/categoricals.py",
    title="Categoricals",
    icon=":material/stacked_bar_chart:",
)
eda_prevalence = st.Page(
    "pages/prevalence.py", title="Prevalence", icon=":material/monitoring:"
)

modeling_page = st.Page(
    "pages/modeling.py", title="Model Development", icon=":material/model_training:"
)

export_page = st.Page(
    "pages/data_export.py", title="Download", icon=":material/download:"
)

pg = st.navigation(
    {
        "": [about_page],
        "Initial Data Analysis": [ida_overview, ida_missing_data],
        "Data Cleaning": [cleaning_imputation, cleaning_encoding],
        "Exploratory Data Analysis": [
            eda_distributions,
            eda_relationships,
            eda_categoricals,
            eda_prevalence,
        ],
        "Modeling": [modeling_page],
        "Data Export": [export_page],
    },
    position="top",
)


@st.cache_data
def load_data():
    return get_clean_data(k=5)


@st.cache_data
def load_raw_data():
    return get_raw_data()


if "df" not in st.session_state:
    st.session_state.df = load_data()
if "df_raw" not in st.session_state:
    st.session_state.df_raw = load_raw_data()

df = st.session_state.df
df_raw = st.session_state.df_raw

if "origin" in df.columns:
    origin_order = sorted(df["origin"].dropna().unique().tolist())
    df["origin"] = pd.Categorical(df["origin"], categories=origin_order, ordered=True)
    st.session_state.origin_order = origin_order

with st.sidebar:
    st.markdown("### :material/tune: **Filters**")
    st.markdown("*Customize your view of the data*")
    st.divider()

    st.markdown("#### :material/location_on: Data Origin")
    st.caption("Select institutions to include")

    origin_opts = (
        sorted(df["origin"].dropna().unique()) if "origin" in df.columns else []
    )
    origin_sel = []

    if origin_opts:
        for origin in origin_opts:
            if st.checkbox(origin, value=True, key=f"origin_{origin}"):
                origin_sel.append(origin)

    st.divider()

    st.markdown("#### :material/calendar_today: Age Range")
    st.caption("Filter patients by age")

    if "age" in df.columns and df["age"].notna().any():
        a_min, a_max = int(np.nanmin(df["age"])), int(np.nanmax(df["age"]))
        age_range = st.slider(
            "", a_min, a_max, (a_min, a_max), label_visibility="collapsed"
        )

        col1, col2 = st.columns(2)
        col1.metric("Min Age", age_range[0], delta=None)
        col2.metric("Max Age", age_range[1], delta=None)
    else:
        age_range = None

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

mask_raw = pd.Series(True, index=df_raw.index)
if origin_sel and "origin" in df_raw.columns:
    mask_raw &= df_raw["origin"].isin(origin_sel)
if age_range and "age" in df_raw.columns:
    mask_raw &= df_raw["age"].between(*age_range)
df_raw_filtered = df_raw.loc[mask_raw].copy()

st.session_state.dfv = dfv
st.session_state.df_raw_filtered = df_raw_filtered
st.session_state.origin_sel = origin_sel
st.session_state.age_range = age_range

pg.run()
