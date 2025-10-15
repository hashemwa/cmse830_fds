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
    kde_by_origin,
    thalach_vs_age_trend,
    stacked_categorical,
    prevalence_bar,
    missingness_heatmap,
)

# Get data from session state
dfv = st.session_state.dfv
df_raw_filtered = st.session_state.df_raw_filtered
df = st.session_state.df
origin_sel = st.session_state.origin_sel
age_range = st.session_state.age_range


st.title("Prevalence")
st.subheader("Heart Disease Prevalence by Origin")

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
