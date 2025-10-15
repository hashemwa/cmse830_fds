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


st.title("Relationships")
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
