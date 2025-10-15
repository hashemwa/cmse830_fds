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
