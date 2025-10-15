import streamlit as st

# Import analysis functions as needed
from analysis import (
    thalach_vs_age_trend,
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
