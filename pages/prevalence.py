import streamlit as st

# Import analysis functions as needed
from analysis import (
    prevalence_bar,
)

# Get data from session state
dfv = st.session_state.dfv
df_raw_filtered = st.session_state.df_raw_filtered
df = st.session_state.df
origin_sel = st.session_state.origin_sel
age_range = st.session_state.age_range


st.title("Prevalence")
st.markdown("*Heart Disease Rates Across Origins*")

st.subheader("Heart Disease Prevalence by Origin")
st.caption(
    "Disease prevalence (the proportion of patients with heart disease) varies significantly across institutions. "
    "Specialty cardiac centers see more sick patients than general hospitals do."
)

if {"origin", "target"}.issubset(dfv.columns) and dfv["target"].notna().any():
    bar = prevalence_bar(dfv)
    st.altair_chart(bar, use_container_width=True)

    st.divider()

    st.info(
        "**Why this matters:** The percentage of patients with heart disease varies dramatically between hospitals (this is called the 'base rate'). "
        "A model trained at one hospital will consistently get predictions wrong at another hospital because it expects a different disease rate. "
        "For example, if you train a model at a specialty clinic where 54% have heart disease, it will predict way too many cases "
        "when used at a general hospital where only 20% have the disease. This is a fundamental problem in medical AI. "
        "We either need separate models for each hospital type, or we need to adjust predictions based on the local disease rate.",
        icon=":material/info:",
    )
else:
    st.info("`origin` and/or `target` not available for prevalence chart.")
