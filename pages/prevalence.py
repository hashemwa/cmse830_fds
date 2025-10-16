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
st.markdown("*Exploratory Data Analysis — Heart Disease Rates Across Origins*")

st.subheader("Heart Disease Prevalence by Origin")
st.markdown(
    "Disease prevalence (the proportion of patients with heart disease) varies significantly across institutions. "
    "These differences reflect both the patient populations served and institutional referral patterns."
)

if {"origin", "target"}.issubset(dfv.columns) and dfv["target"].notna().any():
    bar = prevalence_bar(dfv)
    st.altair_chart(bar, use_container_width=True)

    st.divider()

    st.info(
        "**Why this matters:** The dramatically different prevalence rates across origins (base rates) are critical for prediction. "
        "A model trained on one institution's data will systematically over- or under-estimate risk when applied to data from another institution. "
        "For example, a model trained on a high-prevalence specialty clinic will predict too many cases when used in a general population setting. "
        "This is known as the base rate fallacy, and it's why we must either train origin-specific models or explicitly account for these differences "
        "through techniques like calibration or domain adaptation.",
        icon=":material/info:",
    )
else:
    st.info("`origin` and/or `target` not available for prevalence chart.")
