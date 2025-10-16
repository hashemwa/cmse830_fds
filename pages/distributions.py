import streamlit as st

# Import analysis functions as needed
from analysis import (
    kde_by_origin,
)

# Get data from session state
dfv = st.session_state.dfv
df_raw_filtered = st.session_state.df_raw_filtered
df = st.session_state.df
origin_sel = st.session_state.origin_sel
age_range = st.session_state.age_range


st.title("Distributions")
st.markdown("*Comparing Feature Distributions Across Origins*")

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
    )

    var_display_name = var_labels.get(dist_var, dist_var)
    st.subheader(f"Distribution of {var_display_name} by Origin")

    if "origin" in dfv.columns:
        medians = dfv.groupby("origin")[dist_var].median().round(2)
        cols = st.columns(len(medians))
        for idx, (origin, median_val) in enumerate(medians.items()):
            cols[idx].metric(
                f"{origin}", f"{median_val:.1f}", delta="median", delta_color="off"
            )

    chart = kde_by_origin(dfv, dist_var)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning(f"Not enough data variation to display distribution for {dist_var}.")

    st.divider()

    st.info(
        "**Why this matters:** Each hospital's patient population looks different in the data. "
        "They served different types of patients, used different equipment, and followed different procedures. "
        "These differences are real and not just noise. Any prediction model needs to account for which hospital "
        "the data came from, or it will make bad predictions when used on patients from a different hospital.",
        icon=":material/info:",
    )
else:
    st.info("No numeric variables found for distribution.")
