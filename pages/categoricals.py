import streamlit as st
from analysis import stacked_categorical

# Get data from session state
dfv = st.session_state.dfv

st.title("Categories")
st.markdown(
    "*Exploratory Data Analysis — Analyzing Categorical Feature Distributions by Origin*"
)

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
    )

    st.subheader("Categorical Variable Analysis")
    st.markdown(
        "Examine how categorical features like chest pain type, ECG results, and thalassemia status "
        "vary across different medical institutions. These distributions reveal differences in patient populations "
        "and diagnostic patterns."
    )

    if "origin" in dfv.columns:
        chart = stacked_categorical(dfv, cat_var)
        st.altair_chart(chart, use_container_width=True)

        st.divider()

        st.info(
            "**Why this matters:** Categorical feature distributions vary significantly across origins, reflecting differences in patient populations, "
            "referral patterns, and diagnostic practices. For example, one hospital may see more severe cases (referral bias), while another serves "
            "a general population. These distributional shifts affect how we should interpret results and build predictive models. "
            "Ignoring these origin-specific patterns could lead to models that work well for one population but fail for others.",
            icon=":material/info:",
        )
else:
    st.info("No label columns found.")
