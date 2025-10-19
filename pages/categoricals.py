import streamlit as st
from analysis import stacked_categorical

dfv = st.session_state.dfv

st.title("Categories")
st.markdown("*Analyzing Categorical Features by Origin*")

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
    st.caption(
        "Examine how categorical features like chest pain type, ECG results, and thalassemia status "
        "vary across different medical institutions. These bar plots reveal differences in patient populations "
        "and diagnostic patterns."
    )

    if "origin" in dfv.columns:
        chart = stacked_categorical(dfv, cat_var)
        st.altair_chart(chart, use_container_width=True)

        st.divider()

        st.info(
            "**Why this matters:** The types of patients each hospital sees are quite different. "
            "For example, one hospital might be a specialty clinic that mostly sees severe cases (referral bias), "
            "while another treats general walk-in patients. These differences affect what the data tells us. "
            "A model trained on data from a specialty clinic will expect sicker patients and might give false positives "
            "when used in a general practice setting.",
            icon=":material/info:",
        )
        st.warning(
            "**Important note:** Features like `slope` and `thal` had extremely high missing rates (50-90% in some hospitals). "
            "The patterns you see for these variables are based on very few patients and may not represent the full population. ",
            icon=":material/warning:",
        )
else:
    st.info("No label columns found.")
