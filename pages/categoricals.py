import streamlit as st
from analysis import stacked_categorical

# Get data from session state
dfv = st.session_state.dfv

st.title("Categoricals")
st.subheader("Categorical Analysis by Origin")

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

    if "origin" in dfv.columns:
        chart = stacked_categorical(dfv, cat_var)
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            "Legend and stacked segment order follow the intended clinical sequence. "
            "Different origins show different distributions of categorical features."
        )
else:
    st.info("No label columns found.")
