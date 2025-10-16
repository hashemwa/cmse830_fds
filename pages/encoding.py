import streamlit as st
import pandas as pd

# Get data from session state
dfv = st.session_state.dfv
df_raw_filtered = st.session_state.df_raw_filtered
df = st.session_state.df
origin_sel = st.session_state.origin_sel
age_range = st.session_state.age_range


st.title("Encoding")
st.markdown("*Creating Labels for Categorical Variables*")

# Encoding mappings
st.subheader("Categorical Variable Mappings")
st.caption(
    "Numeric codes were mapped to descriptive labels to improve interpretability:"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Chest Pain", "Resting ECG", "ST Slope", "Thalassemia", "Diagnosis"]
)

with tab1:
    cp_mapping = pd.DataFrame(
        {
            "Code": [1, 2, 3, 4],
            "Label": [
                "typical angina",
                "atypical angina",
                "non-anginal pain",
                "asymptomatic",
            ],
        }
    )
    st.dataframe(cp_mapping, use_container_width=True, hide_index=True)

with tab2:
    restecg_mapping = pd.DataFrame(
        {
            "Code": [0, 1, 2],
            "Label": [
                "normal",
                "ST–T wave abnormality (>0.05 mV)",
                "LVH by Estes",
            ],
        }
    )
    st.dataframe(restecg_mapping, use_container_width=True, hide_index=True)

with tab3:
    slope_mapping = pd.DataFrame(
        {
            "Code": [1, 2, 3],
            "Label": ["upsloping", "flat", "downsloping"],
        }
    )
    st.dataframe(slope_mapping, use_container_width=True, hide_index=True)

with tab4:
    thal_mapping = pd.DataFrame(
        {
            "Code": [3, 6, 7],
            "Label": ["normal", "fixed defect", "reversible defect"],
        }
    )
    st.dataframe(thal_mapping, use_container_width=True, hide_index=True)

with tab5:
    num_mapping = pd.DataFrame(
        {
            "Code": [0, 1, 2, 3, 4],
            "Label": [
                "no heart disease",
                "mild heart disease",
                "moderate heart disease",
                "severe heart disease",
                "critical heart disease",
            ],
        }
    )
    st.dataframe(num_mapping, use_container_width=True, hide_index=True)

# Before/After comparison
st.subheader("Before & After Sample")

label_cols = ["cp_label", "restecg_label", "slope_label", "thal_label", "num_label"]
original_cols = ["cp", "restecg", "slope", "thal", "num"]

if all(col in df.columns for col in label_cols):
    sample_df = df[original_cols + label_cols].head(5)

    col_before, col_after = st.columns(2)

    with col_before:
        st.markdown("**Original (Numeric)**")
        st.dataframe(
            sample_df[original_cols],
            use_container_width=True,
        )

    with col_after:
        st.markdown("**Transformed (Labels)**")
        st.dataframe(
            sample_df[label_cols],
            use_container_width=True,
        )

st.divider()

st.info(
    "**Why this matters:** Machine learning models need numbers, but humans need words to understand the data. "
    "By keeping both the numeric codes (for computers) and text labels (for people), "
    "we can analyze the data properly while still being able to explain our findings "
    "to doctors and patients who are not familiar with data science.",
    icon=":material/info:",
)
