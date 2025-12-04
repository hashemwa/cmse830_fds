import streamlit as st
import pandas as pd
import altair as alt

dfv = st.session_state.dfv
df_raw_filtered = st.session_state.df_raw_filtered
df = st.session_state.df
origin_sel = st.session_state.origin_sel
age_range = st.session_state.age_range


st.title("Feature Engineering")
st.markdown("*Transforming Raw Data into Predictive Features*")

st.subheader("Target Variable Creation")
st.caption("Converting multi-class diagnosis to binary classification")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Original `num` (0-4)**")
    if "num" in df.columns:
        num_counts = df["num"].value_counts().sort_index()
        st.dataframe(
            pd.DataFrame(
                {
                    "Severity": [
                        "No disease",
                        "Mild",
                        "Moderate",
                        "Severe",
                        "Critical",
                    ],
                    "Code": [0, 1, 2, 3, 4],
                    "Count": [num_counts.get(i, 0) for i in range(5)],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

with col2:
    st.markdown("**Derived `target` (Binary)**")
    if "target" in df.columns:
        target_counts = df["target"].value_counts().sort_index()
        st.dataframe(
            pd.DataFrame(
                {
                    "Status": ["No Heart Disease", "Heart Disease Present"],
                    "Code": [0, 1],
                    "Count": [target_counts.get(0, 0), target_counts.get(1, 0)],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

st.code("df['target'] = (df['num'] > 0).astype(int)", language="python")

st.divider()

st.subheader("Categorical Encoding")
st.caption("Numeric codes mapped to descriptive labels for interpretability")

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
            "Clinical Meaning": [
                "Classic heart-related chest pain",
                "Chest pain with some but not all typical features",
                "Chest pain unlikely from heart",
                "No chest pain symptoms",
            ],
        }
    )
    st.dataframe(cp_mapping, use_container_width=True, hide_index=True, height=178)

with tab2:
    restecg_mapping = pd.DataFrame(
        {
            "Code": [0, 1, 2],
            "Label": ["normal", "ST-T wave abnormality", "LVH by Estes"],
            "Clinical Meaning": [
                "Normal resting electrocardiogram",
                "ST-T wave changes suggesting ischemia",
                "Left ventricular hypertrophy pattern",
            ],
        }
    )
    st.dataframe(restecg_mapping, use_container_width=True, hide_index=True, height=143)

with tab3:
    slope_mapping = pd.DataFrame(
        {
            "Code": [1, 2, 3],
            "Label": ["upsloping", "flat", "downsloping"],
            "Clinical Meaning": [
                "Normal response to exercise",
                "Borderline/suspicious",
                "Strongly suggests ischemia",
            ],
        }
    )
    st.dataframe(slope_mapping, use_container_width=True, hide_index=True, height=143)

with tab4:
    thal_mapping = pd.DataFrame(
        {
            "Code": [3, 6, 7],
            "Label": ["normal", "fixed defect", "reversible defect"],
            "Clinical Meaning": [
                "Normal blood flow",
                "Permanent damage (old heart attack)",
                "Temporary reduced flow (active ischemia)",
            ],
        }
    )
    st.dataframe(thal_mapping, use_container_width=True, hide_index=True, height=143)

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
    st.dataframe(num_mapping, use_container_width=True, hide_index=True, height=213)

label_cols = ["cp_label", "restecg_label", "slope_label", "thal_label", "num_label"]
original_cols = ["cp", "restecg", "slope", "thal", "num"]

if all(col in df.columns for col in label_cols):
    st.markdown("**Before & After Sample**")
    sample_df = df[original_cols + label_cols].head(5)

    col_before, col_after = st.columns(2)

    with col_before:
        st.caption("Original (Numeric)")
        st.dataframe(sample_df[original_cols], use_container_width=True, height=213)

    with col_after:
        st.caption("Transformed (Labels)")
        st.dataframe(sample_df[label_cols], use_container_width=True, height=213)

st.divider()

st.subheader("Feature Scaling")
st.caption("Standardization applied during model training via sklearn Pipeline")

with st.expander(
    "Why do we need feature scaling?", icon=":material/help:", expanded=True
):
    st.markdown("""
    **The problem:** Features have vastly different scales:
    - `age`: 29-77 (range ~50)
    - `trestbps`: 94-200 (range ~100)  
    - `oldpeak`: 0-6.2 (range ~6)
    
    **Why it matters:** Logistic Regression adjusts "weights" for each feature to find the best 
    decision boundary. If `trestbps` has values in the hundreds while `oldpeak` is in single digits, 
    the model might incorrectly assume `trestbps` is more important just because its numbers are bigger.
    
    **What StandardScaler does:** Transforms each feature to have Mean=0 and Std=1, putting all 
    features on the same scale so the model can fairly compare their importance.
    """)

num_features = ["age", "trestbps", "thalach", "oldpeak"]
available_num = [f for f in num_features if f in dfv.columns]

if available_num:
    raw_stats = dfv[available_num].describe().loc[["mean", "std", "min", "max"]].T
    raw_stats.columns = ["Mean", "Std", "Min", "Max"]

    scaled_stats = pd.DataFrame(
        {
            "Mean": [0.0] * len(available_num),
            "Std": [1.0] * len(available_num),
            "Min": (
                (dfv[available_num].min() - dfv[available_num].mean())
                / dfv[available_num].std()
            ).values,
            "Max": (
                (dfv[available_num].max() - dfv[available_num].mean())
                / dfv[available_num].std()
            ).values,
        },
        index=available_num,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Before Scaling**")
        st.dataframe(raw_stats.round(2), use_container_width=True, height=178)

    with col2:
        st.markdown("**After StandardScaler**")
        st.dataframe(scaled_stats.round(2), use_container_width=True, height=178)

st.divider()

st.subheader("Derived Features")
st.caption("Additional feature engineering techniques demonstrated")

st.info(
    "**Note:** These derived features are shown for educational purposes to demonstrate feature engineering concepts. "
    "They are **not used** in the final models on the Modeling page, which rely only on the original reliable features.",
    icon=":material/info:",
)

derived_features = pd.DataFrame(
    {
        "Feature": ["age_thalach", "trestbps_oldpeak", "hr_reserve"],
        "Formula": ["age * thalach", "trestbps * oldpeak", "(220 - age) - thalach"],
        "Rationale": [
            "Interaction between age and exercise capacity",
            "Combined blood pressure and ischemia indicator",
            "Heart rate reserve (fitness indicator)",
        ],
    }
)

st.dataframe(derived_features, hide_index=True, use_container_width=True, height=143)

if "age" in dfv.columns and "thalach" in dfv.columns:
    with st.expander(
        "Example: Heart Rate Reserve by Origin", icon=":material/show_chart:"
    ):
        dfv_copy = dfv.copy()
        dfv_copy["hr_reserve"] = (220 - dfv_copy["age"]) - dfv_copy["thalach"]

        hr_chart = (
            alt.Chart(dfv_copy)
            .mark_boxplot()
            .encode(
                x=alt.X("origin:N", title="Origin", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("hr_reserve:Q", title="Heart Rate Reserve"),
                color=alt.Color("origin:N", legend=None),
            )
            .properties(height=300)
        )

        st.altair_chart(hr_chart, use_container_width=True)

        st.caption(
            "Heart rate reserve = (220 - age) - max_heart_rate. "
            "Lower values indicate the patient achieved closer to their predicted maximum."
        )

st.divider()

st.info(
    "**Why this matters:** Feature engineering is where domain knowledge meets data science. "
    "By understanding what each clinical measurement means and how it was collected, "
    "we can create better features and make informed decisions about data quality. "
    "See the **Missing Data** page for details on which features are excluded from modeling.",
    icon=":material/info:",
)
