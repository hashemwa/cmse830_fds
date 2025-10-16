import streamlit as st

# Import analysis functions as needed
from analysis import (
    thalach_vs_age_trend,
    correlation_heatmap,
)

# Get data from session state
dfv = st.session_state.dfv
df_raw_filtered = st.session_state.df_raw_filtered
df = st.session_state.df
origin_sel = st.session_state.origin_sel
age_range = st.session_state.age_range


st.title("Relationships")
st.markdown(
    "*Exploratory Data Analysis — Examining Feature Interactions Across Origins*"
)

st.subheader("Max Heart Rate vs Age by Origin")
st.markdown(
    "Maximum heart rate typically declines with age, but the rate of decline may vary across patient populations. "
    "The LOESS smoothing curves reveal origin-specific patterns in this fundamental cardiovascular relationship."
)

needed_cols = {"age", "thalach", "target", "origin"}
if needed_cols.issubset(dfv.columns):
    chart = thalach_vs_age_trend(dfv)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Need columns: age, thalach, target, origin.")

st.divider()

st.subheader("Feature Correlation Matrix by Origin")
st.markdown(
    "Compare how numerical features correlate with each other and with heart disease across different institutions. "
    "If correlation patterns differ significantly by origin, it suggests that relationships between features are not universal."
)

if "origin" in dfv.columns:
    # Create tabs for each origin
    origins = sorted(dfv["origin"].unique())
    tabs = st.tabs(origins)

    for origin, tab in zip(origins, tabs):
        with tab:
            # Filter data for this origin
            origin_data = dfv[dfv["origin"] == origin]

            # Show sample size
            st.caption(f"**Sample size:** {len(origin_data)} patients")

            # Generate correlation heatmap
            corr_chart = correlation_heatmap(origin_data)
            if corr_chart is not None:
                st.altair_chart(corr_chart, use_container_width=True)
            else:
                st.warning("Not enough numerical features for correlation analysis.")
else:
    # Fallback: show combined heatmap if no origin column
    corr_chart = correlation_heatmap(dfv)
    if corr_chart is not None:
        st.altair_chart(corr_chart, use_container_width=True)
    else:
        st.warning("Not enough numerical features for correlation analysis.")

st.divider()

st.info(
    "**Why this matters:** If correlation patterns vary significantly across origins (e.g., `oldpeak` correlates differently with `target` "
    "in Cleveland vs. Hungary), it reveals that feature relationships are not universal. These differences arise from varying patient populations, "
    "fitness levels, comorbidities, or testing protocols at different institutions. Models assuming uniform relationships across all origins "
    "would miss these nuances, potentially leading to poor predictions when applied to new populations. "
    "Compare the heatmaps above—if you see different correlation strengths for the same feature pairs, that's evidence for origin-specific modeling.",
    icon=":material/info:",
)
