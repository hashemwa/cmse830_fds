import streamlit as st
from analysis import (
    thalach_vs_age_trend,
    correlation_heatmap,
)

dfv = st.session_state.dfv
df_raw_filtered = st.session_state.df_raw_filtered
df = st.session_state.df
origin_sel = st.session_state.origin_sel
age_range = st.session_state.age_range


st.title("Relationships")
st.markdown("*Examining Feature Interactions Across Origins*")

st.subheader("Max Heart Rate vs Age by Origin")
st.caption(
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
st.caption(
    "Compare how numerical features correlate with each other and with heart disease across different institutions. "
    "If correlation patterns differ significantly by origin, it suggests that relationships between features are not universal."
)

if "origin" in dfv.columns:
    origins = sorted(dfv["origin"].unique())
    tabs = st.tabs(origins)

    for origin, tab in zip(origins, tabs):
        with tab:
            origin_data = dfv[dfv["origin"] == origin]

            st.caption(f"**Sample size:** {len(origin_data)} patients")

            corr_chart = correlation_heatmap(origin_data)
            if corr_chart is not None:
                st.altair_chart(corr_chart, use_container_width=True)
            else:
                st.warning("Not enough numerical features for correlation analysis.")
else:
    corr_chart = correlation_heatmap(dfv)
    if corr_chart is not None:
        st.altair_chart(corr_chart, use_container_width=True)
    else:
        st.warning("Not enough numerical features for correlation analysis.")

st.divider()

st.info(
    "**Why this matters:** The relationship between features (like age and heart rate) changes depending on which hospital collected the data. "
    "For example, the link between ST depression (`oldpeak`) and heart disease is twice as strong in Cleveland compared to Long Beach. "
    "This happens because different hospitals served different patient groups with different health characteristics. "
    "A prediction model that assumes the same relationships everywhere will perform poorly when used at a different hospital. "
    "In the correlation heatmap, different numbers for the same feature pairs means we need hospital-specific models.",
    icon=":material/info:",
)
