import streamlit as st
import pandas as pd
import altair as alt
from analysis import get_individual_raw_datasets

df_raw_filtered = st.session_state.df_raw_filtered

st.title("Missing Data")
st.markdown("*Understanding missing values in the **raw data** before imputation*")

total_cells = df_raw_filtered.size
missing_cells = df_raw_filtered.isnull().sum().sum()
missing_pct = 100 * missing_cells / total_cells if total_cells > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cells", f"{total_cells:,}")
col2.metric("Missing Cells", f"{missing_cells:,}")
col3.metric("Missing %", f"{missing_pct:.2f}%")
features_with_missing = (df_raw_filtered.isnull().sum() > 0).sum()
col4.metric(
    "Features Affected", f"{features_with_missing}/{len(df_raw_filtered.columns)}"
)

st.divider()

if df_raw_filtered.isnull().values.any():
    st.subheader("Missing Data by Feature")

    missing_summary = pd.DataFrame(
        {
            "Missing Count": df_raw_filtered.isnull().sum(),
            "Missing %": 100 * df_raw_filtered.isnull().sum() / len(df_raw_filtered),
        }
    ).sort_values("Missing Count", ascending=False)

    missing_summary = missing_summary[missing_summary["Missing Count"] > 0]

    if not missing_summary.empty:
        st.dataframe(
            missing_summary.round(2),
            use_container_width=True,
            column_config={
                "Missing Count": st.column_config.NumberColumn(format="%d"),
                "Missing %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

        st.caption(
            ":material/lightbulb: **Key observation:** Some features like `ca` and `thal` have substantial missing data."
        )

    st.divider()

    st.subheader("Missingness Patterns by Origin")
    st.caption(
        "True missing rates including both explicit NaN values and 'fake zeros' that represent missing data"
    )

    st.warning(
        "**Important:** Some hospitals recorded missing values as `0` instead of leaving them blank. "
        "For clinical features like cholesterol and blood pressure, a value of 0 is **impossible** in living patients. "
        "These 'fake zeros' are treated as missing data in this analysis.",
        icon=":material/warning:",
    )

    # Features where 0 is clinically impossible
    fake_zero_features = {
        "chol": "Serum cholesterol cannot be 0 mg/dL",
        "trestbps": "Resting blood pressure cannot be 0 mmHg",
        "thalach": "Maximum heart rate cannot be 0 bpm",
        "age": "Age cannot be 0 years",
    }

    # Calculate true missing rates
    raw_datasets = get_individual_raw_datasets()

    true_missing_data = []
    for origin, df_origin in raw_datasets.items():
        for col in [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]:
            if col not in df_origin.columns:
                continue

            explicit_nan = df_origin[col].isnull().sum()

            # Check for fake zeros only in continuous features where 0 is impossible
            fake_zeros = 0
            if col in fake_zero_features:
                fake_zeros = (df_origin[col] == 0).sum()

            total_missing = explicit_nan + fake_zeros
            pct = total_missing / len(df_origin) * 100

            true_missing_data.append(
                {
                    "Origin": origin,
                    "Feature": col,
                    "Explicit NaN": explicit_nan,
                    "Fake Zeros": fake_zeros,
                    "Total Missing": total_missing,
                    "Missing %": pct,
                }
            )

    true_missing_df = pd.DataFrame(true_missing_data)

    # Pivot table showing missing % by origin
    pivot_df = true_missing_df.pivot(
        index="Feature", columns="Origin", values="Missing %"
    ).round(1)
    pivot_df = pivot_df[["Cleveland", "Hungary", "Long Beach VA", "Switzerland"]]

    # Create Altair heatmap
    heatmap_data = pivot_df.reset_index().melt(
        id_vars="Feature", var_name="Origin", value_name="Missing %"
    )

    heatmap = (
        alt.Chart(heatmap_data)
        .mark_rect()
        .encode(
            x=alt.X("Origin:N", title="Origin", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Feature:N", title="Feature", sort=list(pivot_df.index)),
            color=alt.Color(
                "Missing %:Q",
                scale=alt.Scale(scheme="viridis", domain=[0, 100]),
                legend=alt.Legend(title="% Missing"),
            ),
            tooltip=[
                alt.Tooltip("Origin:N", title="Origin"),
                alt.Tooltip("Feature:N", title="Feature"),
                alt.Tooltip("Missing %:Q", title="True Missing %", format=".1f"),
            ],
        )
        .properties(height=450)
    )

    text = (
        alt.Chart(heatmap_data)
        .mark_text(baseline="middle")
        .encode(
            x=alt.X("Origin:N"),
            y=alt.Y("Feature:N", sort=list(pivot_df.index)),
            text=alt.Text("Missing %:Q", format=".1f"),
            color=alt.condition(
                alt.datum["Missing %"] > 50,
                alt.value("black"),
                alt.value("#f7f7f7"),
            ),
        )
    )

    st.altair_chart(heatmap + text, use_container_width=True)

    st.info(
        "**Why this matters:** The missing data patterns prove that hospitals collected data in fundamentally different ways. "
        "Some hospitals are missing entire features (like `ca` and `thal`) while others have complete data. "
        "This isn't random. It tells us that hospitals used different equipment, ran different tests, and followed different protocols. "
        "If we ignore these differences and just combine everything together, we're pretending four different hospitals are the same. "
        "That's why a 'one size fits all' approach may miss hospital-specific patterns.",
        icon=":material/info:",
    )

    st.divider()

    # Feature reliability summary
    st.subheader("Feature Reliability for Modeling")
    st.caption("Based on maximum missing rate across all origins (threshold: 50%)")

    max_missing = pivot_df.max(axis=1).sort_values(ascending=False)

    unreliable = max_missing[max_missing > 50]
    reliable = max_missing[max_missing <= 50]

    col_drop, col_keep = st.columns(2)

    with col_drop:
        st.error(f"**Features to Drop** ({len(unreliable)})", icon=":material/cancel:")
        if len(unreliable) > 0:
            for feat, rate in unreliable.items():
                worst_origin = pivot_df.loc[feat].idxmax()
                st.markdown(f"- `{feat}` — {rate:.0f}% in {worst_origin}")
        else:
            st.markdown("*None*")

    with col_keep:
        st.success(
            f"**Features to Keep** ({len(reliable)})", icon=":material/check_circle:"
        )
        for feat, rate in reliable.items():
            if rate > 0:
                worst_origin = pivot_df.loc[feat].idxmax()
                st.markdown(f"- `{feat}` — {rate:.0f}% max ({worst_origin})")
            else:
                st.markdown(f"- `{feat}` — Complete")

else:
    st.success(
        "No missing values in the filtered raw data!", icon=":material/check_circle:"
    )
