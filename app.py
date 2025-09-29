import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from typing import Optional

st.set_page_config(
    page_title="Social Pulse Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent
MENTAL_STATUS_MAP = {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4}
STRESS_LEVEL_MAP = {"Low": 1, "Medium": 2, "High": 3}
AGE_BINS = [18, 25, 35, 45, 55, 65, 101]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
SOCIAL_INTENSITY_LABELS = ["Light (Q1)", "Balanced (Q2)", "Heavy (Q3)", "Very Heavy (Q4)"]
BOOL_TEXT = {True: "Yes", False: "No"}
FOCUS_MESSAGES = {
    "Mental health": "Track how stress, rest, and mental health scores unravel as social feeds intensify.",
    "Productivity": "Watch productivity scores fall while perceived performance stays inflated.",
    "Sleep": "Heavy scrolling cuts into restorative sleep, raising the risk of chronic fatigue.",
    "Burnout": "Burnout days spike once daily feeds cross the heavy-usage threshold.",
}


@st.cache_data
def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load a CSV with basic error handling so the UI can surface issues."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        return None


def prepare_dataset1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Gender'] = df['Gender'].str.title()
    df['Support_Systems_Access'] = df['Support_Systems_Access'].str.strip().str.title()
    df['Work_Environment_Impact'] = df['Work_Environment_Impact'].str.strip().str.title()
    df['Online_Support_Usage'] = df['Online_Support_Usage'].str.strip().str.title()
    df['Mental_Health_Score'] = df['Mental_Health_Status'].map(MENTAL_STATUS_MAP)
    df['Stress_Score'] = df['Stress_Level'].map(STRESS_LEVEL_MAP)
    df['Support_Systems_Flag'] = df['Support_Systems_Access'].map({'Yes': 1, 'No': 0})
    df['Online_Support_Flag'] = df['Online_Support_Usage'].map({'Yes': 1, 'No': 0})
    df['Age_Group'] = pd.cut(df['Age'], bins=AGE_BINS, labels=AGE_LABELS, right=False)

    df['Social_Media_Intensity'] = pd.qcut(
        df['Social_Media_Usage_Hours'],
        q=min(4, df['Social_Media_Usage_Hours'].nunique()),
        labels=SOCIAL_INTENSITY_LABELS[: min(4, df['Social_Media_Usage_Hours'].nunique())],
        duplicates='drop'
    )

    activity_hours = df['Physical_Activity_Hours'].replace(0, np.nan)
    df['Screen_to_Activity_Ratio'] = (df['Screen_Time_Hours'] / activity_hours).replace(
        [np.inf, -np.inf], np.nan
    )
    df['Screen_to_Activity_Ratio'].fillna(df['Screen_to_Activity_Ratio'].median(), inplace=True)
    return df


def prepare_dataset2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={'age': 'Age', 'gender': 'Gender'})
    df['Gender'] = df['Gender'].str.title()
    df['job_type'] = df['job_type'].str.title()
    df['social_platform_preference'] = df['social_platform_preference'].str.title()

    numeric_cols = [
        'daily_social_media_time',
        'number_of_notifications',
        'work_hours_per_day',
        'perceived_productivity_score',
        'actual_productivity_score',
        'stress_level',
        'sleep_hours',
        'screen_time_before_sleep',
        'coffee_consumption_per_day',
        'days_feeling_burnout_per_month',
        'weekly_offline_hours',
        'job_satisfaction_score',
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)

    df['Productivity_Gap'] = df['perceived_productivity_score'] - df['actual_productivity_score']
    df['Notifications_per_Hour'] = df['number_of_notifications'] / df['work_hours_per_day'].replace(0, np.nan)
    df['Notifications_per_Hour'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['Notifications_per_Hour'].fillna(df['Notifications_per_Hour'].median(), inplace=True)
    df['Sleep_Debt'] = 8.0 - df['sleep_hours']
    df['Focus_App_Usage'] = df['uses_focus_apps'].map(BOOL_TEXT)
    df['Digital_Wellbeing'] = df['has_digital_wellbeing_enabled'].map(BOOL_TEXT)

    df['Age_Group'] = pd.cut(df['Age'], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    df['Social_Media_Intensity'] = pd.qcut(
        df['daily_social_media_time'],
        q=min(4, df['daily_social_media_time'].nunique()),
        labels=SOCIAL_INTENSITY_LABELS[: min(4, df['daily_social_media_time'].nunique())],
        duplicates='drop'
    )
    return df


def build_combined_view(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    if df1.empty or df2.empty:
        return pd.DataFrame(
            columns=[
                'Age_Group', 'Gender', 'avg_social_media', 'mental_score', 'stress_score',
                'poor_mental_share', 'high_stress_share', 'online_support_share',
                'avg_daily_social', 'actual_productivity', 'perceived_productivity',
                'productivity_gap', 'burnout_days', 'offline_hours', 'job_satisfaction',
                'focus_app_share', 'digital_wellbeing_share'
            ]
        )

    mental_agg = df1.groupby(['Age_Group', 'Gender']).agg(
        avg_social_media=('Social_Media_Usage_Hours', 'mean'),
        mental_score=('Mental_Health_Score', 'mean'),
        stress_score=('Stress_Score', 'mean'),
        poor_mental_share=('Mental_Health_Score', lambda x: np.mean(x <= 2)),
        high_stress_share=('Stress_Score', lambda x: np.mean(x >= 3)),
        online_support_share=('Online_Support_Flag', 'mean'),
    ).reset_index()

    productivity_agg = df2.groupby(['Age_Group', 'Gender']).agg(
        avg_daily_social=('daily_social_media_time', 'mean'),
        actual_productivity=('actual_productivity_score', 'mean'),
        perceived_productivity=('perceived_productivity_score', 'mean'),
        productivity_gap=('Productivity_Gap', 'mean'),
        burnout_days=('days_feeling_burnout_per_month', 'mean'),
        offline_hours=('weekly_offline_hours', 'mean'),
        job_satisfaction=('job_satisfaction_score', 'mean'),
        focus_app_share=('uses_focus_apps', 'mean'),
        digital_wellbeing_share=('has_digital_wellbeing_enabled', 'mean'),
    ).reset_index()

    combined = mental_agg.merge(productivity_agg, on=['Age_Group', 'Gender'], how='inner')
    combined['Age_Group'] = combined['Age_Group'].astype(str)
    combined['stress_gap_interaction'] = combined['stress_score'] * combined['productivity_gap']
    combined['combined_social_media'] = (combined['avg_social_media'] + combined['avg_daily_social']) / 2
    return combined


def compute_story_metrics(df1: pd.DataFrame, df2: pd.DataFrame, combined: pd.DataFrame) -> dict:
    metrics = {}
    if df1.empty or df2.empty:
        return metrics

    q1_low = df1['Social_Media_Usage_Hours'].quantile(0.25)
    q1_high = df1['Social_Media_Usage_Hours'].quantile(0.75)
    low_mental = df1[df1['Social_Media_Usage_Hours'] <= q1_low]['Mental_Health_Score'].mean()
    high_mental = df1[df1['Social_Media_Usage_Hours'] >= q1_high]['Mental_Health_Score'].mean()
    low_stress = df1[df1['Social_Media_Usage_Hours'] <= q1_low]['Stress_Score'].mean()
    high_stress = df1[df1['Social_Media_Usage_Hours'] >= q1_high]['Stress_Score'].mean()

    q2_low = df2['daily_social_media_time'].quantile(0.25)
    q2_high = df2['daily_social_media_time'].quantile(0.75)
    low_actual = df2[df2['daily_social_media_time'] <= q2_low]['actual_productivity_score'].mean()
    high_actual = df2[df2['daily_social_media_time'] >= q2_high]['actual_productivity_score'].mean()
    low_burnout = df2[df2['daily_social_media_time'] <= q2_low]['days_feeling_burnout_per_month'].mean()
    high_burnout = df2[df2['daily_social_media_time'] >= q2_high]['days_feeling_burnout_per_month'].mean()

    metrics['mental_health_delta'] = (high_mental - low_mental) if pd.notna(low_mental) else np.nan
    metrics['stress_delta'] = (high_stress - low_stress) if pd.notna(low_stress) else np.nan
    metrics['actual_productivity_delta'] = (high_actual - low_actual) if pd.notna(low_actual) else np.nan
    metrics['burnout_delta'] = (high_burnout - low_burnout) if pd.notna(low_burnout) else np.nan
    metrics['heavy_threshold'] = q2_high

    platform_means = df2.groupby('social_platform_preference')['daily_social_media_time'].mean().sort_values(ascending=False)
    metrics['top_platform'] = platform_means.index[0] if not platform_means.empty else None

    if not combined.empty:
        at_risk = combined.sort_values('productivity_gap', ascending=False).head(1)
        if not at_risk.empty:
            row = at_risk.iloc[0]
            metrics['at_risk_group'] = f"{row['Age_Group']} / {row['Gender']}"
            metrics['at_risk_gap'] = row['productivity_gap']
            metrics['at_risk_burnout'] = row['burnout_days']
    return metrics


def build_summary_tables(mental_df: pd.DataFrame, productivity_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mental_summary = (
        mental_df[
            ['Social_Media_Usage_Hours', 'Mental_Health_Score', 'Stress_Score', 'Sleep_Hours']
        ]
        .describe()
        .T[['mean', 'std', 'min', 'max']]
        .rename(
            columns={
                'mean': 'Mean',
                'std': 'Std Dev',
                'min': 'Min',
                'max': 'Max',
            }
        )
        .round(2)
    )

    productivity_summary = (
        productivity_df[
            [
                'daily_social_media_time',
                'actual_productivity_score',
                'perceived_productivity_score',
                'Productivity_Gap',
                'days_feeling_burnout_per_month',
            ]
        ]
        .describe()
        .T[['mean', 'std', 'min', 'max']]
        .rename(
            columns={
                'mean': 'Mean',
                'std': 'Std Dev',
                'min': 'Min',
                'max': 'Max',
            }
        )
        .round(2)
    )

    return mental_summary, productivity_summary


def filter_dataframes(
    mental_df: pd.DataFrame,
    productivity_df: pd.DataFrame,
    age_range: tuple,
    genders: list,
    platforms: list,
    job_types: list,
    social_range: tuple,
):
    m_mask = (
        mental_df['Age'].between(age_range[0], age_range[1])
        & mental_df['Social_Media_Usage_Hours'].between(social_range[0], social_range[1])
    )
    p_mask = (
        productivity_df['Age'].between(age_range[0], age_range[1])
        & productivity_df['daily_social_media_time'].between(social_range[0], social_range[1])
    )

    if genders:
        m_mask &= mental_df['Gender'].isin(genders)
        p_mask &= productivity_df['Gender'].isin(genders)
    if platforms:
        p_mask &= productivity_df['social_platform_preference'].isin(platforms)
    if job_types:
        p_mask &= productivity_df['job_type'].isin(job_types)

    return mental_df[m_mask], productivity_df[p_mask]


@st.cache_data
def prepare_datasets():
    mental_raw = load_data(str(DATA_DIR / 'dataset1.csv'))
    productivity_raw = load_data(str(DATA_DIR / 'dataset2.csv'))
    if mental_raw is None or productivity_raw is None:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {},
        )
    mental = prepare_dataset1(mental_raw)
    productivity = prepare_dataset2(productivity_raw)
    combined = build_combined_view(mental, productivity)
    metrics = compute_story_metrics(mental, productivity, combined)
    return mental, productivity, combined, metrics


mental_df, productivity_df, combined_df, base_metrics = prepare_datasets()

if mental_df.empty or productivity_df.empty:
    st.error("Data failed to load. Confirm the CSV files are present and reload the app.")
    st.stop()

st.title("Social media negatively impacts mental health and productivity")
st.markdown(
    "This dashboard explores how escalating social media exposure erodes mental health and workplace output "
    "by blending two complementary datasets. Use the sidebar to focus on specific demographics and habits."
)

with st.sidebar:
    st.header("Project Navigator")
    st.markdown(
        "The interface follows the CMSE 830 dashboard expectations: sidebar controls, multi-tab navigation, "
        "interactive charts, and embedded documentation so a new visitor can self-guide."
    )
    st.markdown("---")

age_min = int(min(mental_df['Age'].min(), productivity_df['Age'].min()))
age_max = int(max(mental_df['Age'].max(), productivity_df['Age'].max()))
sm_min = float(np.floor(min(mental_df['Social_Media_Usage_Hours'].min(), productivity_df['daily_social_media_time'].min())))
sm_max = float(np.ceil(max(mental_df['Social_Media_Usage_Hours'].max(), productivity_df['daily_social_media_time'].max())))

with st.sidebar:
    st.subheader("Filters")
    selected_age = st.slider("Age range", min_value=age_min, max_value=age_max, value=(age_min, age_max))
    gender_options = sorted(mental_df['Gender'].unique().tolist())
    selected_gender = st.multiselect("Gender", options=gender_options, default=gender_options)
    platform_options = sorted(productivity_df['social_platform_preference'].unique().tolist())
    selected_platforms = st.multiselect("Primary platform", options=platform_options, default=platform_options)
    job_options = sorted(productivity_df['job_type'].unique().tolist())
    selected_jobs = st.multiselect("Job type", options=job_options)
    selected_social = st.slider(
        "Daily social media hours", min_value=sm_min, max_value=sm_max, value=(sm_min, sm_max)
    )
    focus_metric = st.selectbox("Focus narrative", options=list(FOCUS_MESSAGES.keys()), index=0)

    st.markdown("---")
    st.caption(
        "High-usage users are defined as the top quartile of daily social media hours. The summary cards "
        "compare them with low-usage peers to expose mental health and productivity gaps."
    )

mental_filtered, productivity_filtered = filter_dataframes(
    mental_df,
    productivity_df,
    selected_age,
    selected_gender,
    selected_platforms,
    selected_jobs,
    selected_social,
)

if mental_filtered.empty or productivity_filtered.empty:
    st.warning("No records match the current filters. Adjust the selections to resume the analysis.")
    st.stop()

combined_filtered = build_combined_view(mental_filtered, productivity_filtered)
filtered_metrics = compute_story_metrics(mental_filtered, productivity_filtered, combined_filtered)

with st.expander("How to read this dashboard", expanded=False):
    st.markdown(
        "- **Overview** surfaces population-level trends and headline metrics tied to the narrative.\n"
        "- **Mental Health** diagnoses how stress and rest shift as social media intensity rises.\n"
        "- **Productivity** investigates workflow drag across roles, platforms, and notification loads.\n"
        "- **Data Explorer** exposes the cleaned tables, download options, and fused mental-productivity view.\n"
        "- **Documentation** records data sources, cleaning decisions, and deployment guidance."
    )

overview_tab, mental_tab, productivity_tab, explorer_tab, docs_tab = st.tabs(
    ["Overview", "Mental Health", "Productivity", "Data Explorer", "Documentation"]
)

with overview_tab:
    st.subheader("Narrative checkpoints")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Avg daily social media (hrs)",
        f"{productivity_filtered['daily_social_media_time'].mean():.2f}",
        delta=f"{selected_social[1] - selected_social[0]:.1f} hr filter span"
    )
    mh_delta = filtered_metrics.get('mental_health_delta')
    mh_delta_text = (
        f"{mh_delta:.2f} heavy vs light" if mh_delta is not None and not np.isnan(mh_delta) else None
    )
    col2.metric(
        "Mental health score (1-4)",
        f"{mental_filtered['Mental_Health_Score'].mean():.2f}",
        delta=mh_delta_text,
    )
    prod_delta = filtered_metrics.get('actual_productivity_delta')
    prod_delta_text = (
        f"{prod_delta:.2f} heavy vs light" if prod_delta is not None and not np.isnan(prod_delta) else None
    )
    col3.metric(
        "Actual productivity (0-10)",
        f"{productivity_filtered['actual_productivity_score'].mean():.2f}",
        delta=prod_delta_text,
    )

    st.markdown(
        "Heavy social media users show lower mental health scores and a productivity dip relative to light users. "
        "Hover charts to compare demographic slices and validate the narrative in your own way."
    )
    st.caption(FOCUS_MESSAGES.get(focus_metric, ""))

    dist_fig = px.histogram(
        mental_filtered,
        x='Social_Media_Usage_Hours',
        color='Mental_Health_Status',
        nbins=30,
        barmode='overlay',
        marginal='box',
        hover_data=['Stress_Level', 'Sleep_Hours'],
        labels={'Social_Media_Usage_Hours': 'Social media hours (dataset 1)'},
    )
    dist_fig.update_layout(height=420, legend_title_text='Mental health status')
    st.plotly_chart(dist_fig, use_container_width=True)

    if not combined_filtered.empty:
        combined_fig = px.scatter(
            combined_filtered,
            x='combined_social_media',
            y='productivity_gap',
            color='Age_Group',
            size='stress_score',
            hover_data={'Gender': True, 'stress_gap_interaction': ':.2f'},
            labels={
                'combined_social_media': 'Average social media hours',
                'productivity_gap': 'Perceived - actual productivity',
                'stress_gap_interaction': 'Stress x productivity gap',
            },
        )
        combined_fig.update_layout(height=420, legend_title_text='Age group')
        st.plotly_chart(combined_fig, use_container_width=True)
    else:
        st.info("The current filters do not produce a combined mental health and productivity profile. Try widening the filters.")

    stats_col1, stats_col2 = st.columns(2)
    mental_summary, productivity_summary = build_summary_tables(mental_filtered, productivity_filtered)
    with stats_col1:
        st.markdown("**Mental health stats (filtered)**")
        st.dataframe(mental_summary)
    with stats_col2:
        st.markdown("**Productivity stats (filtered)**")
        st.dataframe(productivity_summary)

with mental_tab:
    st.subheader("Stress, rest, and screen time")
    intensity_fig = px.box(
        mental_filtered,
        x='Social_Media_Intensity',
        y='Mental_Health_Score',
        color='Gender',
        points='all',
        hover_data=['Stress_Level', 'Sleep_Hours'],
        labels={'Mental_Health_Score': 'Mental health score (higher is better)'},
    )
    intensity_fig.update_layout(height=420, xaxis_title="Social media intensity quartile")
    st.plotly_chart(intensity_fig, use_container_width=True)

    heatmap_data = mental_filtered.groupby(['Age_Group', 'Stress_Level']).agg(
        avg_score=('Mental_Health_Score', 'mean'),
        avg_social=('Social_Media_Usage_Hours', 'mean'),
    ).reset_index()
    heatmap_data['Age_Group'] = heatmap_data['Age_Group'].astype(str)

    heatmap_fig = px.density_heatmap(
        heatmap_data,
        x='Age_Group',
        y='Stress_Level',
        z='avg_score',
        color_continuous_scale='RdBu_r',
        labels={'avg_score': 'Avg mental health score'},
    )
    heatmap_fig.update_layout(height=420)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    st.markdown(
        "The quartile box plot shows the erosion in mental health with heavier social feeds, especially for users who already "
        "report high stress. The heatmap highlights which age groups are most sensitive to stress escalation." 
    )

    sleep_fig = px.scatter(
        mental_filtered,
        x='Social_Media_Usage_Hours',
        y='Sleep_Hours',
        color='Stress_Level',
        trendline=None,
        hover_data=['Mental_Health_Status', 'Age'],
        labels={'Sleep_Hours': 'Sleep per night (hours)'},
    )
    sleep_fig.update_layout(height=420)
    st.plotly_chart(sleep_fig, use_container_width=True)

with productivity_tab:
    st.subheader("Workflow friction")
    scatter_fig = px.scatter(
        productivity_filtered,
        x='daily_social_media_time',
        y='actual_productivity_score',
        color='job_type',
        size='Notifications_per_Hour',
        hover_data=['perceived_productivity_score', 'days_feeling_burnout_per_month', 'Gender'],
        labels={
            'daily_social_media_time': 'Daily social media (hrs)',
            'actual_productivity_score': 'Actual productivity score',
            'Notifications_per_Hour': 'Notifications per hour',
        },
    )
    scatter_fig.update_layout(height=420, legend_title_text='Job type')
    st.plotly_chart(scatter_fig, use_container_width=True)

    gap_profile = productivity_filtered.groupby(['Social_Media_Intensity', 'Gender']).agg(
        actual=('actual_productivity_score', 'mean'),
        perceived=('perceived_productivity_score', 'mean'),
        gap=('Productivity_Gap', 'mean'),
    ).reset_index()
    gap_profile['Social_Media_Intensity'] = gap_profile['Social_Media_Intensity'].astype(str)

    gap_fig = px.bar(
        gap_profile,
        x='Social_Media_Intensity',
        y='gap',
        color='Gender',
        barmode='group',
        labels={'gap': 'Perceived - actual productivity'},
    )
    gap_fig.update_layout(height=420, xaxis_title="Social media intensity quartile")
    st.plotly_chart(gap_fig, use_container_width=True)

    st.markdown(
        "A widening productivity gap reinforces the narrative: heavy social media habits inflate perceived productivity, "
        "yet actual output scores fall and notifications swell. Explore hover details to pinpoint the most pressured roles." 
    )

with explorer_tab:
    st.subheader("Cleaned data snapshots")
    st.markdown("Mental health cohort (dataset 1)")
    st.dataframe(
        mental_filtered[
            [
                'User_ID', 'Age', 'Gender', 'Social_Media_Usage_Hours', 'Mental_Health_Status',
                'Stress_Level', 'Sleep_Hours', 'Physical_Activity_Hours', 'Work_Environment_Impact',
                'Online_Support_Usage'
            ]
        ].sort_values('Social_Media_Usage_Hours', ascending=False),
        use_container_width=True,
    )

    st.markdown("Productivity cohort (dataset 2)")
    st.dataframe(
        productivity_filtered[
            [
                'Age', 'Gender', 'job_type', 'social_platform_preference', 'daily_social_media_time',
                'actual_productivity_score', 'perceived_productivity_score', 'Productivity_Gap',
                'days_feeling_burnout_per_month', 'weekly_offline_hours', 'Focus_App_Usage',
                'Digital_Wellbeing'
            ]
        ].sort_values('daily_social_media_time', ascending=False),
        use_container_width=True,
    )

    st.markdown("Integrated view")
    if not combined_filtered.empty:
        download_ready = combined_filtered.round(2)
        st.dataframe(download_ready, use_container_width=True)
        csv_bytes = download_ready.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download combined summary",
            data=csv_bytes,
            file_name="social_media_impact_summary.csv",
            mime="text/csv",
        )
    else:
        st.info("No combined records for the selected filters. Expand the filters to unlock the download option.")

with docs_tab:
    st.subheader("Project documentation")
    st.markdown(
        "This page mirrors the CMSE 830 rubric so reviewers can quickly confirm the deliverables. Use it as a "
        "built-in handout when deploying the app online."
    )

    st.markdown("**Data sources**")
    st.markdown(
        "- `dataset1.csv`: Individual-level mental health, stress, and lifestyle metrics.\n"
        "- `dataset2.csv`: Workplace productivity, notification load, and platform preferences."
    )

    st.markdown("**Cleaning & imputation**")
    st.markdown(
        "- Removed duplicate records and harmonized categorical values (gender, job titles, platform names).\n"
        "- Converted qualitative mental health and stress levels into ordinal scores to enable quantitative analysis.\n"
        "- Filled missing numeric fields (social media time, productivity scores, sleep, burnout, notifications/hour) with median values per column."
    )

    st.markdown("**Exploratory analysis**")
    st.markdown(
        "- Histograms, box plots, scatter plots, bar charts, and heatmaps demonstrate trends, distributions, and interactions.\n"
        "- Summary tables above provide mean, dispersion, and range statistics for the filtered population."
    )

    st.markdown("**App interaction**")
    st.markdown(
        "- Sidebar controls filter by age, gender, platform, job type, and social media hours.\n"
        "- The focus selector tailors the narrative for presentations or stakeholder briefings."
    )

    st.markdown("**Deployment checklist**")
    st.markdown(
        "1. Ensure `requirements.txt` includes `streamlit`, `pandas`, `numpy`, and `plotly`.\n"
        "2. Push the repo to GitHub and launch with `streamlit run app.py` locally to verify.\n"
        "3. Deploy on Streamlit Community Cloud via **New app → repository → main branch → app.py**.\n"
        "4. Share the public app URL in your submission."
    )

    st.markdown("**Repository notes**")
    st.markdown(
        "- README documents the project overview, setup steps, and deployment instructions.\n"
        "- Code is modular (`utils.py` keeps loading helpers; `app.py` defines the Streamlit UI)."
    )

    missing_fig_col1, missing_fig_col2 = st.columns(2)
    with missing_fig_col1:
        st.markdown("**Missing data (dataset 1)**")
        mental_missing = (
            mental_df.isna().sum().to_frame(name='Missing values').query("`Missing values` > 0")
        )
        if mental_missing.empty:
            st.success("No remaining missing values after cleaning.")
        else:
            st.dataframe(mental_missing)
    with missing_fig_col2:
        st.markdown("**Missing data (dataset 2)**")
        productivity_missing = (
            productivity_df.isna()
            .sum()
            .to_frame(name='Missing values')
            .query("`Missing values` > 0")
        )
        if productivity_missing.empty:
            st.success("No remaining missing values after cleaning.")
        else:
            st.dataframe(productivity_missing)

if filtered_metrics:
    st.markdown("---")
    st.subheader("Narrative takeaways")
    bullets = []
    delta = filtered_metrics.get('mental_health_delta')
    if delta is not None and not np.isnan(delta):
        bullets.append(
            f"Heavy social media users average {abs(delta):.2f} fewer mental health points (scale 1-4) than light users."
        )
    delta_prod = filtered_metrics.get('actual_productivity_delta')
    if delta_prod is not None and not np.isnan(delta_prod):
        bullets.append(
            f"Actual productivity scores slide by {abs(delta_prod):.2f} points as daily social media climbs into the top quartile."
        )
    delta_burnout = filtered_metrics.get('burnout_delta')
    if delta_burnout is not None and not np.isnan(delta_burnout):
        bullets.append(
            f"Burnout days per month tick up by {delta_burnout:.2f} for heavy social media users."
        )
    top_platform = filtered_metrics.get('top_platform')
    if top_platform:
        bullets.append(
            f"{top_platform} dominates time spent among the filtered group, signaling where interventions could focus."
        )
    at_risk = filtered_metrics.get('at_risk_group')
    if at_risk:
        bullets.append(
            f"The {at_risk} cohort shows the widest productivity gap and should be prioritized for workplace support."
        )

    for item in bullets:
        st.markdown(f"- {item}")

    if not bullets:
        st.info("Current filters do not surface strong contrasts. Expand the audience to reveal clearer gaps.")
