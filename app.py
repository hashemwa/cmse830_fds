"""Streamlit dashboard tying social media habits to mental health and productivity.

The inline comments call out how each section satisfies the CMSE 830 rubric so you can
quickly tweak the project while staying on spec.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from typing import Optional, Tuple
from sklearn.impute import KNNImputer

# UI: App configuration (favicon, layout, sidebar state)
st.set_page_config(
    page_title="Social Media, Mental Health, and Productivity",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Requirement 1: Define shared resources for two distinct CSV sources.
DATA_DIR = Path(__file__).parent
MENTAL_STATUS_ORDER = ["Poor", "Fair", "Good", "Excellent"]
STRESS_ORDER = ["Low", "Medium", "High"]
MENTAL_MAP = {status: idx + 1 for idx, status in enumerate(MENTAL_STATUS_ORDER)}
STRESS_MAP = {level: idx + 1 for idx, level in enumerate(STRESS_ORDER)}


@st.cache_data
def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """Requirement 1: read each dataset once and report if it's missing."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"The file '{file_path}' could not be found. Make sure it exists alongside app.py.")
        return None


def prepare_dataset1(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Requirement 1 & 2: clean well-being data and encode ordinal categories."""
    if df is None or df.empty:
        return pd.DataFrame()

    mental = df.copy()
    mental.drop_duplicates(inplace=True)
    mental['Gender'] = mental['Gender'].str.title()
    mental['Support_Systems_Access'] = mental['Support_Systems_Access'].str.title()
    mental['Work_Environment_Impact'] = mental['Work_Environment_Impact'].str.title()
    mental['Online_Support_Usage'] = mental['Online_Support_Usage'].str.title()

    mental['Mental_Health_Status'] = pd.Categorical(
        mental['Mental_Health_Status'], categories=MENTAL_STATUS_ORDER, ordered=True
    )
    mental['Stress_Level'] = pd.Categorical(
        mental['Stress_Level'], categories=STRESS_ORDER, ordered=True
    )
    mental['Mental_Health_Code'] = (
        mental['Mental_Health_Status']
        .map(MENTAL_MAP)
        .astype(float)
    )
    mental['Stress_Code'] = (
        mental['Stress_Level']
        .map(STRESS_MAP)
        .astype(float)
    )
    return mental


def prepare_dataset2(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Requirement 1 & 3: tidy productivity data and impute missing numeric fields."""
    if df is None or df.empty:
        return pd.DataFrame()

    productivity = df.copy()
    productivity.drop_duplicates(inplace=True)
    productivity.rename(columns={'age': 'Age', 'gender': 'Gender'}, inplace=True)

    productivity['Gender'] = productivity['Gender'].str.title()
    productivity['job_type'] = productivity['job_type'].str.title()
    productivity['social_platform_preference'] = productivity['social_platform_preference'].str.title()
    productivity['uses_focus_apps'] = productivity['uses_focus_apps'].map({True: 'Yes', False: 'No'})
    productivity['has_digital_wellbeing_enabled'] = productivity[
        'has_digital_wellbeing_enabled'
    ].map({True: 'Yes', False: 'No'})

    # Requirement 3: KNNImputer provides a basic-yet-strong technique for filling gaps.
    numeric_cols = productivity.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        imputer = KNNImputer(n_neighbors=5)
        productivity[numeric_cols] = imputer.fit_transform(productivity[numeric_cols])
        if 'Age' in numeric_cols:
            productivity['Age'] = productivity['Age'].round().astype(int)

    productivity['Productivity_Gap'] = (
        productivity['perceived_productivity_score'] - productivity['actual_productivity_score']
    )
    return productivity


@st.cache_data
def load_and_prepare() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Requirement 1: load both datasets and return clean copies for the app."""
    mental_raw = load_data(str(DATA_DIR / 'dataset1.csv'))
    productivity_raw = load_data(str(DATA_DIR / 'dataset2.csv'))
    mental = prepare_dataset1(mental_raw)
    productivity = prepare_dataset2(productivity_raw)
    return mental, productivity


# Clear cached data first so changes to the CSVs propagate during development.
load_and_prepare.clear()
mental_df, productivity_df = load_and_prepare()

if mental_df.empty or productivity_df.empty:
    st.stop()

# Requirement 4: Streamlit UI with title, narrative, and interactive sidebar.
# UI: Page header (title + intro text)
st.title("Social media negatively impacts mental health and productivity")
st.markdown(
    "Two datasets come together here: one focuses on well-being and the other on workplace outcomes. "
    "Use the sidebar to focus on specific demographics and explore how heavy social media habits relate to stress, sleep, and productivity."
)

with st.sidebar:
    # UI: Sidebar – Filters section container
    st.header("Filters")  # Requirement 4: interactive controls (slider + multiselects).
    age_min = int(min(mental_df['Age'].min(), productivity_df['Age'].min()))
    age_max = int(max(mental_df['Age'].max(), productivity_df['Age'].max()))
    # UI: Sidebar control – Age slider
    selected_age = st.slider("Age range", min_value=age_min, max_value=age_max, value=(age_min, age_max))

    gender_options = sorted(mental_df['Gender'].unique().tolist())
    # UI: Sidebar control – Gender multiselect
    selected_genders = st.multiselect("Gender", options=gender_options, default=gender_options)

    job_options = sorted(productivity_df['job_type'].unique().tolist())
    # UI: Sidebar control – Job type multiselect
    selected_jobs = st.multiselect("Job type", options=job_options, default=job_options[:5])

    platform_options = sorted(productivity_df['social_platform_preference'].unique().tolist())
    # UI: Sidebar control – Preferred platform multiselect
    selected_platforms = st.multiselect("Preferred platform", options=platform_options, default=platform_options)

    # UI: Sidebar helper text – brief note about imputation
    st.caption("KNN imputation (k=5) fills missing productivity records. See the documentation tab for the full cleaning story.")


def apply_filters(
    mental: pd.DataFrame,
    productivity: pd.DataFrame,
    age_range: Tuple[int, int],
    genders: list,
    jobs: list,
    platforms: list,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Requirement 4: subset both datasets according to sidebar selections."""
    mental_filtered = mental[
        mental['Age'].between(age_range[0], age_range[1])
        & mental['Gender'].isin(genders)
    ]

    productivity_filtered = productivity[
        productivity['Age'].between(age_range[0], age_range[1])
        & productivity['Gender'].isin(genders)
        & productivity['job_type'].isin(jobs if jobs else job_options)
        & productivity['social_platform_preference'].isin(platforms if platforms else platform_options)
    ]
    return mental_filtered, productivity_filtered


mental_filtered, productivity_filtered = apply_filters(
    mental_df,
    productivity_df,
    selected_age,
    selected_genders or gender_options,
    selected_jobs or job_options,
    selected_platforms or platform_options,
)

if mental_filtered.empty or productivity_filtered.empty:
    st.warning("No records match the current filters. Try widening the selections.")
    st.stop()

# UI: Tabs for main app sections
overview_tab, mental_tab, productivity_tab, data_tab, docs_tab = st.tabs(["Overview", "Mental Health", "Productivity", "Data Details", "Documentation"])

with overview_tab:
    # Requirement 2
    # UI: Overview tab – metrics row (three KPIs)
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Avg social media hours (well-being dataset)",
        f"{mental_filtered['Social_Media_Usage_Hours'].mean():.2f}"
    )
    col2.metric(
        "Mental health score (1-4)",
        f"{mental_filtered['Mental_Health_Code'].mean():.2f}"
    )
    col3.metric(
        "Actual productivity (0-10)",
        f"{productivity_filtered['actual_productivity_score'].mean():.2f}"
    )

    st.markdown(
        "Heavy social media users lean toward higher stress codes and slightly lower productivity scores. "
        "The histogram below highlights how usage spreads across the selected audience."
    )

    # UI: Overview tab – Usage histogram (distribution of social media hours)
    usage_hist = px.histogram(
        mental_filtered,
        x='Social_Media_Usage_Hours',
        color='Mental_Health_Status',
        nbins=30,
        labels={'Social_Media_Usage_Hours': 'Daily social media hours'},
    )
    usage_hist.update_layout(height=400, legend_title="Mental health status")
    st.plotly_chart(usage_hist, use_container_width=True)

    # UI: Overview tab – side-by-side summary statistics tables
    summary_cols = st.columns(2)
    with summary_cols[0]:
        st.markdown("**Well-being stats (filtered)**")
        mental_summary = mental_filtered[
            ['Social_Media_Usage_Hours', 'Mental_Health_Code', 'Stress_Code', 'Sleep_Hours']
        ].describe().T[['mean', 'std', 'min', 'max']].round(2)
        st.dataframe(mental_summary, use_container_width=True)
    with summary_cols[1]:
        st.markdown("**Productivity stats (filtered)**")
        productivity_summary = productivity_filtered[
            [
                'daily_social_media_time',
                'actual_productivity_score',
                'perceived_productivity_score',
                'Productivity_Gap',
                'days_feeling_burnout_per_month'
            ]
        ].describe().T[['mean', 'std', 'min', 'max']].round(2)
        st.dataframe(productivity_summary, use_container_width=True)

with mental_tab:
    # Requirement 2
    # UI: Mental Health tab – sleep vs stress box plot
    st.subheader("Sleep and stress patterns")
    sleep_box = px.box(
        mental_filtered,
        x='Stress_Level',
        y='Sleep_Hours',
        color='Gender',
        points='all',
        labels={
            'Stress_Level': 'Reported stress level',
            'Sleep_Hours': 'Sleep per night (hours)'
        },
    )
    sleep_box.update_layout(height=420)
    st.plotly_chart(sleep_box, use_container_width=True)

    st.markdown(
        "Most stressed respondents tend to cluster around lower nightly sleep. Heavier social media segments also show a drop in the median mental health score."
    )

    # UI: Mental Health tab – mental health by gender (bar, colored by avg social use)
    mh_bar = px.bar(
        mental_filtered.groupby('Gender', as_index=False, observed=False).agg(
            avg_social=('Social_Media_Usage_Hours', 'mean'),
            avg_mental=('Mental_Health_Code', 'mean'),
        ),
        x='Gender',
        y='avg_mental',
        color='avg_social',
        labels={'avg_mental': 'Mental health score (1-4)', 'avg_social': 'Avg social media hours'},
        color_continuous_scale='Blues',
    )
    mh_bar.update_layout(height=400)
    st.plotly_chart(mh_bar, use_container_width=True)

with productivity_tab:
    # Requirement 2
    # UI: Productivity tab – job type bar (actual vs perceived as color)
    st.subheader("Job type outlook")
    job_bar = px.bar(
        productivity_filtered.groupby('job_type', as_index=False, observed=False).agg(
            actual=('actual_productivity_score', 'mean'),
            perceived=('perceived_productivity_score', 'mean'),
        ),
        x='job_type',
        y='actual',
        color='perceived',
        labels={
            'job_type': 'Job type',
            'actual': 'Actual productivity score',
            'perceived': 'Perceived productivity score',
        },
        color_continuous_scale='Magma',
    )
    job_bar.update_layout(height=420)
    st.plotly_chart(job_bar, use_container_width=True)

    st.markdown(
        "The gap between perceived and actual productivity widens for roles with higher social media exposure."
    )

    # UI: Productivity tab – notifications histogram split by platform
    notifications_hist = px.histogram(
        productivity_filtered,
        x='number_of_notifications',
        nbins=30,
        color='social_platform_preference',
        labels={'number_of_notifications': 'Notifications per day'},
    )
    notifications_hist.update_layout(height=400, legend_title="Platform")
    st.plotly_chart(notifications_hist, use_container_width=True)

with data_tab:
    # Requirement 1 & 2
    # UI: Data Details tab – mental dataset sample table
    st.subheader("Sample rows")
    st.markdown("Well-being dataset (after cleaning)")
    st.dataframe(  # UI: Data Details tab – mental dataframe
        mental_filtered[
            [
                'User_ID', 'Age', 'Gender', 'Social_Media_Usage_Hours', 'Mental_Health_Status',
                'Stress_Level', 'Sleep_Hours', 'Physical_Activity_Hours'
            ]
        ].head(100),
        use_container_width=True,
    )

    st.markdown("Productivity dataset (after KNN imputation)")
    st.dataframe(  # UI: Data Details tab – productivity dataframe
        productivity_filtered[
            [
                'Age', 'Gender', 'job_type', 'social_platform_preference', 'daily_social_media_time',
                'actual_productivity_score', 'perceived_productivity_score', 'Productivity_Gap',
                'days_feeling_burnout_per_month', 'weekly_offline_hours'
            ]
        ].head(100),
        use_container_width=True,
    )

with docs_tab:
    # Requirement 4
    # UI: Documentation tab – checklist and validation helpers
    st.subheader("Project checklist")  # In-app documentation for reviewers
    st.markdown("**Data sources**")
    st.markdown(
        "- `dataset1.csv`: mental health, stress, and lifestyle responses.\n"
        "- `dataset2.csv`: workplace productivity, notification load, and social platform usage."
    )

    st.markdown("**Cleaning & encoding**")
    st.markdown(
        "- Dropped duplicate rows in each file.\n"
        "- Standardized categorical strings (gender, job type, platform).\n"
        "- Converted ordinal categories (mental health, stress) into numeric scores for analysis."
    )

    st.markdown("**Missing data handling**")
    st.markdown(
        "- Applied KNN imputation (k=5) to numeric productivity fields with gaps (social media time, productivity scores, sleep, burnout).\n"
        "- Re-checked that all missing values are now resolved:"
    )

    # UI: Documentation tab – missing values report (side-by-side)
    missing_cols = st.columns(2)
    with missing_cols[0]:
        st.caption("Mental health dataset")
        missing_mental = mental_df.isna().sum()
        if (missing_mental == 0).all():
            st.success("No missing values present.")
        else:
            st.dataframe(missing_mental.to_frame('Missing values'))
    with missing_cols[1]:
        st.caption("Productivity dataset")
        missing_productivity = productivity_df.isna().sum()
        if (missing_productivity == 0).all():
            st.success("No missing values after KNN imputation.")
        else:
            st.dataframe(missing_productivity.to_frame('Missing values'))

    st.markdown("**Exploratory analysis**")
    st.markdown(
        "- Visuals in the tabs: histogram (usage distribution), box plot (sleep vs stress), and bar charts (mental health by gender, productivity by job).\n"
        "- Summary tables display mean, spread, and ranges for key variables."
    )

    st.markdown("**Streamlit features**")
    st.markdown(
        "- Sidebar sliders and multiselects provide at least two interactive controls.\n"
        "- Tabs and inline notes document how to read the data."
    )

    st.markdown("**Deployment checklist**")
    st.markdown(
        "1. Ensure `requirements.txt` lists `streamlit`, `pandas`, `numpy`, `plotly`, and `scikit-learn`.\n"
        "2. Push the repository to GitHub.\n"
        "3. Deploy via Streamlit Community Cloud (`streamlit run app.py`)."
    )

    st.markdown("**Repository notes**")
    st.markdown(
        "- README already describes setup steps and project overview.\n"
        "- Keep data files and this app together when publishing."
    )
