import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer

st.set_page_config(
    page_title="CMSE 830 Midterm Project",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Data Overview ---
st.title("Data Overview")

# Load datasets (subset and rename for readability)
diabetes_df = pd.read_csv(
    "diabetes.csv", usecols=["Glucose", "BloodPressure", "BMI", "Age", "Outcome"]
).rename(
    columns={
        "Glucose": "Blood Glucose",
        "BloodPressure": "Blood Pressure",
        "Outcome": "Diabetes",
    }
)
# Ensure Diabetes stays numeric (0/1) with nullable integer dtype
diabetes_df["Diabetes"] = pd.to_numeric(diabetes_df["Diabetes"]).astype("Int64")

kidney_df = pd.read_csv(
    "kidney_disease.csv", usecols=["age", "bp", "bgr", "dm", "classification"]
).rename(
    columns={
        "age": "Age",
        "bp": "Blood Pressure",
        "bgr": "Blood Glucose",
        "dm": "Diabetes",
        "classification": "Kidney Disease",
    }
)

# Normalize text labels and map to 1/0
for col in ["Diabetes", "Kidney Disease"]:
    kidney_df[col] = (
        kidney_df[col]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({"yes": "1", "no": "0", "ckd": "1", "notckd": "0"})
        .astype("Int64")
    )

# Simple cleaning: drop duplicates, impute numeric with mean and categoricals with mode
imputer_mean = SimpleImputer(strategy="mean")
imputer_mode = SimpleImputer(strategy="most_frequent")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.drop_duplicates().copy()
    num_cols = df_clean.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        df_clean[num_cols] = imputer_mean.fit_transform(df_clean[num_cols])
    # Include object, string extension, and category dtypes
    cat_cols = df_clean.select_dtypes(
        include=["object", "string", "category", "boolean"]
    ).columns
    if len(cat_cols) > 0:
        df_clean[cat_cols] = imputer_mode.fit_transform(df_clean[cat_cols])
    return df_clean


diabetes_df_clean = clean_data(diabetes_df)
kidney_df_clean = clean_data(kidney_df)


# Two tabs: Dataset 1 (Diabetes) and Dataset 2 (Kidney Disease)
tab1, tab2 = st.tabs(["Dataset 1: Diabetes", "Dataset 2: Kidney Disease"])

with tab1:
    show_clean = st.toggle("Show Cleaned Data", value=False)
    df = diabetes_df_clean if show_clean else diabetes_df
    # Format numeric columns with no decimals for display
    styled_df = df.style.format(
        "{:.0f}", subset=df.select_dtypes(include="number").columns
    ).highlight_null(color="rgba(250, 0, 0, 0.11)")
    st.dataframe(styled_df, width="stretch", height=500)
    st.write(f"Rows × Columns: {diabetes_df.shape[0]} × {diabetes_df.shape[1]}")

with tab2:
    show_clean2 = st.toggle("Show Cleaned Data", value=False, key="kidney")
    df = kidney_df_clean if show_clean2 else kidney_df
    # Format numeric columns with no decimals for display
    styled_df = df.style.format(
        "{:.0f}", subset=df.select_dtypes(include="number").columns
    ).highlight_null(color="rgba(250, 0, 0, 0.11)")
    st.dataframe(styled_df, width="stretch", height=500)
    st.write(f"Rows × Columns: {kidney_df.shape[0]} × {kidney_df.shape[1]}")


# --- Basic Stats ---
st.title("Basic Stats")
tab3, tab4 = st.tabs(["Dataset 1: Diabetes", "Dataset 2: Kidney Disease"])

with tab3:
    st.subheader("Summary Statistics")
    stats = (
        diabetes_df_clean.describe()
        .rename(
            index={
                "count": "Count",
                "mean": "Mean",
                "std": "Standard Deviation",
                "min": "Minimum",
                "25%": "25th Percentile",
                "50%": "Median",
                "75%": "75th Percentile",
                "max": "Maximum",
            }
        )
        .T.round(3)
    )
    st.write(stats)
    st.subheader("Correlation Matrix")
    st.write(diabetes_df_clean.corr().T.round(3))
with tab4:
    st.subheader("Summary Statistics")
    stats = (
        kidney_df_clean.describe()
        .rename(
            index={
                "count": "Count",
                "mean": "Mean",
                "std": "Standard Deviation",
                "min": "Minimum",
                "25%": "25th Percentile",
                "50%": "Median",
                "75%": "75th Percentile",
                "max": "Maximum",
            }
        )
        .T.round(3)
    )
    st.write(stats)
    st.subheader("Correlation Matrix")
    st.write(kidney_df_clean.corr().T.round(3))

# --- Dataset Variable Counts ---
st.title("Count of Variables")


# Helper function to add ranges into labels
# Helper
bp_bins = [0, 60, 80, 90, 200]
bp_labels = ["Low (<60)", "Normal (60–79)", "Elevated (80–89)", "High (90+)"]

glucose_bins = [0, 100, 126, 500]
glucose_labels = ["Normal (<100)", "Prediabetes (100–125)", "Diabetes (126+)"]

bmi_bins = [0, 18.5, 25, 30, 70]
bmi_labels = [
    "Underweight (<18.5)",
    "Normal (18.5–24.9)",
    "Overweight (25–29.9)",
    "Obese (30+)",
]

age_bins = [0, 18, 40, 60, 100]
age_labels = [
    "Child (0–17)",
    "Young Adult (18–39)",
    "Middle Age (40–59)",
    "Senior (60+)",
]


# Function to categorize and count
def categorize_and_count(df, col, bins, labels, dataset_name):
    cat_col = pd.cut(df[col], bins=bins, labels=labels, right=False)
    counts = cat_col.value_counts().reindex(labels, fill_value=0).reset_index()
    counts.columns = [col, "Count"]
    counts["Dataset"] = dataset_name
    return counts


# Create summary tables for each variable
bp_counts = pd.concat(
    [
        categorize_and_count(
            diabetes_df_clean, "Blood Pressure", bp_bins, bp_labels, "Diabetes"
        ),
        categorize_and_count(
            kidney_df_clean, "Blood Pressure", bp_bins, bp_labels, "Kidney Disease"
        ),
    ]
)

glucose_counts = pd.concat(
    [
        categorize_and_count(
            diabetes_df_clean, "Blood Glucose", glucose_bins, glucose_labels, "Diabetes"
        ),
        categorize_and_count(
            kidney_df_clean,
            "Blood Glucose",
            glucose_bins,
            glucose_labels,
            "Kidney Disease",
        ),
    ]
)

bmi_counts = categorize_and_count(
    diabetes_df_clean, "BMI", bmi_bins, bmi_labels, "Diabetes"
)  # Only Dataset 1 has BMI

age_counts = pd.concat(
    [
        categorize_and_count(
            diabetes_df_clean, "Age", age_bins, age_labels, "Diabetes"
        ),
        categorize_and_count(
            kidney_df_clean, "Age", age_bins, age_labels, "Kidney Disease"
        ),
    ]
)

# --- Display in Streamlit ---
tab5, tab6, tab7, tab8 = st.tabs(["Blood Pressure", "Blood Glucose", "BMI", "Age"])

# ---------------- Blood Pressure ----------------
with tab5:
    st.subheader("Blood Pressure Categories")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Dataset 1**")
        st.dataframe(
            categorize_and_count(
                diabetes_df_clean, "Blood Pressure", bp_bins, bp_labels, "Diabetes"
            )[["Blood Pressure", "Count"]]
        )
    with col2:
        st.markdown("**Dataset 2**")
        st.dataframe(
            categorize_and_count(
                kidney_df_clean, "Blood Pressure", bp_bins, bp_labels, "Kidney Disease"
            )[["Blood Pressure", "Count"]]
        )

    st.markdown("**Combined Summary**")
    combined_bp = bp_counts.groupby("Blood Pressure")["Count"].sum().reset_index()
    st.dataframe(combined_bp)

# ---------------- Blood Glucose ----------------
with tab6:
    st.subheader("Blood Glucose Categories")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Dataset 1**")
        st.dataframe(
            categorize_and_count(
                diabetes_df_clean,
                "Blood Glucose",
                glucose_bins,
                glucose_labels,
                "Diabetes",
            )[["Blood Glucose", "Count"]]
        )
    with col2:
        st.markdown("**Dataset 2**")
        st.dataframe(
            categorize_and_count(
                kidney_df_clean,
                "Blood Glucose",
                glucose_bins,
                glucose_labels,
                "Kidney Disease",
            )[["Blood Glucose", "Count"]]
        )

    st.markdown("**Combined Summary**")
    combined_glucose = (
        glucose_counts.groupby("Blood Glucose")["Count"].sum().reset_index()
    )
    st.dataframe(combined_glucose)

# ---------------- BMI ----------------
with tab7:
    st.subheader("BMI Categories (Dataset 1 only)")

    st.markdown("**Dataset 1**")
    st.dataframe(bmi_counts[["BMI", "Count"]])

# ---------------- Age ----------------
with tab8:
    st.subheader("Age Categories")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Dataset 1**")
        st.dataframe(
            categorize_and_count(
                diabetes_df_clean, "Age", age_bins, age_labels, "Diabetes"
            )[["Age", "Count"]]
        )
    with col2:
        st.markdown("**Dataset 2**")
        st.dataframe(
            categorize_and_count(
                kidney_df_clean, "Age", age_bins, age_labels, "Kidney Disease"
            )[["Age", "Count"]]
        )

    st.markdown("**Combined Summary**")
    combined_age = age_counts.groupby("Age")["Count"].sum().reset_index()
    st.dataframe(combined_age)

# ---------------- Health Conditions ----------------
st.subheader("Health Conditions")

# Diabetes count (both datasets)
diabetes_counts = pd.DataFrame(
    {
        "Condition": ["Has Diabetes", "No Diabetes"],
        "Count": [
            pd.concat([diabetes_df_clean["Diabetes"], kidney_df_clean["Diabetes"]])
            .sum()
            .round(0),
            pd.concat(
                [diabetes_df_clean["Diabetes"], kidney_df_clean["Diabetes"]]
            ).shape[0]
            - pd.concat([diabetes_df_clean["Diabetes"], kidney_df_clean["Diabetes"]])
            .sum()
            .round(0),
        ],
    }
)

# Kidney disease count (only in kidney dataset)
kidney_counts = pd.DataFrame(
    {
        "Condition": ["Has Kidney Disease", "No Kidney Disease"],
        "Count": [
            kidney_df_clean["Kidney Disease"].sum(),
            kidney_df_clean.shape[0] - kidney_df_clean["Kidney Disease"].sum(),
        ],
    }
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Diabetes (Both Datasets)**")
    st.dataframe(diabetes_counts)
with col2:
    st.markdown("**Kidney Disease (Kidney Dataset)**")
    st.dataframe(kidney_counts)
