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

# Normalize text labels and map to 1/0 (handle stray spaces/tabs/case)
for col in ["Diabetes", "Kidney Disease"]:
    kidney_df[col] = (
        kidney_df[col]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "", regex=True)
    )
    kidney_df[col] = kidney_df[col].replace({"yes": 1, "no": 0, "ckd": 1, "notckd": 0})
    kidney_df[col] = pd.to_numeric(kidney_df[col]).astype("Int64")

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
    num_cols_view = list(df.select_dtypes(include="number").columns)
    fmt_map = {c: "{:.0f}" for c in num_cols_view}
    styled_df = df.style.format(fmt_map).highlight_null(color="rgba(250, 0, 0, 0.11)")
    st.dataframe(styled_df, use_container_width=True, height=500)
    st.write(f"Rows × Columns: {diabetes_df.shape[0]} × {diabetes_df.shape[1]}")

with tab2:
    show_clean2 = st.toggle("Show Cleaned Data", value=False, key="kidney")
    df = kidney_df_clean if show_clean2 else kidney_df
    # Format numeric columns with no decimals for display
    num_cols_view = list(df.select_dtypes(include="number").columns)
    fmt_map = {c: "{:.0f}" for c in num_cols_view}
    styled_df = df.style.format(fmt_map).highlight_null(color="rgba(250, 0, 0, 0.11)")
    st.dataframe(styled_df, use_container_width=True, height=500)
    st.write(f"Rows × Columns: {kidney_df.shape[0]} × {kidney_df.shape[1]}")


# --- Basic Stats ---
st.title("Basic Stats")
tab3, tab4 = st.tabs(["Dataset 1: Diabetes", "Dataset 2: Kidney Disease"])

with tab3:
    st.subheader("Summary Statistics")
    st.write(diabetes_df_clean.describe().T)
    st.subheader("Correlation Matrix")
    st.write(diabetes_df_clean.corr(numeric_only=True).T)

with tab4:
    st.subheader("Summary Statistics")
    st.write(kidney_df_clean.describe().T)
    st.subheader("Correlation Matrix")
    st.write(kidney_df_clean.corr(numeric_only=True).T)
