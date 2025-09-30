import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer

st.set_page_config(
    page_title="CMSE 830 Midterm Project",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
# ---Data Cleaning---
st.title("Data Overview")

# Load datasets
diabetes_df = pd.read_csv(
    "diabetes.csv", usecols=["Glucose", "BloodPressure", "BMI", "Age", "Outcome"]
).rename(
    columns={
        "Glucose": "Blood Glucose",
        "BloodPressure": "Blood Pressure",
        "Outcome": "Diabetes",
    }
)
diabetes_df["Diabetes"] = diabetes_df["Diabetes"].astype("category")
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
# Convert categorical columns to 'Yes'/'No'
# Normalize whitespace/case, then map to 1/0
cols_map = ["Diabetes", "Kidney Disease"]
kidney_df[cols_map] = kidney_df[cols_map].apply(
    lambda s: s.astype("string").str.strip().str.lower().str.replace(r"\s+", "", regex=True)
)
kidney_df[cols_map] = kidney_df[cols_map].replace({"yes": 1, "no": 0, "ckd": 1, "notckd": 0})

# Imputer for numeric columns (mean) and categorical columns (mode)
imputer_mean = SimpleImputer(strategy="mean")
imputer_mode = SimpleImputer(strategy="most_frequent")


# Drop duplicates and impute missing values
def clean_data(df):
    df_clean = df.drop_duplicates().copy()

    num_cols = df_clean.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        df_clean[num_cols] = imputer_mean.fit_transform(df_clean[num_cols])

    cat_cols = df_clean.select_dtypes(include="object").columns
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
    styled_df = df.style.highlight_null(color="rgba(250, 0, 0, 0.11)")
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=500,
    )
    st.write(
        f"{diabetes_df.shape[0]} Rows × {diabetes_df.shape[1]} Columns",
    )

with tab2:
    show_clean2 = st.toggle(
        "Show Cleaned Data",
        value=False,
        key="kidney",
    )
    df = kidney_df_clean if show_clean2 else kidney_df

    # Highlight missing values in red
    styled_df = df.style.highlight_null(color="rgba(250, 0, 0, 0.11)")
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=500,
    )
    st.write(f"{kidney_df.shape[0]} Rows × {kidney_df.shape[1]} Columns")


# ---Stats---
st.title("Basic Stats")
tab3, tab4 = st.tabs(["Dataset 1: Diabetes", "Dataset 2: Kidney Disease"])
with tab3:
    st.write(diabetes_df_clean.describe().T)
    st.write(diabetes_df_clean.corr().T)
with tab4:
    st.write(kidney_df_clean.describe().T)
    st.write(kidney_df_clean.corr().T)
