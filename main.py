import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Midterm Data Narrative", layout="wide")

# ---- Load Data ----
@st.cache_data
def load_data():
    dataset1 = pd.read_csv("dataset1.csv")
    dataset2 = pd.read_csv("dataset2.csv")
    return dataset1, dataset2

dataset1, dataset2 = load_data()

st.title("📊 Midterm Project: University Student Data Analysis")

# ---- Narrative Intro ----
st.markdown("""
This dashboard explores **two datasets** about university students.  
- **Dataset 1** (10,064 rows): Covers demographics, study habits, stress, grades, lifestyle factors.  
- **Dataset 2** (760 rows): Focuses on stress, coping mechanisms, exercise, and mental health.  

The goal is to integrate insights and build a narrative about how **academic performance** and **stress levels**  
relate to **lifestyle and support systems**.
""")

# ---- Dataset Overview ----
st.header("📂 Dataset Overview")
tab1, tab2 = st.tabs(["Dataset 1", "Dataset 2"])

with tab1:
    st.subheader("Dataset 1 Preview")
    st.dataframe(dataset1.head())
    st.write("Shape:", dataset1.shape)
    st.write("Missing values:")
    st.write(dataset1.isnull().sum())

with tab2:
    st.subheader("Dataset 2 Preview")
    st.dataframe(dataset2.head())
    st.write("Shape:", dataset2.shape)
    st.write("Missing values:")
    st.write(dataset2.isnull().sum())

# ---- Analysis Section ----
st.header("📈 Exploratory Analysis")

# Grades vs Stress Levels (Dataset 1)
st.subheader("Grades vs Stress Levels (Dataset 1)")
if "Grades" in dataset1.columns and "Stress_Levels" in dataset1.columns:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=dataset1, x="Grades", hue="Stress_Levels", ax=ax)
    plt.title("Stress Levels Distribution by Grades")
    st.pyplot(fig)

# Physical Activity vs Mental Stress (Dataset 2)
st.subheader("Physical Exercise vs Mental Stress Level (Dataset 2)")
if "Physical Exercise (Hours per week)" in dataset2.columns:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=dataset2,
                    x="Physical Exercise (Hours per week)",
                    y="Mental Stress Level",
                    hue="Gender",
                    ax=ax)
    plt.title("Exercise vs Mental Stress")
    st.pyplot(fig)

# Sleep Patterns vs GPA
st.subheader("Sleep Duration vs Academic Performance (Dataset 2)")
if "Sleep Duration (Hours per night)" in dataset2.columns:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(data=dataset2,
                x="Sleep Duration (Hours per night)",
                y="Academic Performance (GPA)",
                ax=ax)
    plt.title("Sleep vs GPA")
    st.pyplot(fig)

# ---- Merge Attempt ----
st.header("🔗 Integrating the Datasets")

st.markdown("""
Dataset 1 measures **Physical Activity** as categorical (Low, Medium, High).  
Dataset 2 measures **Physical Exercise** in hours per week.  

To compare them, we can **bin exercise hours** into categories:
- **Low**: 0–3 hours  
- **Medium**: 4–7 hours  
- **High**: 8+ hours  
""")

if "Physical Exercise (Hours per week)" in dataset2.columns:
    dataset2["Exercise_Category"] = pd.cut(
        dataset2["Physical Exercise (Hours per week)"],
        bins=[-1,3,7,100],
        labels=["Low","Medium","High"]
    )
    st.write(dataset2[["Physical Exercise (Hours per week)","Exercise_Category"]].head())

    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=dataset2, x="Exercise_Category", ax=ax)
    plt.title("Distribution of Exercise Categories (Dataset 2)")
    st.pyplot(fig)

st.success("✅ Now you can narrate how lifestyle factors (exercise, sleep, stress) impact student performance.")
