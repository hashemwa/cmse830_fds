import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def load_data(file_path):
    """Loads data from a given file path and returns a pandas DataFrame."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        return None

def clean_and_prepare_data(df1, df2):
    """
    Cleans and merges the two provided dataframes.
    - Standardizes gender columns
    - Performs basic imputation for missing values
    - Merges the two dataframes
    - Creates standardized and categorical versions for better visualizations
    """
    # --- Clean Dataset 1 (Academic) ---
    df1_cleaned = df1.copy()
    df1_cleaned.drop_duplicates(inplace=True)
    # Standardize Gender column
    df1_cleaned['Gender'] = df1_cleaned['Gender'].map({'M': 'Male', 'F': 'Female'})

    # --- Clean Dataset 2 (Well-being) ---
    df2_cleaned = df2.copy()
    df2_cleaned.drop_duplicates(inplace=True)

    # --- Imputation (Done before merge) ---
    for col in df1_cleaned.columns:
        if df1_cleaned[col].dtype == 'object':
            df1_cleaned[col].fillna(df1_cleaned[col].mode()[0], inplace=True)
        elif pd.api.types.is_numeric_dtype(df1_cleaned[col]):
            df1_cleaned[col].fillna(df1_cleaned[col].median(), inplace=True)

    for col in df2_cleaned.columns:
        if df2_cleaned[col].dtype == 'object':
            df2_cleaned[col].fillna(df2_cleaned[col].mode()[0], inplace=True)
        elif pd.api.types.is_numeric_dtype(df2_cleaned[col]):
            df2_cleaned[col].fillna(df2_cleaned[col].median(), inplace=True)

    # --- Merge Datasets ---
    # Merge on 'Age' and 'Gender' as common keys. This is an assumption for the project.
    df_merged = pd.merge(df1_cleaned, df2_cleaned, on=['Age', 'Gender'], how='inner')

    # --- Create Enhanced Columns for Better Visualizations ---
    # GPA Categories
    df_merged['GPA_Category'] = pd.cut(df_merged['Academic Performance (GPA)'], 
                                      bins=[-0.1, 1, 2, 3, 4.1], 
                                      labels=['Poor (0-1)', 'Below Average (1-2)', 'Average (2-3)', 'Good (3-4)'])
    
    # Stress Categories
    df_merged['Stress_Category'] = pd.cut(df_merged['Mental Stress Level'], 
                                         bins=[0, 3, 6, 8, 10], 
                                         labels=['Low (1-3)', 'Moderate (4-6)', 'High (7-8)', 'Very High (9-10)'])
    
    # Family Support Categories
    df_merged['Support_Category'] = pd.cut(df_merged['Family Support  '], 
                                          bins=[0, 2, 3, 4, 5], 
                                          labels=['Low (1-2)', 'Below Average (3)', 'Good (4)', 'High (5)'])
    
    # Sleep Categories
    df_merged['Sleep_Category'] = pd.cut(df_merged['Sleep Duration (Hours per night)'], 
                                        bins=[0, 6, 7, 8, 12], 
                                        labels=['Insufficient (<6h)', 'Below Optimal (6-7h)', 'Optimal (7-8h)', 'Excessive (>8h)'])
    
    # Exercise Categories
    df_merged['Exercise_Category'] = pd.cut(df_merged['Physical Exercise (Hours per week)'], 
                                           bins=[-1, 0, 2, 5, 10], 
                                           labels=['None (0h)', 'Low (1-2h)', 'Moderate (3-5h)', 'High (6+h)'])
    
    # Social Media Categories
    df_merged['Social_Media_Category'] = pd.cut(df_merged['Social Media Usage (Hours per day)'], 
                                               bins=[-1, 1, 3, 5, 8], 
                                               labels=['Low (0-1h)', 'Moderate (2-3h)', 'High (4-5h)', 'Very High (6+h)'])
    
    # Financial Stress Categories
    df_merged['Financial_Stress_Category'] = pd.cut(df_merged['Financial Stress'], 
                                                   bins=[0, 2, 3, 4, 5], 
                                                   labels=['Low (1-2)', 'Moderate (3)', 'High (4)', 'Very High (5)'])

    return df1_cleaned, df2_cleaned, df_merged
