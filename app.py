# Create data frames
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

columns = [
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
    "num",
]
df_cleveland = pd.read_csv("cleveland.data", na_values="?", names=columns)
df_hungarian = pd.read_csv("hungary.data", na_values="?", names=columns)
df_long_beach_va = pd.read_csv("long_beach_va.data", na_values="?", names=columns)
df_switzerland = pd.read_csv("switzerland.data", na_values="?", names=columns)

# Add origin column to each DataFrame
df_cleveland["origin"] = "Cleveland"
df_hungarian["origin"] = "Hungary"
df_long_beach_va["origin"] = "Long Beach VA"
df_switzerland["origin"] = "Switzerland"

# Combine all DataFrames
df_combined = pd.concat([df_cleveland, df_hungarian, df_switzerland, df_long_beach_va])

df_combined["target"] = (df_combined["num"] > 0).astype(int)

# Missingness Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df_combined.isnull(), cbar=False, cmap="viridis")
plt.title("Missingness Heatmap")
plt.xlabel("Features")
plt.ylabel("Samples")
plt.show()

missing_counts_by_origin = df_combined.groupby("origin").apply(lambda g: g.isna().sum())

# --- Run this code in the next cell ---

# First, let's get the total number of missing values for each origin by summing across the columns (axis=1)
total_missing_by_origin = missing_counts_by_origin.sum(axis=1)

# Now, let's create a bar plot to visualize these totals
plt.figure(figsize=(10, 6))
sns.barplot(
    x=total_missing_by_origin.index, y=total_missing_by_origin.values, palette="viridis"
)
plt.title("Total Number of Missing Values by Data Source")
plt.xlabel("Data Source (Origin)")
plt.ylabel("Total Count of Missing Cells")
plt.show()
