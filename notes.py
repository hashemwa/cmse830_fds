import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import numpy as np

# Create data frames
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
df_cleveland = pd.read_csv(
    "data/cleveland.data", na_values="?", names=columns
).drop_duplicates()
df_hungary = pd.read_csv(
    "data/hungary.data", na_values="?", names=columns
).drop_duplicates()
df_long_beach_va = pd.read_csv(
    "data/long_beach_va.data", na_values="?", names=columns
).drop_duplicates()
df_switzerland = pd.read_csv(
    "data/switzerland.data", na_values="?", names=columns
).drop_duplicates()

# Add origin column to each DataFrame
df_cleveland["origin"] = "Cleveland"
df_hungary["origin"] = "Hungary"
df_long_beach_va["origin"] = "Long Beach VA"
df_switzerland["origin"] = "Switzerland"

# Add target column
df_cleveland["target"] = (df_cleveland["num"] > 0).astype(int)
df_hungary["target"] = (df_hungary["num"] > 0).astype(int)
df_switzerland["target"] = (df_switzerland["num"] > 0).astype(int)
df_long_beach_va["target"] = (df_long_beach_va["num"] > 0).astype(int)

# Combine all DataFrames
df_combined = pd.concat([df_cleveland, df_hungary, df_switzerland, df_long_beach_va])

# Missingness Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df_combined.isnull(), cbar=False, cmap="viridis")
plt.title("Missingness Heatmap")
plt.xlabel("Features")
plt.ylabel("Samples")
plt.show()

missing_counts_by_origin = df_combined.groupby("origin").apply(lambda g: g.isna().sum())

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

# Missingness Rate by Column and Origin
num_cols = df_combined.select_dtypes(include=[np.number]).columns.tolist()
miss_by_origin = (
    df_combined.groupby("origin")[num_cols].apply(lambda g: g.isna().mean()).T
)

plt.figure(figsize=(8, 5))
sns.heatmap(miss_by_origin, cmap="viridis", annot=True, fmt=".2f")
plt.title("Missingness Rate by Column and Origin")
plt.xlabel("Origin")
plt.ylabel("Column")
plt.show()


# Simple Imputer
def simple_impute(df):
    df2 = df.copy()

    num_cols = df2.select_dtypes(include=[np.number]).columns
    cat_cols = df2.select_dtypes(exclude=[np.number]).columns

    if len(num_cols) > 0:
        num_imp = SimpleImputer(strategy="mean")
        df2[num_cols] = num_imp.fit_transform(df2[num_cols])

    if len(cat_cols) > 0:
        cat_imp = SimpleImputer(strategy="most_frequent")
        df2[cat_cols] = cat_imp.fit_transform(df2[cat_cols])

    return df2


df_cleveland_simple = simple_impute(df_cleveland)
df_hungary_simple = simple_impute(df_hungary)
df_switzerland_simple = simple_impute(df_switzerland)
df_long_beach_va_simple = simple_impute(df_long_beach_va)

df_combined_simple = pd.concat(
    [
        df_cleveland_simple,
        df_hungary_simple,
        df_switzerland_simple,
        df_long_beach_va_simple,
    ]
)


# KNN Imputer
def knn_impute(df, n_neighbors=5):
    df2 = df.copy()

    # Split columns
    num_cols = df2.select_dtypes(include=[np.number]).columns
    cat_cols = df2.select_dtypes(exclude=[np.number]).columns

    # KNN for numeric columns
    if len(num_cols) > 0:
        knn = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
        df2[num_cols] = knn.fit_transform(df2[num_cols])

    # Most frequent for any non-numeric columns (if you ever have them)
    if len(cat_cols) > 0:
        cat_imp = SimpleImputer(strategy="most_frequent")
        df2[cat_cols] = cat_imp.fit_transform(df2[cat_cols])

    return df2


df_cleveland_knn = knn_impute(df_cleveland, n_neighbors=5)
df_hungary_knn = knn_impute(df_hungary, n_neighbors=5)
df_switzerland_knn = knn_impute(df_switzerland, n_neighbors=5)
df_long_beach_va_knn = knn_impute(df_long_beach_va, n_neighbors=5)

df_combined_knn = pd.concat(
    [df_cleveland_knn, df_hungary_knn, df_switzerland_knn, df_long_beach_va_knn]
)

# Binary columns -> 0/1
bin_cols = ["sex", "fbs", "exang", "target"]
for c in bin_cols:
    if c in df_combined_knn.columns:
        df_combined_knn[c] = (df_combined_knn[c].clip(0, 1) >= 0.5).astype("Int64")

# Ordinal/ranged codes
range_cols = {
    "cp": (1, 4),
    "restecg": (0, 2),
    "slope": (1, 3),
    "ca": (0, 3),
    "num": (0, 4),
}
for c, (lo, hi) in range_cols.items():
    if c in df_combined_knn.columns:
        df_combined_knn[c] = df_combined_knn[c].round().clip(lo, hi).astype("Int64")

# thal must be one of {3, 6, 7}
if "thal" in df_combined_knn.columns:
    allowed = np.array([3, 6, 7])
    df_combined_knn["thal"] = (
        df_combined_knn["thal"]
        .apply(
            lambda v: int(allowed[np.argmin(np.abs(allowed - v))])
            if pd.notna(v)
            else pd.NA
        )
        .astype("Int64")
    )

# Convert age to Int64
if "age" in df_combined_knn.columns:
    df_combined_knn["age"] = (
        df_combined_knn["age"].round().clip(lower=0).astype("Int64")
    )


# Set up the plot
plt.figure(figsize=(12, 7))

# Plot the distribution of the original data (where it's not missing)
# We use a kernel density estimate (kde) plot, which is like a smooth histogram.
sns.kdeplot(
    df_combined["trestbps"].dropna(), label="Original Data", color="blue", linewidth=2
)

# Plot the distribution of the data after Simple Imputation
sns.kdeplot(
    df_combined_simple["trestbps"],
    label="Simple Imputation (Mean)",
    color="red",
    linestyle="--",
)

# Plot the distribution of the data after KNN Imputation
sns.kdeplot(
    df_combined_knn["trestbps"], label="KNN Imputation", color="green", linestyle=":"
)

# Add titles and labels for clarity
plt.title('Comparison of Data Distributions After Imputation for "trestbps"')
plt.xlabel("Resting Blood Pressure (trestbps, mmHg)")
plt.ylabel("Density")
plt.legend()
plt.show()

df_final = df_combined_knn

# Set up the plot with a larger size for clarity
plt.figure(figsize=(12, 8))

# Create a box plot using seaborn
# We will use your best dataset: df_final
sns.boxplot(data=df_final, x="origin", y="oldpeak", palette="viridis")

# Add titles and labels
plt.title("ST Depression (oldpeak) by Data Source (Origin)")
plt.xlabel("Data Source (Origin)")
plt.ylabel("ST Depression (oldpeak)")
plt.show()

df_final.describe(include="all")

df_final.groupby("origin")[
    ["age", "trestbps", "chol", "thalach", "oldpeak"]
].describe().round(2)

# Missingness Heatmap Imputed
plt.figure(figsize=(12, 6))
sns.heatmap(df_final.isnull(), cbar=False, cmap="viridis")
plt.title("Missingness Heatmap After KNN Imputation")
plt.xlabel("Features")
plt.ylabel("Samples")
plt.show()

cp_map = pd.DataFrame(
    {
        "cp": [1, 2, 3, 4],
        "cp_label": [
            "typical angina",
            "atypical angina",
            "non-anginal pain",
            "asymptomatic",
        ],
    }
)

restecg_map = pd.DataFrame(
    {
        "restecg": [0, 1, 2],
        "restecg_label": ["normal", "ST–T wave abnormality (>0.05 mV)", "LVH by Estes"],
    }
)

slope_map = pd.DataFrame(
    {"slope": [1, 2, 3], "slope_label": ["upsloping", "flat", "downsloping"]}
)

thal_map = pd.DataFrame(
    {"thal": [3, 6, 7], "thal_label": ["normal", "fixed defect", "reversible defect"]}
)

num_map = pd.DataFrame(
    {
        "num": [0, 1, 2, 3, 4],
        "num_label": [
            "no heart disease",
            "mild heart disease",
            "moderate heart disease",
            "severe heart disease",
            "critical heart disease",
        ],
    }
)

df_final = (
    df_final.merge(cp_map, on="cp", how="left")
    .merge(restecg_map, on="restecg", how="left")
    .merge(slope_map, on="slope", how="left")
    .merge(thal_map, on="thal", how="left")
    .merge(num_map, on="num", how="left")
)
