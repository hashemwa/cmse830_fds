import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import numpy as np
import altair as alt
from scipy.stats import gaussian_kde


alt.data_transformers.disable_max_rows()

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

# info
print(df_cleveland.info())
print(df_hungary.info())
print(df_long_beach_va.info())
print(df_switzerland.info())

# describe
print(df_cleveland.describe())
print(df_hungary.describe())
print(df_long_beach_va.describe())
print(df_switzerland.describe())

# Combine all DataFrames
df_combined = pd.concat([df_cleveland, df_hungary, df_switzerland, df_long_beach_va])

print(df_combined.info())
print(df_combined.describe())

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

print(df_final.info())
print(df_final.describe(include="all"))

print(
    df_final.groupby("origin")[["age", "trestbps", "chol", "thalach", "oldpeak"]]
    .describe()
    .round(2)
)

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


# ==================== Streamlit App Content ====================
# clean data for visualization
def get_clean_data(k=5):
    return df_final.copy()


# raw data for missingness
def get_raw_data():
    return df_combined.copy()


# individual raw datasets for IDA display
def get_individual_raw_datasets():
    return {
        "Cleveland": df_cleveland.copy(),
        "Hungary": df_hungary.copy(),
        "Long Beach VA": df_long_beach_va.copy(),
        "Switzerland": df_switzerland.copy(),
    }


# individual simple imputed datasets
def get_individual_simple_imputed():
    return {
        "Cleveland": df_cleveland_simple.copy(),
        "Hungary": df_hungary_simple.copy(),
        "Long Beach VA": df_long_beach_va_simple.copy(),
        "Switzerland": df_switzerland_simple.copy(),
    }


# individual KNN imputed datasets
def get_individual_knn_imputed():
    return {
        "Cleveland": df_cleveland_knn.copy(),
        "Hungary": df_hungary_knn.copy(),
        "Long Beach VA": df_long_beach_va_knn.copy(),
        "Switzerland": df_switzerland_knn.copy(),
    }


# combined simple imputed
def get_combined_simple_imputed():
    return df_combined_simple.copy()


# combined KNN imputed (before label merging)
def get_combined_knn_imputed():
    return df_combined_knn.copy()


# colors for origins
COLOR_MAP = {
    "Cleveland": "#4c78a8",
    "Hungary": "#f58518",
    "Long Beach VA": "#54a24b",
    "Switzerland": "#e45756",
}

# category orders for categorical variables
CATEGORY_ORDERS = {
    "cp_label": [
        "typical angina",
        "atypical angina",
        "non-anginal pain",
        "asymptomatic",
    ],
    "restecg_label": ["normal", "ST–T wave abnormality (>0.05 mV)", "LVH by Estes"],
    "slope_label": ["upsloping", "flat", "downsloping"],
    "thal_label": ["normal", "fixed defect", "reversible defect"],
    "num_label": [
        "no heart disease",
        "mild heart disease",
        "moderate heart disease",
        "severe heart disease",
        "critical heart disease",
    ],
}


def kde_by_origin(df, col):
    # variable labels and units
    var_labels = {
        "trestbps": "Resting Blood Pressure",
        "chol": "Serum Cholesterol",
        "thalach": "Maximum Heart Rate",
        "oldpeak": "ST Depression",
        "age": "Age",
    }
    units = {"trestbps": "(mmHg)", "chol": "(mg/dL)", "thalach": "(bpm)", "oldpeak": ""}

    var_display_name = var_labels.get(col, col)
    x_axis_label = f"{var_display_name} {units.get(col, '')}".strip()

    x_min = df[col].min()
    x_max = df[col].max()

    # handle where all values are the same
    if x_max - x_min < 1e-10:
        return None

    # 200 smooth points for x-axis
    x_smooth = np.linspace(x_min, x_max, 200)

    density_data = {}
    for origin in sorted(df["origin"].unique()):
        origin_data = df[df["origin"] == origin][col].dropna().values
        if len(origin_data) > 1 and origin_data.std() > 1e-10:
            try:
                # gaussian KDE for smooth density curves
                kde = gaussian_kde(origin_data, bw_method="scott")
                density = kde(x_smooth)
                density_data[origin] = density
            except (np.linalg.LinAlgError, ValueError):
                # fallback to interpolated histogram if KDE fails
                bins = np.linspace(origin_data.min(), origin_data.max(), 50)
                counts, _ = np.histogram(origin_data, bins=bins, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                density_data[origin] = np.interp(x_smooth, bin_centers, counts)

    if not density_data:
        return None

    chart_data = []
    for origin, density in density_data.items():
        for x_val, y_val in zip(x_smooth, density):
            chart_data.append(
                {
                    x_axis_label: x_val,
                    "Density": y_val,
                    "Origin": origin,
                }
            )
    chart_df = pd.DataFrame(chart_data)

    area_chart = (
        alt.Chart(chart_df)
        .mark_area(opacity=0.6)
        .encode(
            x=alt.X(
                f"{x_axis_label}:Q",
                title=x_axis_label,
                axis=alt.Axis(grid=False),
            ),
            y=alt.Y(
                "Density:Q",
                title="Density",
                axis=alt.Axis(gridOpacity=0.5),
            ),
            color=alt.Color(
                "Origin:N",
                legend=alt.Legend(title="Origin", orient="right"),
                scale=alt.Scale(
                    domain=list(COLOR_MAP.keys()),
                    range=list(COLOR_MAP.values()),
                ),
            ),
            tooltip=[
                alt.Tooltip("Origin:N", title="Origin"),
                alt.Tooltip(f"{x_axis_label}:Q", title=x_axis_label, format=".2f"),
                alt.Tooltip("Density:Q", title="Density", format=".4f"),
            ],
        )
        .properties(height=400)
    )

    return area_chart


def thalach_vs_age_trend(df):
    # multi-line tooltip
    hover = alt.selection_point(
        fields=["age"],
        nearest=True,
        on="mouseover",
        empty=False,
    )

    # loess trend lines by origin
    lines = (
        alt.Chart(df)
        .transform_loess("age", "thalach", groupby=["origin"])
        .mark_line(size=3)
        .encode(
            x=alt.X("age:Q", title="Age (years)", axis=alt.Axis(grid=False)),
            y=alt.Y(
                "thalach:Q",
                title="Max Heart Rate (thalach)",
                axis=alt.Axis(gridOpacity=0.5),
            ),
            color=alt.Color(
                "origin:N",
                legend=alt.Legend(title="Origin"),
                scale=alt.Scale(
                    domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())
                ),
            ),
        )
    )

    points = (
        lines.mark_point(size=100, filled=True)
        .encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("origin:N", title="Origin"),
                alt.Tooltip("age:Q", title="Age", format=".0f"),
                alt.Tooltip("thalach:Q", title="Max Heart Rate", format=".1f"),
            ],
        )
        .add_params(hover)
    )

    # vertical line to show where you're hovering
    rule = (
        alt.Chart(df)
        .transform_loess("age", "thalach", groupby=["origin"])
        .mark_rule(color="gray", opacity=0.5)
        .encode(x="age:Q")
        .transform_filter(hover)
    )

    chart = (lines + points + rule).properties(height=450)

    return chart


def stacked_categorical(df, cat_col):
    # descriptive titles for legend
    legend_titles = {
        "cp_label": "Chest Pain Type",
        "restecg_label": "Resting ECG",
        "slope_label": "ST Segment Slope",
        "thal_label": "Thalassemia",
        "num_label": "Disease Severity",
    }

    # order categories legend
    df_copy = df.copy()
    if cat_col in CATEGORY_ORDERS:
        df_copy[cat_col] = pd.Categorical(
            df_copy[cat_col], categories=CATEGORY_ORDERS[cat_col], ordered=True
        )

    cat_df = df_copy.groupby(["origin", cat_col]).size().reset_index(name="n")
    cat_df["pct"] = cat_df.groupby("origin")["n"].transform(lambda x: 100 * x / x.sum())

    # order categories stack within each bar
    if cat_col in CATEGORY_ORDERS:
        cat_df["cat_rank"] = cat_df[cat_col].cat.codes

    chart = (
        alt.Chart(cat_df)
        .mark_bar()
        .encode(
            x=alt.X("origin:N", title="Origin", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "pct:Q",
                title="Percent Within Origin",
                stack="normalize",
                axis=alt.Axis(gridOpacity=0.5),
            ),
            color=alt.Color(
                f"{cat_col}:N",
                legend=alt.Legend(
                    title=legend_titles.get(cat_col, cat_col.replace("_", " ")),
                    symbolType="circle",
                ),
                scale=alt.Scale(domain=CATEGORY_ORDERS.get(cat_col)),
            ),
            order=alt.Order("cat_rank:Q")
            if "cat_rank" in cat_df.columns
            else alt.Undefined,
            tooltip=[
                alt.Tooltip("origin:N", title="Origin"),
                alt.Tooltip(f"{cat_col}:N", title="Label"),
                alt.Tooltip("n:Q", title="Count"),
                alt.Tooltip("pct:Q", title="Percent", format=".1f"),
            ],
        )
        .properties(height=400)
    )

    return chart


def prevalence_bar(df):
    prev = df.groupby("origin")["target"].agg(["mean", "count"]).reset_index()
    prev["Prevalence (target)"] = (100 * prev["mean"]).round(1)
    prev = prev.rename(columns={"origin": "Origin", "count": "n"})

    bar = (
        alt.Chart(prev)
        .mark_bar()
        .encode(
            x=alt.X(
                "Origin:N",
                title="Origin",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "Prevalence (target):Q",
                title="Prevalence (target)",
                axis=alt.Axis(
                    gridOpacity=0.5, format=".0f", labelExpr="datum.value + '%'"
                ),
            ),
            tooltip=[
                alt.Tooltip("Origin:N", title="Origin"),
                alt.Tooltip("n:Q", title="N"),
                alt.Tooltip(
                    "Prevalence (target):Q", format=".1f", title="Prevalence (%)"
                ),
            ],
        )
        .properties(height=450)
    )

    return bar


def missingness_heatmap(df, cols):
    # missingness rate by origin for each column
    miss_by_origin = df.groupby("origin")[cols].apply(lambda g: g.isna().mean() * 100)

    miss_long = miss_by_origin.reset_index().melt(
        id_vars="origin", var_name="Column", value_name="Missing (%)"
    )

    heatmap = (
        alt.Chart(miss_long)
        .mark_rect()
        .encode(
            x=alt.X("origin:N", title="Origin", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Column:N", title="Feature"),
            color=alt.Color(
                "Missing (%):Q",
                scale=alt.Scale(scheme="viridis", domain=[0, 100]),
                legend=alt.Legend(title="% Missing"),
            ),
            tooltip=[
                alt.Tooltip("origin:N", title="Origin"),
                alt.Tooltip("Column:N", title="Feature"),
                alt.Tooltip("Missing (%):Q", title="Missing %", format=".1f"),
            ],
        )
        .properties(height=500)
    )

    text = (
        alt.Chart(miss_long)
        .mark_text(baseline="middle")
        .encode(
            x=alt.X("origin:N"),
            y=alt.Y("Column:N"),
            text=alt.Text("Missing (%):Q", format=".1f"),
            color=alt.condition(
                alt.datum["Missing (%)"] > 50,
                alt.value("black"),
                alt.value("#f7f7f7"),
            ),
            tooltip=[
                alt.Tooltip("origin:N", title="Origin"),
                alt.Tooltip("Column:N", title="Feature"),
                alt.Tooltip("Missing (%):Q", title="Missing %", format=".1f"),
            ],
        )
    )

    return heatmap + text


def correlation_heatmap(df):
    """
    Creates a correlation heatmap for numerical features with heart disease target.

    Args:
        df: DataFrame with numerical features and target column

    Returns:
        Altair chart object showing correlation heatmap
    """
    # Select numerical columns (exclude origin which is categorical)
    num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    available_cols = [c for c in num_cols if c in df.columns]

    # Add target if available
    if "target" in df.columns:
        available_cols.append("target")

    if len(available_cols) < 2:
        return None

    # Calculate correlation matrix
    corr_matrix = df[available_cols].corr()

    # Reshape for Altair (long format)
    corr_data = []
    for i, row_var in enumerate(corr_matrix.index):
        for j, col_var in enumerate(corr_matrix.columns):
            corr_data.append(
                {
                    "Variable 1": row_var,
                    "Variable 2": col_var,
                    "Correlation": corr_matrix.iloc[i, j],
                }
            )

    corr_df = pd.DataFrame(corr_data)

    # Create heatmap
    heatmap = (
        alt.Chart(corr_df)
        .mark_rect()
        .encode(
            x=alt.X("Variable 1:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Variable 2:N", title=None),
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(
                    domain=[-1, 0, 1],
                    range=["#d73027", "#f7f7f7", "#1a9850"],
                ),
                legend=alt.Legend(title="Correlation"),
            ),
            tooltip=[
                alt.Tooltip("Variable 1:N", title="Variable 1"),
                alt.Tooltip("Variable 2:N", title="Variable 2"),
                alt.Tooltip("Correlation:Q", title="Correlation", format=".3f"),
            ],
        )
        .properties(height=400)
    )

    # Add text labels for correlation values
    text = (
        alt.Chart(corr_df)
        .mark_text(fontSize=10)
        .encode(
            x=alt.X("Variable 1:N"),
            y=alt.Y("Variable 2:N"),
            text=alt.Text("Correlation:Q", format=".2f"),
            color=alt.condition(
                alt.datum.Correlation > 0.5, alt.value("white"), alt.value("black")
            ),
        )
    )

    return heatmap + text
