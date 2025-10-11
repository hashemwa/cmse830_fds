# =========================
# 0) Imports & config
# =========================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", 100)

# =========================
# 1) Load processed files & add origin
# =========================
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

df_cleveland["origin"] = "Cleveland"
df_hungarian["origin"] = "Hungary"
df_long_beach_va["origin"] = "Long Beach VA"
df_switzerland["origin"] = "Switzerland"


# =========================
# 2) Duplicate report & clean (BEFORE any imputation)
# =========================
def duplicate_report(df, site_name):
    print(f"\n=== {site_name} ===")
    n_exact = df.duplicated(keep=False).sum()
    print("Exact duplicates (all columns):", n_exact)

    feature_cols = [c for c in df.columns if c not in ["origin", "num", "target"]]
    n_feat_dups = df.duplicated(subset=feature_cols, keep=False).sum()
    print("Duplicates on features (ignoring label/origin):", n_feat_dups)

    grp = df.groupby(feature_cols, dropna=False)["num"].nunique()
    conflictings = grp[grp > 1].shape[0]
    print("Conflicting-label groups (same features, different num):", conflictings)


# Inspect
duplicate_report(df_cleveland, "Cleveland")
duplicate_report(df_hungarian, "Hungary")
duplicate_report(df_switzerland, "Switzerland")
duplicate_report(df_long_beach_va, "Long Beach VA")

# Drop exact duplicates per your findings (Hungary:2, Long Beach VA:2)
before_hu = len(df_hungarian)
df_hungarian = df_hungarian.drop_duplicates()
before_va = len(df_long_beach_va)
df_long_beach_va = df_long_beach_va.drop_duplicates()
print(f"\nDropped {before_hu - len(df_hungarian)} duplicate row(s) from Hungary")
print(
    f"Dropped {before_va - len(df_long_beach_va)} duplicate row(s) from Long Beach VA"
)

# (Optional) re-check
duplicate_report(df_hungarian, "Hungary (after dedupe)")
duplicate_report(df_long_beach_va, "Long Beach VA (after dedupe)")

# =========================
# 3) Site-wise imputation (categoricals=mode; numerics=KNN(k=5) with scaling)
#    Switzerland: chol==0 → NaN; DO NOT impute chol for Switzerland
# =========================
num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]  # numeric-ish
cat_cols = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
]  # integer-coded categories
label_cols = ["num"]  # label; target created later


def impute_one_site(df_site: pd.DataFrame, site_name: str, k: int = 5) -> pd.DataFrame:
    g = df_site.copy()

    # Site-specific pre-fix: Switzerland chol==0 means "unknown" → structural missing
    if site_name == "Switzerland":
        g.loc[g["chol"] == 0, "chol"] = np.nan

    # A) CATEGORICALS → most_frequent per site
    if cat_cols:
        g[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(g[cat_cols])

    # B) NUMERICS → scale → KNN → inverse scale per site
    knn_num_cols = num_cols.copy()
    if site_name == "Switzerland":
        # Leave 'chol' as NaN; exclude it from KNN distance
        knn_num_cols = [c for c in knn_num_cols if c != "chol"]

    if knn_num_cols:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(g[knn_num_cols])
        X_imp_scaled = KNNImputer(n_neighbors=k).fit_transform(X_scaled)
        g[knn_num_cols] = scaler.inverse_transform(X_imp_scaled)

    # Restore nullable Int for categorical codes
    for c in cat_cols:
        g[c] = pd.to_numeric(g[c], errors="coerce").astype("Int64")

    return g


# Run imputation per site
cle_imputed = impute_one_site(df_cleveland, "Cleveland", k=5)
hun_imputed = impute_one_site(df_hungarian, "Hungary", k=5)
swi_imputed = impute_one_site(df_switzerland, "Switzerland", k=5)
va_imputed = impute_one_site(df_long_beach_va, "Long Beach VA", k=5)

# =========================
# 4) Combine & create binary target
# =========================
df_sitewise = pd.concat(
    [cle_imputed, hun_imputed, swi_imputed, va_imputed], ignore_index=True
)
df_sitewise["target"] = (df_sitewise["num"] > 0).astype(int)
df_sitewise["origin"] = df_sitewise["origin"].astype("category")

# =========================
# 5) Sanity checks
# =========================
print("\nRows per origin (after dedupe + site-wise impute):")
print(df_sitewise["origin"].value_counts())

print("\nRemaining NaNs per column (expect CH 'chol' to be NaN):")
print(df_sitewise.isna().sum().sort_values(ascending=False).head(12))

print(
    "\nSwitzerland chol missing %:",
    (
        df_sitewise.loc[df_sitewise["origin"] == "Switzerland", "chol"].isna().mean()
        * 100
    ).round(1),
    "%",
)

# =========================
# 6) Simple plots to verify story
# =========================
# Missingness snapshot (shows CH chol left NaN)
plt.figure(figsize=(10, 4))
sns.heatmap(
    df_sitewise[["chol", "ca", "thal", "origin", "target"]].isna(),
    cbar=False,
    cmap="viridis",
)
plt.title("Post-Imputation Missingness (Site-wise; Switzerland 'chol' left NaN)")
plt.tight_layout()
plt.show()

# Chol by origin (CH will be missing)
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_sitewise, x="origin", y="chol", palette="viridis")
plt.title("Serum Cholesterol (chol) by Origin (Site-wise Imputed; CH chol left NaN)")
plt.xlabel("Origin")
plt.ylabel("chol (mg/dl)")
plt.show()

# Outcome-split numerics (presence vs absence)
plt.figure(figsize=(5, 3))
sns.boxplot(data=df_sitewise, x="target", y="oldpeak")
plt.title("oldpeak by target")
plt.tight_layout()
plt.show()
plt.figure(figsize=(5, 3))
sns.boxplot(data=df_sitewise, x="target", y="thalach")
plt.title("thalach by target")
plt.tight_layout()
plt.show()

# Categorical by target
plt.figure(figsize=(5, 3))
sns.countplot(data=df_sitewise, x="cp", hue="target")
plt.title("cp by target")
plt.tight_layout()
plt.show()
plt.figure(figsize=(5, 3))
sns.countplot(data=df_sitewise, x="exang", hue="target")
plt.title("exang by target")
plt.tight_layout()
plt.show()

# Interaction: oldpeak × thalach
plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=df_sitewise, x="oldpeak", y="thalach", hue="target", style="origin", alpha=0.75
)
plt.title("oldpeak × thalach (color=target, style=origin)")
plt.tight_layout()
plt.show()

# =========================
# 7) Save combined CSV for your app/repo
# =========================
df_sitewise.to_csv("heart_imputed_sitewise.csv", index=False)
print("\nSaved: heart_imputed_sitewise.csv")
