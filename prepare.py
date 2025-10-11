"""
Beginner-friendly preparation script for the UCI Heart Disease site files.

Run: python prepare.py

Steps:
1. Load each site file and tag the origin (Cleveland, Hungary, Switzerland, Long Beach VA).
2. Print duplicate diagnostics per site and drop exact duplicate rows only.
3. Impute missing data per site to avoid cross-site leakage:
   - Categorical codes -> most frequent value.
   - Numeric measures -> StandardScaler + KNNImputer (k=5) with origin-specific scaling.
   - Switzerland special rule: treat chol==0 as missing, leave chol as NaN, and exclude it from KNN.
4. Build a comparison dataset with simple (median) numeric imputation for optional analyses.
5. Concatenate all sites, create a binary target, set dtypes, and save results.
6. Print summary statistics, stability snapshots, and a generalization demo.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

COLUMNS = [
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

CAT_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

SITE_FILES = [
    ("cleveland.data", "Cleveland"),
    ("hungary.data", "Hungary"),
    ("switzerland.data", "Switzerland"),
    ("long_beach_va.data", "Long Beach VA"),
]


def load_sites():
    """Load each CSV, coerce numeric columns, and attach the site origin."""
    sites = {}
    print("Loading site files:")
    for filename, origin in SITE_FILES:
        df = pd.read_csv(filename, names=COLUMNS, na_values="?")
        for col in COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["origin"] = origin
        sites[origin] = df
        print(f" - {origin}: {len(df)} rows from {filename}")
    return sites


def duplicate_report(df, origin):
    """Print duplicate diagnostics for a single site."""
    feature_cols = [c for c in df.columns if c not in ["origin", "num"]]
    exact_dups = int(df.duplicated(keep=False).sum())
    feature_dups = int(df.duplicated(subset=feature_cols, keep=False).sum())
    conflicting = int(
        df.groupby(feature_cols, dropna=False)["num"].nunique().gt(1).sum()
    )
    print(f"\nDuplicate report for {origin}")
    print(f" exact duplicates (all columns): {exact_dups}")
    print(f" duplicate feature rows (ignoring origin/num): {feature_dups}")
    print(f" conflicting labels with same features: {conflicting}")


def drop_exact_duplicates(df, origin):
    """Drop exact duplicate rows and report how many were removed."""
    before = len(df)
    df_clean = df.drop_duplicates()
    removed = before - len(df_clean)
    if removed:
        print(f" Removed {removed} exact duplicate row(s) from {origin}.")
    else:
        print(f" No exact duplicates removed from {origin}.")
    return df_clean


def scale_for_knn(values, scaler):
    """Scale numeric values (preserving NaNs) using StandardScaler parameters."""
    scale = scaler.scale_.copy()
    scale[scale == 0] = 1.0  # avoid division by zero when a feature is constant
    return (values - scaler.mean_) / scale


def inverse_scale(values, scaler):
    """Undo scaling while keeping the StandardScaler parameters."""
    scale = scaler.scale_.copy()
    scale[scale == 0] = 1.0
    return values * scale + scaler.mean_


def impute_site_knn(df_site, origin, k=5):
    """
    Site-wise imputation:
    - Categorical codes -> most frequent.
    - Numeric -> StandardScaler + KNNImputer.
      Switzerland: chol==0 set to NaN and excluded from KNN (left missing post-impute).
    """
    site_df = df_site.copy()
    if origin == "Switzerland":
        chol_zero = int((site_df["chol"] == 0).sum(skipna=True))
        if chol_zero:
            print(f" Switzerland special case: {chol_zero} chol values set to NaN.")
        site_df.loc[site_df["chol"] == 0, "chol"] = np.nan

    if CAT_COLUMNS:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        site_df[CAT_COLUMNS] = cat_imputer.fit_transform(site_df[CAT_COLUMNS])

    numeric_cols = NUMERIC_COLUMNS.copy()
    if origin == "Switzerland":
        numeric_cols = [c for c in numeric_cols if c != "chol"]

    if numeric_cols:
        numeric_values = site_df[numeric_cols].to_numpy(dtype=float)
        col_means = np.nanmean(numeric_values, axis=0)
        values_for_fit = np.where(np.isnan(numeric_values), col_means, numeric_values)
        scaler = StandardScaler()
        scaler.fit(values_for_fit)
        scaled = scale_for_knn(numeric_values, scaler)
        knn = KNNImputer(n_neighbors=k)
        imputed_scaled = knn.fit_transform(scaled)
        site_df[numeric_cols] = inverse_scale(imputed_scaled, scaler)

    for col in CAT_COLUMNS:
        site_df[col] = pd.to_numeric(site_df[col], errors="coerce").astype("Int64")

    return site_df


def impute_site_simple(df_site, origin):
    """Simple baseline imputation (mode for categorical, median for numeric)."""
    site_df = df_site.copy()
    if origin == "Switzerland":
        chol_zero = int((site_df["chol"] == 0).sum(skipna=True))
        if chol_zero:
            site_df.loc[site_df["chol"] == 0, "chol"] = np.nan

    if CAT_COLUMNS:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        site_df[CAT_COLUMNS] = cat_imputer.fit_transform(site_df[CAT_COLUMNS])

    if NUMERIC_COLUMNS:
        num_imputer = SimpleImputer(strategy="median")
        site_df[NUMERIC_COLUMNS] = num_imputer.fit_transform(site_df[NUMERIC_COLUMNS])

    for col in CAT_COLUMNS:
        site_df[col] = pd.to_numeric(site_df[col], errors="coerce").astype("Int64")

    return site_df


def finalize_dataset(frames):
    """Concatenate site DataFrames, set dtypes, and add the binary target."""
    combined = pd.concat(frames, ignore_index=True)
    combined["target"] = (combined["num"] > 0).astype(int)
    combined["origin"] = combined["origin"].astype("category")
    combined["num"] = pd.to_numeric(combined["num"], errors="coerce").astype("Int64")
    for col in CAT_COLUMNS:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("Int64")
    return combined


def print_basic_statistics(df):
    """Show quick count/mean/std/min/max summaries for key variables."""
    key_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    summary = df[key_cols].describe().loc[["count", "mean", "std", "min", "max"]]
    print("\nBasic summary statistics (site-wise KNN imputed data):")
    print(summary.round(2))


def print_stability_tables(df_sitewise, df_simple):
    """Compare stability metrics between imputers to support the narrative."""
    focus_cols = ["oldpeak", "thalach", "ca"]
    sitewise_means = df_sitewise.groupby("origin")[focus_cols].mean()
    simple_means = df_simple.groupby("origin")[focus_cols].mean()
    stability = pd.concat(
        {"Site-wise KNN": sitewise_means, "Simple Imputer": simple_means}, axis=1
    )
    print("\nStability snapshot (mean oldpeak, thalach, ca by origin):")
    print(stability.round(2))


def generalization_demo(df_simple):
    """Train on Cleveland only and score on the other sites to show generalization gaps."""
    print("\nGeneralization demo (train on Cleveland, test on the other sites):")
    feature_cols = CAT_COLUMNS + NUMERIC_COLUMNS
    cle_mask = df_simple["origin"] == "Cleveland"
    X_train = df_simple.loc[cle_mask, feature_cols].astype(float)
    y_train = df_simple.loc[cle_mask, "target"].astype(int)

    if X_train.empty:
        print(" Not enough Cleveland data to train the model.")
        return

    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_train, y_train)

    for origin in df_simple["origin"].unique():
        if origin == "Cleveland":
            continue
        test_mask = df_simple["origin"] == origin
        X_test = df_simple.loc[test_mask, feature_cols].astype(float)
        y_test = df_simple.loc[test_mask, "target"].astype(int)

        if X_test.empty:
            print(f" {origin}: no rows available for evaluation.")
            continue
        if X_test.isna().any().any():
            print(
                f" {origin}: skipped (imputation left NaNs; ensure prepare.py ran successfully)."
            )
            continue

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = float("nan")
        print(f" {origin:<15} | accuracy: {accuracy:.3f} | AUC: {auc:.3f}")


def main():
    print("=== Preparing UCI Heart Disease site-wise dataset ===")
    sites = load_sites()

    print("\n=== Duplicate diagnostics ===")
    cleaned_sites = {}
    for origin, df in sites.items():
        duplicate_report(df, origin)
        cleaned_sites[origin] = drop_exact_duplicates(df, origin)

    print("\n=== Site-wise imputation ===")
    sitewise_frames = []
    simple_frames = []
    for origin, df in cleaned_sites.items():
        sitewise_frames.append(impute_site_knn(df, origin, k=5))
        simple_frames.append(impute_site_simple(df, origin))

    df_sitewise = finalize_dataset(sitewise_frames)
    df_simple = finalize_dataset(simple_frames)

    df_sitewise.to_csv("heart_imputed_sitewise.csv", index=False)
    df_simple.to_csv("heart_imputed_simple.csv", index=False)
    print("\nSaved: heart_imputed_sitewise.csv")
    print("Saved: heart_imputed_simple.csv")

    print("\nRows per origin (site-wise KNN result):")
    print(df_sitewise["origin"].value_counts())

    print("\nRemaining NaNs per column (site-wise KNN result):")
    print(df_sitewise.isna().sum().sort_values(ascending=False).head(12))

    print_basic_statistics(df_sitewise)
    print_stability_tables(df_sitewise, df_simple)
    generalization_demo(df_simple)

    print("\nPreparation complete. You can now run streamlit run app.py")


if __name__ == "__main__":
    main()
