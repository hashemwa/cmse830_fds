import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(
    page_title="Heart Disease Sites Explorer",
    layout="wide",
    page_icon="❤️",
)

CAT_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
DISPLAY_NUMERICS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
DATA_FILES = {
    "Site-wise KNN": "heart_imputed_sitewise.csv",
    "Simple Per-Site Median": "heart_imputed_simple.csv",
}

sns.set_theme(style="whitegrid", palette="deep")


def format_dataframe(df):
    """Coerce data types for consistent downstream use."""
    formatted = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in formatted.columns:
            formatted[col] = pd.to_numeric(formatted[col], errors="coerce")
    for col in CAT_COLUMNS:
        if col in formatted.columns:
            formatted[col] = pd.to_numeric(formatted[col], errors="coerce").astype(
                "Int64"
            )
    if "num" in formatted.columns:
        formatted["num"] = pd.to_numeric(
            formatted["num"], errors="coerce"
        ).astype("Int64")
    if "target" in formatted.columns:
        formatted["target"] = pd.to_numeric(
            formatted["target"], errors="coerce"
        ).astype("Int64")
    if "origin" in formatted.columns:
        formatted["origin"] = formatted["origin"].astype("category")
    return formatted


@st.cache_data(show_spinner=False)
def load_all_datasets():
    """Load both imputed datasets; warn later if a file is missing."""
    datasets = {}
    for label, path in DATA_FILES.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            datasets[label] = format_dataframe(df)
    return datasets


def filter_data(df, selected_origins, age_range):
    """Apply origin and age filters, returning a copy to avoid SettingWithCopy warnings."""
    filtered = df[df["origin"].isin(selected_origins)].copy()
    filtered = filtered[
        (filtered["age"].fillna(age_range[0]) >= age_range[0])
        & (filtered["age"].fillna(age_range[1]) <= age_range[1])
    ]
    return filtered


def show_boxplot(df, value_col, ylabel, title, subtitle):
    if df.empty or df[value_col].dropna().empty:
        st.info(f"Not enough data to draw {value_col} boxplot.")
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(data=df, x="target", y=value_col, ax=ax)
    ax.set_xlabel("Disease Present (target)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(subtitle)


def show_countplot(df, column, title, subtitle):
    if df.empty:
        st.info("Not enough data to draw the count plot.")
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x=column, hue="target", ax=ax)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(subtitle)


def show_scatter(df):
    if df.empty:
        st.info("Not enough data to draw the interaction scatter.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=df,
        x="oldpeak",
        y="thalach",
        hue="target",
        style="origin",
        ax=ax,
        alpha=0.75,
    )
    ax.set_xlabel("ST depression induced by exercise (oldpeak)")
    ax.set_ylabel("Max heart rate achieved (thalach)")
    ax.set_title("oldpeak vs thalach by origin and disease status")
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Switzerland points group differently than Cleveland; elevation in oldpeak relates to disease more strongly in Cleveland, reinforcing that a single threshold does not suit every site."
    )


def show_missingness_heatmap(df):
    if df.empty:
        st.info("Not enough data to show missingness patterns.")
        return
    heatmap_cols = ["chol", "ca", "thal", "origin", "target"]
    available_cols = [c for c in heatmap_cols if c in df.columns]
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(
        df[available_cols].isna(),
        cmap="rocket_r",
        cbar=False,
        ax=ax,
    )
    ax.set_title("Missingness pattern (1 = missing)")
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Switzerland retains missing cholesterol even after site-wise KNN. The structured gap highlights why cross-site averages can be misleading."
    )


def show_stats_and_balance(df, show_chol):
    if df.empty:
        st.info("No rows left after filtering. Adjust selections to see statistics.")
        return
    numeric_cols = DISPLAY_NUMERICS.copy()
    if show_chol is False and "chol" in numeric_cols:
        numeric_cols.remove("chol")
    stats_df = pd.DataFrame(
        {
            "mean": df[numeric_cols].mean(),
            "median": df[numeric_cols].median(),
            "std": df[numeric_cols].std(),
        }
    ).round(2)
    st.subheader("Summary statistics for selected sites")
    st.dataframe(stats_df)
    if show_chol:
        st.caption(
            "Means and medians shift by origin; Cleveland trends toward higher cholesterol while other sites show lower but more missing values, underscoring the one-size-fits-all warning."
        )
    else:
        st.caption(
            "With cholesterol hidden, the remaining vitals still show site-by-site differences—evidence that we need tailored preprocessing for each origin."
        )

    st.subheader("Class balance by origin")
    balance = (
        df.groupby(["origin", "target"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "No disease", 1: "Disease"})
    )
    balance["Total"] = balance.sum(axis=1)
    if "Disease" in balance.columns:
        balance["% Disease"] = (
            (balance["Disease"] / balance["Total"]).replace([np.inf, np.nan], 0) * 100
        ).round(1)
    st.dataframe(balance)
    st.caption(
        "Disease prevalence varies sharply: Cleveland has the richest positive class, while other sites drop in balance, so training on Cleveland alone overestimates real-world performance."
    )


def show_stability_snapshot(datasets):
    if len(datasets) < 2:
        return
    st.subheader("Stability snapshot across imputers")
    metrics = ["oldpeak", "thalach", "ca"]
    frames = []
    for label, df in datasets.items():
        if not set(metrics).issubset(df.columns):
            continue
        summary = df.groupby("origin")[metrics].mean().round(2)
        summary["Imputation"] = label
        frames.append(summary.reset_index())
    if not frames:
        return
    stability = pd.concat(frames, ignore_index=True)
    stability = stability.pivot_table(
        index="origin", columns="Imputation", values=metrics
    )
    st.dataframe(stability)
    st.caption(
        "Site-wise KNN preserves Switzerland’s missing cholesterol while keeping oldpeak and ca stable; the simple median smoother shrinks site differences, masking the structural gaps."
    )


@st.cache_data(show_spinner=False)
def compute_generalization_table(df_simple):
    if df_simple.empty:
        return pd.DataFrame()
    feature_cols = CAT_COLUMNS + NUMERIC_COLUMNS
    cle_mask = df_simple["origin"] == "Cleveland"
    X_train = df_simple.loc[cle_mask, feature_cols].astype(float)
    y_train = df_simple.loc[cle_mask, "target"].astype(int)
    if X_train.empty:
        return pd.DataFrame()
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_train, y_train)

    rows = []
    for origin in df_simple["origin"].unique():
        if origin == "Cleveland":
            continue
        mask = df_simple["origin"] == origin
        X_test = df_simple.loc[mask, feature_cols].astype(float)
        y_test = df_simple.loc[mask, "target"].astype(int)
        if X_test.empty:
            continue
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = np.nan
        rows.append(
            {
                "Test site": origin,
                "Accuracy": round(accuracy, 3),
                "AUC": np.nan if np.isnan(auc) else round(float(auc), 3),
            }
        )
    return pd.DataFrame(rows)


def main():
    datasets = load_all_datasets()
    if not datasets:
        st.error(
            "Processed files not found. Please run `python prepare.py` to generate the imputed datasets."
        )
        return

    default_dataset = datasets.get("Site-wise KNN")
    all_origins = sorted(default_dataset["origin"].cat.categories.tolist())

    st.sidebar.header("Filter the view")
    imputation_choice = st.sidebar.radio(
        "Imputation strategy",
        list(DATA_FILES.keys()),
        index=0,
    )
    selected_origins = st.sidebar.multiselect(
        "Origin(s)",
        all_origins,
        default=all_origins,
    )
    age_min = int(default_dataset["age"].min(skipna=True))
    age_max = int(default_dataset["age"].max(skipna=True))
    age_selection = st.sidebar.slider(
        "Age range",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max),
    )
    hide_cholesterol = st.sidebar.checkbox(
        "Hide cholesterol in summary tables",
        value=False,
        help="Use this when focusing on Switzerland where cholesterol is structurally missing.",
    )

    st.title("Heart Disease Sites: Why One-Size-Fits-All Fails")
    st.markdown(
        "Cleveland data is dense and drives most historical benchmarks, yet Hungary, Switzerland, and Long Beach VA show structured gaps—especially Switzerland’s cholesterol. "
        "Use the sidebar to compare sites and see why pooling everything into a single recipe can backfire."
    )

    current_df = datasets.get(imputation_choice)
    if current_df is None:
        st.error(
            f"{DATA_FILES[imputation_choice]} was not found. Re-run `python prepare.py`."
        )
        return

    filtered_df = filter_data(current_df, selected_origins, age_selection)

    if filtered_df.empty:
        st.warning(
            "No rows match your filters. Try adding more origins or widening the age range."
        )
        return

    with st.expander("Quick dataset preview", expanded=False):
        st.write(filtered_df.head())

    st.subheader("Disease-linked distributions")
    col1, col2 = st.columns(2)
    with col1:
        show_boxplot(
            filtered_df,
            "oldpeak",
            "ST depression (oldpeak)",
            "oldpeak vs disease status",
            "Higher oldpeak values align with confirmed disease in Cleveland, but other sites show flatter patterns.",
        )
    with col2:
        show_boxplot(
            filtered_df,
            "thalach",
            "Max heart rate (thalach)",
            "thalach vs disease status",
            "Lower thalach combines with disease more often outside Cleveland—cardio capacity drops faster at other sites.",
        )

    show_countplot(
        filtered_df,
        "cp",
        "Chest pain types by disease status",
        "Chest pain encoding skews positive in Cleveland; other sites report more atypical pain even without disease.",
    )

    show_scatter(filtered_df)
    show_missingness_heatmap(current_df)

    show_stats_and_balance(filtered_df, not hide_cholesterol)
    show_stability_snapshot(datasets)

    simple_df = datasets.get("Simple Per-Site Median", pd.DataFrame())
    generalization_table = compute_generalization_table(simple_df)
    if not generalization_table.empty:
        st.subheader("Generalization gap (train on Cleveland, test elsewhere)")
        st.dataframe(generalization_table)
        st.caption(
            "Model performance drops when evaluated on other sites, especially Switzerland and Hungary, reinforcing that tuning to Cleveland alone overstates expected accuracy."
        )
    else:
        st.info(
            "Generalization demo skipped (simple median dataset missing or lacks enough data)."
        )

    st.markdown(
        "**Takeaway:** Site-wise decisions about missing data and feature scaling matter. Cleveland and Switzerland behave differently, so a single preprocessing recipe does not generalize."
    )


if __name__ == "__main__":
    main()
