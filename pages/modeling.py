import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
)
from analysis import get_raw_data

st.title("Model Development")
st.markdown("*Can one model work for all hospitals, or do we need separate models?*")

st.caption(
    ":material/info: This page uses all data (ignoring sidebar filters) to ensure proper train/test splitting."
)

with st.expander("About the Models", icon=":material/info:"):
    st.markdown("""
    We compare **four approaches** to predict heart disease:
    
    | Model | What it does |
    |-------|-------------|
    | **Global Logistic Regression** | One linear model for ALL patients. Assumes the same patterns work everywhere. |
    | **Global Decision Tree** | One tree model for ALL patients. Can find non-linear patterns like "if age > 55 AND chest pain = 4 → high risk". |
    | **Stratified Logistic Regression** | Four SEPARATE linear models, one per hospital. Each patient uses their own hospital's model. |
    | **Stratified Decision Tree** | Four SEPARATE tree models, one per hospital. |
    
    **The question:** Do hospitals share the same disease patterns, or are they too different?
    """)


# ========== HELPER FUNCTIONS ==========


def get_unreliable_features(df: pd.DataFrame, threshold: float = 0.5) -> list:
    """
    Dynamically identify features where missingness exceeds threshold in ANY origin.
    Note: "Fake zeros" (like chol in Switzerland) are converted to NaN in analysis.py
    """
    unreliable = set()

    for origin in df["origin"].unique():
        origin_data = df[df["origin"] == origin]
        missing_rates = (
            origin_data.drop(columns=["origin", "target", "num"], errors="ignore")
            .isnull()
            .mean()
        )

        high_missing = missing_rates[missing_rates > threshold].index.tolist()
        unreliable.update(high_missing)

    return sorted(list(unreliable))


def preprocess_origin_aware(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    origins_train: pd.Series,
    origins_test: pd.Series,
    n_neighbors: int = 5,
) -> tuple:
    """
    Apply KNN imputation per-origin to avoid mixing hospital distributions.
    Fit on training data only, transform both train and test.
    """
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()

    imputers = {}

    for origin in origins_train.unique():
        mask_train = origins_train == origin
        X_origin = X_train[mask_train]

        if len(X_origin) > n_neighbors:
            imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
            imputer.fit(X_origin)
            imputers[origin] = imputer
            X_train_imputed.loc[mask_train, :] = imputer.transform(X_origin)

    for origin in origins_test.unique():
        mask_test = origins_test == origin
        X_origin_test = X_test[mask_test]

        if origin in imputers and len(X_origin_test) > 0:
            X_test_imputed.loc[mask_test, :] = imputers[origin].transform(X_origin_test)
        elif len(X_origin_test) > 0:
            global_imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
            global_imputer.fit(X_train)
            X_test_imputed.loc[mask_test, :] = global_imputer.transform(X_origin_test)

    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang"]
    for col in categorical_cols:
        if col in X_train_imputed.columns:
            X_train_imputed[col] = X_train_imputed[col].round().astype(int)
            X_test_imputed[col] = X_test_imputed[col].round().astype(int)

    return X_train_imputed, X_test_imputed


def preprocess_mice_origin_aware(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    origins_train: pd.Series,
    origins_test: pd.Series,
    max_iter: int = 10,
    random_state: int = 42,
) -> tuple:
    """
    Apply MICE (IterativeImputer) per-origin to avoid mixing hospital distributions.
    Fit on training data only, transform both train and test.
    """
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()

    imputers = {}

    for origin in origins_train.unique():
        mask_train = origins_train == origin
        X_origin = X_train[mask_train]

        if len(X_origin) > 5:  # Need enough samples for MICE
            imputer = IterativeImputer(
                max_iter=max_iter,
                random_state=random_state,
                initial_strategy="mean",
            )
            imputer.fit(X_origin)
            imputers[origin] = imputer
            X_train_imputed.loc[mask_train, :] = imputer.transform(X_origin)

    for origin in origins_test.unique():
        mask_test = origins_test == origin
        X_origin_test = X_test[mask_test]

        if origin in imputers and len(X_origin_test) > 0:
            X_test_imputed.loc[mask_test, :] = imputers[origin].transform(X_origin_test)
        elif len(X_origin_test) > 0:
            # Fallback to global imputer
            global_imputer = IterativeImputer(
                max_iter=max_iter,
                random_state=random_state,
                initial_strategy="mean",
            )
            global_imputer.fit(X_train)
            X_test_imputed.loc[mask_test, :] = global_imputer.transform(X_origin_test)

    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang"]
    for col in categorical_cols:
        if col in X_train_imputed.columns:
            X_train_imputed[col] = X_train_imputed[col].round().astype(int)
            X_test_imputed[col] = X_test_imputed[col].round().astype(int)

    return X_train_imputed, X_test_imputed


# ========== LOAD RAW DATA ==========
df_raw = get_raw_data()

# Sidebar configuration
st.sidebar.markdown("### :material/tune: **Model Configuration**")
st.sidebar.divider()

missing_threshold = st.sidebar.slider(
    "Missingness Threshold",
    0.1,
    0.9,
    0.5,
    0.1,
    help="Drop features with >X% missing in any origin",
)

unreliable_features = get_unreliable_features(df_raw, threshold=missing_threshold)

test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)

imputation_method = st.sidebar.radio(
    "Imputation Method",
    ["KNN", "MICE"],
    index=1,  # Default to MICE
    help="KNN: K-Nearest Neighbors | MICE: Multiple Imputation by Chained Equations",
)

if imputation_method == "KNN":
    n_neighbors = st.sidebar.slider("KNN Neighbors", 3, 10, 5, 1)
    mice_max_iter = 10  # default, not used
else:
    mice_max_iter = st.sidebar.slider("MICE Max Iterations", 5, 20, 10, 1)
    n_neighbors = 5  # default, not used

random_state = 42

if unreliable_features:
    with st.sidebar.expander("Unreliable Features"):
        for feat in unreliable_features:
            max_missing = 0
            worst_origin = ""
            for origin in df_raw["origin"].unique():
                origin_data = df_raw[df_raw["origin"] == origin]
                if feat in origin_data.columns:
                    missing_rate = origin_data[feat].isnull().mean()
                    if missing_rate > max_missing:
                        max_missing = missing_rate
                        worst_origin = origin
            st.caption(f"**{feat}**: {max_missing:.0%} missing in {worst_origin}")

all_features = [
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
]
target_col = "target"

# Always drop unreliable features (>threshold missing in any origin)
feature_cols = [f for f in all_features if f not in unreliable_features]

available_features = [c for c in feature_cols if c in df_raw.columns]

# ========== PREPARE DATA ==========
X = df_raw[available_features].copy()
y = df_raw[target_col].copy()
origins = df_raw["origin"].copy()

valid_mask = y.notna()
X = X[valid_mask]
y = y[valid_mask]
origins = origins[valid_mask]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Features", len(available_features))
col2.metric("Raw Samples", len(X))
col3.metric("Test Size", f"{test_size:.0%}")
col4.metric("Origins", origins.nunique())

# ========== TRAIN/TEST SPLIT FIRST ==========
X_train, X_test, y_train, y_test, origins_train, origins_test = train_test_split(
    X, y, origins, test_size=test_size, random_state=random_state, stratify=y
)

st.divider()

# ========== IMPUTATION ==========
st.subheader("Data Preprocessing")
st.caption(
    f"Missing values filled using {imputation_method}, separately for each hospital"
)

with st.expander("Why split before imputation?", icon=":material/help:"):
    st.markdown("""
    **Preventing Data Leakage:**
    
    We split into train/test sets **BEFORE** filling in missing values. Why?
    
    - If we imputed first, test patients' missing values would be filled using training data
    - This "leaks" training information into the test set
    - The model would appear better than it actually is
    
    **Our approach:** Fit the imputer on training data only, then apply it to both sets.
    This keeps the test set truly "unseen" for fair evaluation.
    """)

with st.spinner(f"Applying origin-aware {imputation_method} imputation..."):
    if imputation_method == "KNN":
        X_train_imputed, X_test_imputed = preprocess_origin_aware(
            X_train,
            X_test,
            y_train,
            origins_train,
            origins_test,
            n_neighbors=n_neighbors,
        )
    else:  # MICE
        X_train_imputed, X_test_imputed = preprocess_mice_origin_aware(
            X_train,
            X_test,
            y_train,
            origins_train,
            origins_test,
            max_iter=mice_max_iter,
            random_state=random_state,
        )

train_nans = X_train_imputed.isnull().sum().sum()
test_nans = X_test_imputed.isnull().sum().sum()

col_imp1, col_imp2, col_imp3 = st.columns(3)
col_imp1.metric("Training Samples", len(X_train_imputed))
col_imp2.metric("Test Samples", len(X_test_imputed))
col_imp3.metric("Remaining NaNs", train_nans + test_nans)

if train_nans + test_nans > 0:
    st.warning(
        f"{train_nans + test_nans} NaN values remain. Dropping affected rows.",
        icon=":material/warning:",
    )
    train_valid = X_train_imputed.notna().all(axis=1)
    test_valid = X_test_imputed.notna().all(axis=1)

    X_train_imputed = X_train_imputed[train_valid]
    y_train = y_train[train_valid]
    origins_train = origins_train[train_valid]

    X_test_imputed = X_test_imputed[test_valid]
    y_test = y_test[test_valid]
    origins_test = origins_test[test_valid]

st.divider()

st.subheader("Model Training")
st.caption("Training one global model vs. separate models for each hospital")

tab_global, tab_stratified = st.tabs(["Global Models", "Origin-Stratified"])

with tab_global:
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("**Logistic Regression**")
        global_lr = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=random_state),
                ),
            ]
        )
        global_lr.fit(X_train_imputed, y_train)
        st.caption("StandardScaler + Logistic Regression")

    with col_m2:
        st.markdown("**Decision Tree**")
        global_dt = DecisionTreeClassifier(max_depth=5, random_state=random_state)
        global_dt.fit(X_train_imputed, y_train)
        st.caption("Max Depth = 5")

    st.metric(
        "Training Samples",
        f"{X_train_imputed.shape[0]:,}",
        delta="all origins combined",
        delta_color="off",
    )

with tab_stratified:
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("**Logistic Regression per Institution**")
        st.caption("Training independent LR models for each origin...")

        origin_models_lr = {}
        origin_stats = []

        for origin in sorted(origins_train.unique()):
            origin_mask_train = origins_train == origin
            X_origin_train = X_train_imputed[origin_mask_train]
            y_origin_train = y_train[origin_mask_train]

            if len(X_origin_train) > 10:
                model = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "classifier",
                            LogisticRegression(
                                max_iter=1000, random_state=random_state
                            ),
                        ),
                    ]
                )
                model.fit(X_origin_train, y_origin_train)
                origin_models_lr[origin] = model
                origin_stats.append(
                    {"Origin": origin, "Training Samples": len(X_origin_train)}
                )

        stats_df = pd.DataFrame(origin_stats)
        for idx, row in stats_df.iterrows():
            st.caption(f"{row['Origin']}: {row['Training Samples']} samples")

    with col_s2:
        st.markdown("**Decision Tree per Institution**")
        st.caption("Training independent DT models for each origin...")

        origin_models_dt = {}

        for origin in sorted(origins_train.unique()):
            origin_mask_train = origins_train == origin
            X_origin_train = X_train_imputed[origin_mask_train]
            y_origin_train = y_train[origin_mask_train]

            if len(X_origin_train) > 10:
                model = DecisionTreeClassifier(max_depth=5, random_state=random_state)
                model.fit(X_origin_train, y_origin_train)
                origin_models_dt[origin] = model

        for origin in sorted(origin_models_dt.keys()):
            st.caption(f"{origin}: Max Depth = 5")

st.divider()

st.subheader("Performance Comparison")
st.caption("How well does each model predict on unseen test data?")

y_pred_lr = global_lr.predict(X_test_imputed)
y_prob_lr = global_lr.predict_proba(X_test_imputed)[:, 1]

y_pred_dt = global_dt.predict(X_test_imputed)
y_prob_dt = global_dt.predict_proba(X_test_imputed)[:, 1]

# Stratified Logistic Regression predictions
y_pred_stratified_lr = []
y_prob_stratified_lr = []

for idx in range(len(X_test_imputed)):
    origin = origins_test.iloc[idx]
    X_sample = X_test_imputed.iloc[[idx]]

    if origin in origin_models_lr:
        pred = origin_models_lr[origin].predict(X_sample)[0]
        prob = origin_models_lr[origin].predict_proba(X_sample)[0, 1]
    else:
        pred = global_lr.predict(X_sample)[0]
        prob = global_lr.predict_proba(X_sample)[0, 1]

    y_pred_stratified_lr.append(pred)
    y_prob_stratified_lr.append(prob)

y_pred_stratified_lr = pd.Series(y_pred_stratified_lr, index=X_test_imputed.index)
y_prob_stratified_lr = pd.Series(y_prob_stratified_lr, index=X_test_imputed.index)

# Stratified Decision Tree predictions
y_pred_stratified_dt = []
y_prob_stratified_dt = []

for idx in range(len(X_test_imputed)):
    origin = origins_test.iloc[idx]
    X_sample = X_test_imputed.iloc[[idx]]

    if origin in origin_models_dt:
        pred = origin_models_dt[origin].predict(X_sample)[0]
        prob = origin_models_dt[origin].predict_proba(X_sample)[0, 1]
    else:
        pred = global_dt.predict(X_sample)[0]
        prob = global_dt.predict_proba(X_sample)[0, 1]

    y_pred_stratified_dt.append(pred)
    y_prob_stratified_dt.append(prob)

y_pred_stratified_dt = pd.Series(y_pred_stratified_dt, index=X_test_imputed.index)
y_prob_stratified_dt = pd.Series(y_prob_stratified_dt, index=X_test_imputed.index)


def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
    }


metrics_df = pd.DataFrame(
    {
        "Global Logistic": get_metrics(y_test, y_pred_lr),
        "Global Decision Tree": get_metrics(y_test, y_pred_dt),
        "Stratified Logistic": get_metrics(y_test, y_pred_stratified_lr),
        "Stratified Decision Tree": get_metrics(y_test, y_pred_stratified_dt),
    }
).T

st.dataframe(
    metrics_df.style.format("{:.3f}").highlight_max(axis=0, color="darkgreen"),
    use_container_width=True,
)

st.divider()

st.subheader("ROC Curves")
st.caption(
    "How well can each model distinguish between disease and no-disease? Higher AUC = better."
)

roc_data = []
models_roc = [
    (y_prob_lr, "Global Logistic"),
    (y_prob_dt, "Global Decision Tree"),
    (y_prob_stratified_lr, "Stratified Logistic"),
    (y_prob_stratified_dt, "Stratified Decision Tree"),
]

for probs, name in models_roc:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_score = auc(fpr, tpr)
    temp_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    temp_df["Model"] = f"{name} (AUC = {auc_score:.3f})"
    roc_data.append(temp_df)

roc_df = pd.concat(roc_data)

roc_chart = (
    alt.Chart(roc_df)
    .mark_line(size=3)
    .encode(
        x=alt.X("FPR:Q", title="False Positive Rate"),
        y=alt.Y("TPR:Q", title="True Positive Rate"),
        color=alt.Color("Model:N", scale=alt.Scale(scheme="category10")),
    )
    .properties(height=400)
)

diagonal = (
    alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
    .mark_line(color="gray", strokeDash=[5, 5])
    .encode(x="x", y="y")
)

st.altair_chart(roc_chart + diagonal, use_container_width=True)

st.divider()

st.subheader("Performance by Origin")
st.caption("Does the model work equally well for all hospitals, or does it favor some?")

origin_comparison = []

for origin in sorted(origins_test.unique()):
    mask = origins_test == origin
    if mask.sum() > 0:
        X_sub = X_test_imputed[mask]
        y_sub = y_test[mask]

        acc_lr = accuracy_score(y_sub, global_lr.predict(X_sub))
        acc_dt = accuracy_score(y_sub, global_dt.predict(X_sub))
        y_pred_strat_lr_sub = y_pred_stratified_lr[mask]
        y_pred_strat_dt_sub = y_pred_stratified_dt[mask]
        acc_strat_lr = accuracy_score(y_sub, y_pred_strat_lr_sub)
        acc_strat_dt = accuracy_score(y_sub, y_pred_strat_dt_sub)

        origin_comparison.append(
            {
                "Origin": origin,
                "Samples": mask.sum(),
                "Global LR": acc_lr,
                "Global DT": acc_dt,
                "Stratified LR": acc_strat_lr,
                "Stratified DT": acc_strat_dt,
            }
        )

comp_df = pd.DataFrame(origin_comparison)

comp_long = comp_df.melt(
    id_vars=["Origin", "Samples"],
    value_vars=["Global LR", "Global DT", "Stratified LR", "Stratified DT"],
    var_name="Model",
    value_name="Accuracy",
)

chart = (
    alt.Chart(comp_long)
    .mark_bar()
    .encode(
        x=alt.X("Origin:N", title="Origin", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "Model:N",
            scale=alt.Scale(scheme="category10"),
            legend=alt.Legend(title="Model"),
        ),
        xOffset="Model:N",
        tooltip=["Origin", "Model", "Samples", alt.Tooltip("Accuracy", format=".1%")],
    )
    .properties(height=400)
)

st.altair_chart(chart, use_container_width=True)

st.divider()

st.subheader("Cross-Validation Results")
st.caption(
    "Testing model stability by training on 5 different train/test splits. Low Std Dev = consistent model."
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

cv_scores_lr = cross_val_score(
    global_lr, X_train_imputed, y_train, cv=cv, scoring="accuracy"
)
cv_scores_dt = cross_val_score(
    global_dt, X_train_imputed, y_train, cv=cv, scoring="accuracy"
)

cv_results = pd.DataFrame(
    {
        "Model": ["Global Logistic Regression", "Global Decision Tree"],
        "Mean CV Accuracy": [cv_scores_lr.mean(), cv_scores_dt.mean()],
        "Std Dev": [cv_scores_lr.std(), cv_scores_dt.std()],
        "Min": [cv_scores_lr.min(), cv_scores_dt.min()],
        "Max": [cv_scores_lr.max(), cv_scores_dt.max()],
    }
)

st.dataframe(
    cv_results.style.format(
        {
            "Mean CV Accuracy": "{:.3f}",
            "Std Dev": "{:.3f}",
            "Min": "{:.3f}",
            "Max": "{:.3f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
    height=108,
)

st.divider()

st.subheader("Confusion Matrices")
st.caption(
    "Where did each model make mistakes? Bottom-left (missed disease cases) is the most dangerous error."
)


# Build confusion matrix data for all three models
def get_cm_data(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    return [
        {
            "Model": model_name,
            "Actual": "No Disease",
            "Predicted": "No",
            "Count": int(cm[0, 0]),
        },
        {
            "Model": model_name,
            "Actual": "No Disease",
            "Predicted": "Yes",
            "Count": int(cm[0, 1]),
        },
        {
            "Model": model_name,
            "Actual": "Disease",
            "Predicted": "No",
            "Count": int(cm[1, 0]),
        },
        {
            "Model": model_name,
            "Actual": "Disease",
            "Predicted": "Yes",
            "Count": int(cm[1, 1]),
        },
    ]


cm_data = (
    get_cm_data(y_test, y_pred_lr, "Global Logistic")
    + get_cm_data(y_test, y_pred_dt, "Global Decision Tree")
    + get_cm_data(y_test, np.array(y_pred_stratified_lr), "Stratified Logistic")
    + get_cm_data(y_test, np.array(y_pred_stratified_dt), "Stratified Decision Tree")
)
cm_df = pd.DataFrame(cm_data)

cm_chart = (
    alt.Chart(cm_df)
    .mark_rect()
    .encode(
        x=alt.X("Predicted:N", title="Predicted", sort=["No", "Yes"]),
        y=alt.Y("Actual:N", title="Actual", sort=["No Disease", "Disease"]),
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["Model", "Actual", "Predicted", "Count"],
    )
)

cm_text = (
    alt.Chart(cm_df)
    .mark_text(fontSize=16, fontWeight="bold")
    .encode(
        x=alt.X("Predicted:N", sort=["No", "Yes"]),
        y=alt.Y("Actual:N", sort=["No Disease", "Disease"]),
        text="Count:Q",
        color=alt.condition(
            alt.datum.Count > 30, alt.value("white"), alt.value("black")
        ),
    )
)

cm_combined = (
    (cm_chart + cm_text)
    .properties(
        width=150,
        height=150,
    )
    .facet(
        column=alt.Column(
            "Model:N",
            title=None,
            sort=["Global Logistic", "Decision Tree", "Origin-Stratified"],
        )
    )
)

st.altair_chart(cm_combined, use_container_width=True)

st.divider()

st.subheader("Feature Importance")

st.markdown("**Decision Tree Feature Importance**")
st.caption(
    "Which features did the tree use most to make splits? Taller bars = more important for predictions."
)

dt_importance = pd.DataFrame(
    {"Feature": available_features, "Importance": global_dt.feature_importances_}
).sort_values("Importance", ascending=False)

importance_chart = (
    alt.Chart(dt_importance)
    .mark_bar()
    .encode(
        x=alt.X("Importance:Q", title="Importance"),
        y=alt.Y("Feature:N", sort="-x", title="Feature"),
        color=alt.Color("Importance:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f")],
    )
    .properties(height=400)
)

st.altair_chart(importance_chart, use_container_width=True)

st.markdown("**Logistic Regression Feature Influence**")
st.caption(
    "Taller bars = feature has more impact on the prediction. "
    "These show how strongly each feature affects the model's decision."
)

lr_model = global_lr.named_steps["classifier"]
lr_coefs = pd.DataFrame(
    {"Feature": available_features, "Influence": np.abs(lr_model.coef_[0])}
).sort_values("Influence", ascending=False)

coef_chart = (
    alt.Chart(lr_coefs)
    .mark_bar()
    .encode(
        x=alt.X("Influence:Q", title="Influence Strength"),
        y=alt.Y("Feature:N", sort="-x", title="Feature"),
        color=alt.Color("Influence:Q", scale=alt.Scale(scheme="oranges"), legend=None),
        tooltip=["Feature", alt.Tooltip("Influence:Q", format=".3f")],
    )
    .properties(height=400)
)

st.altair_chart(coef_chart, use_container_width=True)

st.divider()

st.subheader("Model Comparison Summary")
st.caption("Which model performed best overall?")

acc_global_lr = metrics_df.loc["Global Logistic", "Accuracy"]
acc_global_dt = metrics_df.loc["Global Decision Tree", "Accuracy"]
acc_stratified_lr = metrics_df.loc["Stratified Logistic", "Accuracy"]
acc_stratified_dt = metrics_df.loc["Stratified Decision Tree", "Accuracy"]

# Create a simple comparison table
comparison_data = pd.DataFrame(
    {
        "Model": [
            "Global Logistic Regression",
            "Global Decision Tree",
            "Stratified Logistic Regression",
            "Stratified Decision Tree",
        ],
        "Accuracy": [
            acc_global_lr,
            acc_global_dt,
            acc_stratified_lr,
            acc_stratified_dt,
        ],
    }
).sort_values("Accuracy", ascending=False)

comparison_data["Rank"] = ["1st", "2nd", "3rd", "4th"][: len(comparison_data)]
comparison_data = comparison_data[["Rank", "Model", "Accuracy"]]

st.dataframe(
    comparison_data.style.format({"Accuracy": "{:.1%}"}).hide(axis="index"),
    use_container_width=True,
    hide_index=True,
)

st.divider()

best_model = metrics_df["Accuracy"].idxmax()
acc_diff = acc_stratified_lr - acc_global_lr

st.success(f"### :material/workspace_premium: Best Performing Approach: {best_model}")

# Check Switzerland performance for insights
switzerland_insight = ""
for row in origin_comparison:
    if row["Origin"] == "Switzerland":
        swiss_strat = row.get("Stratified LR", 0)
        swiss_global = row.get("Global LR", 0)
        if swiss_strat > swiss_global + 0.1:  # >10% better
            switzerland_insight = f"""
    
    **Notable:** Switzerland shows a large improvement with the stratified model 
    ({swiss_strat:.0%} vs {swiss_global:.0%}), suggesting its patient population 
    may have different characteristics than other hospitals.
    """

if abs(acc_diff) < 0.03:
    st.markdown(f"""
    **Key Finding:** The stratified and global models perform **similarly overall**.
    
    - Institutional differences were primarily in **data collection quality**, not disease patterns
    - We dropped {len(unreliable_features)} features with >{missing_threshold:.0%} missing data
    - The remaining {len(available_features)} features work consistently across hospitals
    {switzerland_insight}
    """)
elif best_model == "Origin-Stratified":
    st.markdown(f"""
    **Key Finding:** Hospital-specific models work better than a single global model.
    
    - This suggests real differences in patient populations across institutions
    - Each hospital may have different risk factors or diagnostic patterns
    {switzerland_insight}
    """)
elif best_model == "Global Decision Tree":
    st.markdown(f"""
    **Key Finding:** The Decision Tree outperformed other approaches.
    
    - This suggests **non-linear patterns** in the data (e.g., interactions between features)
    - A tree can capture rules like "if chest pain = asymptomatic AND age > 55 → high risk"
    {switzerland_insight}
    """)
else:
    st.markdown(f"""
    **Key Finding:** A simple Logistic Regression works well across all hospitals.
    
    - The core clinical features (chest pain, max heart rate, ST depression) are universal
    - No need for complex models or hospital-specific approaches
    {switzerland_insight}
    """)

st.divider()

with st.expander(":material/checklist: Pipeline Summary", expanded=False):
    imputation_detail = (
        f"KNN (k={n_neighbors})"
        if imputation_method == "KNN"
        else f"MICE (max_iter={mice_max_iter})"
    )
    st.markdown(f"""
    **Data Preprocessing Pipeline (No Leakage):**
    
    1. **Load raw data** - No pre-imputation
    2. **Dynamic feature selection** - Dropped features with >{missing_threshold:.0%} missing
    3. **Train/test split FIRST** - {1 - test_size:.0%}/{test_size:.0%} split
    4. **Origin-aware {imputation_method} imputation** - Fitted on training only
    5. **Model training** - StandardScaler + Classifiers
    6. **Evaluation** - On held-out test set
    
    **Imputation:** {imputation_detail} per-origin
    """)

st.info(
    "**Why this matters:** We split the data before imputing missing values, so the test set stays truly 'unseen'. "
    "Each hospital's missing values are filled using only that hospital's patterns.",
    icon=":material/info:",
)
