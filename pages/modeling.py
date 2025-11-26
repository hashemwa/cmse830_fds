import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
)
from analysis import get_clean_data

st.title("Model Development")
st.markdown("*Testing the 'One Size Fits All' Hypothesis*")

# Load data
df = get_clean_data()

# Features with high missingness that should be dropped
HIGH_MISSING_FEATURES = ["ca", "thal", "slope", "fbs", "chol"]

# Define features - only those reliably collected across all origins
feature_cols = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
]
target_col = "target"

# Sidebar configuration
st.sidebar.subheader("Model Configuration")
drop_high_missing = st.sidebar.checkbox(
    "Drop unreliable features",
    value=True,
    help="Removes ca, thal, slope, fbs, chol to avoid imputation artifacts",
)
test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
random_state = 42

if not drop_high_missing:
    feature_cols = feature_cols + HIGH_MISSING_FEATURES

available_features = [c for c in feature_cols if c in df.columns]

# Summary metrics at top
col1, col2, col3, col4 = st.columns(4)
col1.metric("Features", len(available_features))
col2.metric("Samples", len(df))
col3.metric("Test Size", f"{test_size:.0%}")
col4.metric("Origins", df["origin"].nunique())

# Prepare X and y
X = df[available_features].copy()
y = df[target_col].copy()
origins = df["origin"].copy()

# Check for NaNs
if X.isna().any().any():
    st.warning(
        "Data contains missing values. Dropping rows with missing values for modeling."
    )
    X = X.dropna()
    y = y[X.index]
    origins = origins[X.index]

X_train, X_test, y_train, y_test, origins_train, origins_test = train_test_split(
    X, y, origins, test_size=test_size, random_state=random_state, stratify=y
)

st.divider()

st.subheader("Model Training")
st.caption("Comparing global models vs origin-stratified models")

# ========== GLOBAL MODELS ==========
tab_global, tab_stratified = st.tabs(["Global Models", "Origin-Stratified"])

with tab_global:
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("**Logistic Regression**")
        global_lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=random_state),
        )
        global_lr.fit(X_train, y_train)
        st.caption("StandardScaler + Logistic Regression")

    with col_m2:
        st.markdown("**Decision Tree**")
        global_dt = DecisionTreeClassifier(max_depth=5, random_state=random_state)
        global_dt.fit(X_train, y_train)
        st.caption("Max Depth = 5")

    st.metric(
        "Training Samples",
        f"{X_train.shape[0]:,}",
        delta="all origins combined",
        delta_color="off",
    )

with tab_stratified:
    st.markdown("**Separate Logistic Regression per Institution**")
    st.caption("Training independent models for each origin...")

    origin_models = {}
    origin_stats = []

    for origin in sorted(origins_train.unique()):
        origin_mask_train = origins_train == origin
        X_origin_train = X_train[origin_mask_train]
        y_origin_train = y_train[origin_mask_train]

        if len(X_origin_train) > 10:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=random_state),
            )
            model.fit(X_origin_train, y_origin_train)
            origin_models[origin] = model
            origin_stats.append(
                {"Origin": origin, "Training Samples": len(X_origin_train)}
            )

    stats_df = pd.DataFrame(origin_stats)

    cols = st.columns(len(stats_df))
    for idx, row in stats_df.iterrows():
        cols[idx].metric(
            row["Origin"], row["Training Samples"], delta="samples", delta_color="off"
        )

st.divider()

st.subheader("Performance Comparison")
st.caption("Evaluating all three approaches on the held-out test set")

# Predictions
# 1. Global LR
y_pred_lr = global_lr.predict(X_test)
y_prob_lr = global_lr.predict_proba(X_test)[:, 1]

# 2. Global DT
y_pred_dt = global_dt.predict(X_test)
y_prob_dt = global_dt.predict_proba(X_test)[:, 1]

# 3. Stratified
y_pred_stratified = []
y_prob_stratified = []

for idx in range(len(X_test)):
    origin = origins_test.iloc[idx]
    X_sample = X_test.iloc[[idx]]

    if origin in origin_models:
        pred = origin_models[origin].predict(X_sample)[0]
        prob = origin_models[origin].predict_proba(X_sample)[0, 1]
    else:
        # Fallback to global LR
        pred = global_lr.predict(X_sample)[0]
        prob = global_lr.predict_proba(X_sample)[0, 1]

    y_pred_stratified.append(pred)
    y_prob_stratified.append(prob)

y_pred_stratified = pd.Series(y_pred_stratified, index=X_test.index)
y_prob_stratified = pd.Series(y_prob_stratified, index=X_test.index)


# Metrics Helper
def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
    }


metrics_df = pd.DataFrame(
    {
        "Global Logistic": get_metrics(y_test, y_pred_lr),
        "Global Decision Tree": get_metrics(y_test, y_pred_dt),
        "Origin-Stratified": get_metrics(y_test, y_pred_stratified),
    }
).T

st.dataframe(
    metrics_df.style.format("{:.3f}").highlight_max(axis=0, color="lightgreen"),
    use_container_width=True,
)

st.divider()

st.subheader("ROC Curves")
st.caption(
    "Receiver Operating Characteristic curves compare model discrimination ability"
)

roc_data = []
models_roc = [
    (y_prob_lr, "Global Logistic"),
    (y_prob_dt, "Global Decision Tree"),
    (y_prob_stratified, "Origin-Stratified"),
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
st.caption("Breakdown of model accuracy across different data collection sites")

origin_comparison = []

for origin in sorted(origins_test.unique()):
    mask = origins_test == origin
    if mask.sum() > 0:
        X_sub = X_test[mask]
        y_sub = y_test[mask]

        acc_lr = accuracy_score(y_sub, global_lr.predict(X_sub))
        acc_dt = accuracy_score(y_sub, global_dt.predict(X_sub))

        # Stratified predictions for this subset
        y_pred_strat_sub = y_pred_stratified[mask]
        acc_strat = accuracy_score(y_sub, y_pred_strat_sub)

        origin_comparison.append(
            {
                "Origin": origin,
                "Samples": mask.sum(),
                "Global LR": acc_lr,
                "Global DT": acc_dt,
                "Stratified": acc_strat,
            }
        )

comp_df = pd.DataFrame(origin_comparison)

comp_long = comp_df.melt(
    id_vars=["Origin", "Samples"],
    value_vars=["Global LR", "Global DT", "Stratified"],
    var_name="Model",
    value_name="Accuracy",
)

chart = (
    alt.Chart(comp_long)
    .mark_bar()
    .encode(
        x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("Accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Model:N", scale=alt.Scale(scheme="category10")),
        column=alt.Column("Origin:N", title=None, header=alt.Header(labelAngle=0)),
        tooltip=["Origin", "Model", "Samples", alt.Tooltip("Accuracy", format=".1%")],
    )
    .properties(width=100, height=300)
)

st.altair_chart(chart, use_container_width=True)

st.divider()

# ========== CROSS-VALIDATION ==========
st.subheader("Cross-Validation Results")
st.caption(
    "5-fold stratified cross-validation provides more robust accuracy estimates than a single train/test split."
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

cv_scores_lr = cross_val_score(global_lr, X, y, cv=cv, scoring="accuracy")
cv_scores_dt = cross_val_score(global_dt, X, y, cv=cv, scoring="accuracy")

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
)

# ========== CONFUSION MATRICES ==========
st.divider()

st.subheader("Confusion Matrices")
st.caption(
    "Visualize prediction errors: false positives (healthy predicted as sick) vs false negatives (sick predicted as healthy)."
)


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm, index=["No Disease", "Disease"], columns=["Pred: No", "Pred: Yes"]
    )

    cm_long = cm_df.reset_index().melt(
        id_vars="index", var_name="Predicted", value_name="Count"
    )
    cm_long = cm_long.rename(columns={"index": "Actual"})

    chart = (
        alt.Chart(cm_long)
        .mark_rect()
        .encode(
            x=alt.X("Predicted:N", title="Predicted"),
            y=alt.Y("Actual:N", title="Actual"),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Actual", "Predicted", "Count"],
        )
        .properties(title=title, width=180, height=180)
    )

    text = chart.mark_text(baseline="middle", fontSize=18, fontWeight="bold").encode(
        text="Count:Q",
        color=alt.condition(
            alt.datum.Count > cm.max() / 2, alt.value("white"), alt.value("black")
        ),
    )

    return chart + text


col_cm1, col_cm2, col_cm3 = st.columns(3)
with col_cm1:
    st.altair_chart(
        plot_confusion_matrix(y_test, y_pred_lr, "Global Logistic"),
        use_container_width=True,
    )
with col_cm2:
    st.altair_chart(
        plot_confusion_matrix(y_test, y_pred_dt, "Global Decision Tree"),
        use_container_width=True,
    )
with col_cm3:
    st.altair_chart(
        plot_confusion_matrix(y_test, y_pred_stratified, "Origin-Stratified"),
        use_container_width=True,
    )

# ========== FEATURE IMPORTANCE ==========
st.divider()

st.subheader("Feature Importance")
st.caption(
    "Which features matter most for predicting heart disease? Decision Tree importance shows feature contribution."
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
    .properties(height=400, title="Decision Tree Feature Importance")
)

st.altair_chart(importance_chart, use_container_width=True)

# Also show Logistic Regression coefficients
st.caption(
    "Logistic Regression coefficients (absolute values) indicate feature influence on prediction."
)

lr_model = global_lr.named_steps["logisticregression"]
lr_coefs = pd.DataFrame(
    {"Feature": available_features, "Coefficient": np.abs(lr_model.coef_[0])}
).sort_values("Coefficient", ascending=False)

coef_chart = (
    alt.Chart(lr_coefs)
    .mark_bar()
    .encode(
        x=alt.X("Coefficient:Q", title="|Coefficient|"),
        y=alt.Y("Feature:N", sort="-x", title="Feature"),
        color=alt.Color(
            "Coefficient:Q", scale=alt.Scale(scheme="oranges"), legend=None
        ),
        tooltip=["Feature", alt.Tooltip("Coefficient:Q", format=".3f")],
    )
    .properties(height=400, title="Logistic Regression |Coefficients|")
)

st.altair_chart(coef_chart, use_container_width=True)

# ========== STATISTICAL COMPARISON ==========
st.divider()

st.subheader("Statistical Comparison")
st.caption("Direct performance metrics comparing the three modeling approaches")

acc_global_lr = metrics_df.loc["Global Logistic", "Accuracy"]
acc_global_dt = metrics_df.loc["Global Decision Tree", "Accuracy"]
acc_stratified = metrics_df.loc["Origin-Stratified", "Accuracy"]

col_stat1, col_stat2, col_stat3 = st.columns(3)

with col_stat1:
    diff_strat_vs_lr = acc_stratified - acc_global_lr
    st.metric(
        "Stratified vs Global LR",
        f"{acc_stratified:.1%}",
        delta=f"{diff_strat_vs_lr:+.1%}",
        delta_color="normal" if diff_strat_vs_lr > 0 else "inverse",
    )

with col_stat2:
    diff_dt_vs_lr = acc_global_dt - acc_global_lr
    st.metric(
        "Decision Tree vs Global LR",
        f"{acc_global_dt:.1%}",
        delta=f"{diff_dt_vs_lr:+.1%}",
        delta_color="normal" if diff_dt_vs_lr > 0 else "inverse",
    )

with col_stat3:
    best_acc = max(acc_global_lr, acc_global_dt, acc_stratified)
    st.metric(
        "Best Model Accuracy",
        f"{best_acc:.1%}",
        delta=f"{best_acc - acc_global_lr:+.1%} vs baseline",
    )

st.divider()

# Conclusion Logic
best_model = metrics_df["Accuracy"].idxmax()
acc_diff = acc_stratified - acc_global_lr

st.success(f"### 🏆 Best Performing Approach: {best_model}")

# More nuanced interpretation based on actual results
if abs(acc_diff) < 0.03:  # Less than 3% difference
    st.markdown("""
    **Key Finding:** The stratified and global models perform **nearly identically**.
    
    This tells us something important about the "One Size Fits All" hypothesis:
    
    - **The institutional differences were primarily in data collection, not in disease patterns.**
    - Features like `ca`, `thal`, `slope`, `fbs`, and `chol` varied dramatically across hospitals because different institutions ran different tests.
    - Once we removed these unreliable features, the remaining 8 core features (`age`, `sex`, `cp`, `trestbps`, `restecg`, `thalach`, `exang`, `oldpeak`) are **consistent enough across institutions** that a single global model works.
    
    **Clinical Implication:** A simpler global model may be sufficient for deployment, but only if we restrict to features that are reliably collected across all institutions. The "one size fits all" debate is really about **data quality and standardization**, not model architecture.
    """)
elif best_model == "Origin-Stratified":
    st.markdown("""
    **Interpretation:** The Stratified approach performed best. This confirms that **institutional differences** 
    remain important even after feature selection. Local calibration provides measurable benefit.
    """)
elif best_model == "Global Decision Tree":
    st.markdown("""
    **Interpretation:** The Decision Tree model performed best. This suggests that **non-linear relationships** 
    in the data are more important than institutional differences. A flexible global model can capture 
    the patterns across all populations.
    """)
else:
    st.markdown("""
    **Interpretation:** The simple Global Logistic Regression performed well. 
    The core features are robust enough across institutions for a simple linear model to generalize.
    """)

# Add a summary of what was learned
st.divider()
st.subheader("Key Takeaways")

col_s1, col_s2 = st.columns(2)

with col_s1:
    st.markdown("""
    **❌ Features Dropped (Unreliable):**
    - `ca` - 99% missing in 3/4 datasets
    - `thal` - 83-90% missing in 3/4 datasets
    - `slope` - 51-65% missing in 3/4 datasets
    - `fbs` - 61% missing in Switzerland
    - `chol` - All zeros in Switzerland
    
    These features drove the "institutional differences" narrative but were really just data quality issues.
    """)

with col_s2:
    st.markdown(f"""
    **✅ Features Kept (Reliable):**
    - `age`, `sex`, `cp`, `trestbps`
    - `restecg`, `thalach`, `exang`, `oldpeak`
    
    With these 8 features, all three modeling approaches achieve ~{acc_global_lr:.0%} accuracy, 
    suggesting the core clinical indicators are consistent across institutions.
    """)

st.divider()
st.info(
    """
    **Why this matters:** The "One Size Fits All" debate in healthcare ML often assumes institutional 
    differences reflect genuine population variation. Our analysis shows that after controlling for 
    data quality, a single global model can generalize effectively—but only when using features 
    consistently collected across all sites.
    """,
    icon=":material/info:",
)
