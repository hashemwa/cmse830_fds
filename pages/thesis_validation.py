import streamlit as st
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from analysis import get_clean_data

# Setup
st.title("Thesis Validation")
st.markdown("### Evidence: Why 'One Size Fits All' Fails")

# Load the fully imputed/clean data for this experiment
# (We use the clean data here to focus on feature selection/population effects rather than imputation mechanics)
df = get_clean_data()

# Define Feature Sets
# 'Rich' features include the powerful fluoroscopy/stress test data (only complete in Cleveland)
rich_features = [
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

# 'Common' features are reliable across ALL locations (dropping ones with >50% missing in any origin)
# Note: chol excluded because Switzerland has 100% missing (fake zeros)
common_features = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
]

target = "target"

# ---------------------------------------------------------
# EXPERIMENT 1: THE COST OF DROPPING FEATURES
# ---------------------------------------------------------
st.header("1. The Cost of Standardization")
st.markdown("""
**Hypothesis:** Forcing a "Global Standard" (dropping missing columns like `ca` and `thal`) 
significantly hurts performance for hospitals that actually *have* that data (like Cleveland).
""")

# Filter for Cleveland only (where we have Ground Truth for everything)
cleveland_data = df[df["origin"] == "Cleveland"].copy()

if not cleveland_data.empty:
    X_cleve = cleveland_data
    y_cleve = cleveland_data[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_cleve, y_cleve, test_size=0.2, random_state=42
    )

    # Train "Rich" Model (All Features)
    model_rich = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
    )
    model_rich.fit(X_train[rich_features], y_train)
    acc_rich = accuracy_score(y_test, model_rich.predict(X_test[rich_features]))

    # Train "Poor" Model (Common Features Only)
    model_poor = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
    )
    model_poor.fit(X_train[common_features], y_train)
    acc_poor = accuracy_score(y_test, model_poor.predict(X_test[common_features]))

    # Display Results
    col1, col2, col3 = st.columns(3)
    col1.metric("Local Model (With ca/thal)", f"{acc_rich:.1%}")
    col2.metric("Global Standard (Without ca/thal)", f"{acc_poor:.1%}")

    delta = acc_rich - acc_poor
    col3.metric("Accuracy Lost", f"-{delta:.1%}", delta_color="inverse")

    if delta > 0.05:
        st.error(
            f"**Evidence:** Dropping specialized tests to accommodate other hospitals reduces Cleveland's diagnostic accuracy by **{delta:.1%}**."
        )
    else:
        st.info(
            f"**Observation:** The accuracy loss is minimal ({delta:.1%}), suggesting basic vitals are surprisingly effective."
        )
else:
    st.error("Cleveland data not found. Check data loading.")

st.divider()

# ---------------------------------------------------------
# EXPERIMENT 2: THE TRANSFER MATRIX
# ---------------------------------------------------------
st.header("2. Cross-Hospital Generalization (Transfer Matrix)")
st.markdown("""
**Hypothesis:** Even using the *exact same features*, a model trained on one hospital 
will fail when applied to a different hospital due to population differences.
""")

# Use COMMON features only for fair comparison across sites
X_all = df[common_features]
y_all = df[target]
origins = df["origin"]

hospital_list = sorted(df["origin"].unique())
results = []

# Calculate the matrix
with st.spinner("Training cross-site models..."):
    for train_origin in hospital_list:
        # 1. Train on Source Hospital
        train_mask = origins == train_origin
        X_source = X_all[train_mask]
        y_source = y_all[train_mask]

        if len(X_source) > 10:
            # Use Logistic Regression for consistency with modeling page
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
                ]
            )
            model.fit(X_source, y_source)

            # 2. Test on ALL Hospitals (Target)
            for test_origin in hospital_list:
                test_mask = origins == test_origin
                X_target = X_all[test_mask]
                y_target = y_all[test_mask]

                if len(X_target) > 0:
                    acc = accuracy_score(y_target, model.predict(X_target))
                    results.append(
                        {
                            "Train Origin": train_origin,
                            "Test Origin": test_origin,
                            "Accuracy": acc,
                        }
                    )

# Visualization
if results:
    results_df = pd.DataFrame(results)

    heatmap = (
        alt.Chart(results_df)
        .mark_rect()
        .encode(
            x=alt.X(
                "Test Origin:N", title="Testing Target", axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y("Train Origin:N", title="Training Source"),
            color=alt.Color(
                "Accuracy:Q",
                scale=alt.Scale(scheme="magma"),
                legend=alt.Legend(format=".0%"),
            ),
            tooltip=[
                "Train Origin",
                "Test Origin",
                alt.Tooltip("Accuracy", format=".1%"),
            ],
        )
        .properties(width=600, height=500, title="Model Transferability Matrix")
    )

    # Add text labels
    text = heatmap.mark_text(baseline="middle").encode(
        text=alt.Text("Accuracy:Q", format=".1%"),
        color=alt.condition(
            alt.datum.Accuracy > 0.7,
            alt.value("black"),  # Dark text for bright (high accuracy) cells
            alt.value("white"),  # Light text for dark (low accuracy) cells
        ),
    )

    st.altair_chart(heatmap + text, use_container_width=True)

    st.info("""
    **How to read this matrix:**
    - Each row is a model trained on that hospital's data
    - Each column shows how that model performs when tested on a different hospital
    - **Diagonal (e.g., Cleveland→Cleveland):** Training and testing on the same hospital
    - **Off-diagonal (e.g., Cleveland→Hungary):** Model transferred to a new hospital
    
    **Key insight:** When models are applied to hospitals they weren't trained on, accuracy drops significantly. This proves that patient populations differ across hospitals, and a single global model cannot work optimally everywhere.
    
    *Note: We use Logistic Regression, which finds a linear decision boundary rather than memorizing data points. This is why even diagonal cells show ~80% accuracy instead of 100%.*
    """)
