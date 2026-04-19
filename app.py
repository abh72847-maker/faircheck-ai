import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from fairlearn.metrics import demographic_parity_difference
from fairlearn.reductions import (
    DemographicParity,
    ExponentiatedGradient,
)

import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Unbiased AI Decision System",
    layout="wide"
)

# =========================
# Helper Functions
# =========================

@st.cache_data
def load_data(file):
    return pd.read_csv(file)


def encode_data(df):
    encoders = {}
    df_encoded = df.copy()

    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(
                df_encoded[col].astype(str)
            )
            encoders[col] = le

    return df_encoded, encoders


def get_model(name):

    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)

    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

    return DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    )


def compute_bias(
        y_true,
        y_pred,
        sensitive_feature
):

    bias = demographic_parity_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )

    return abs(float(bias))


def explain_bias_groups(
        df,
        sensitive_col,
        target_col
):

    explanation = []

    groups = df[sensitive_col].unique()

    for g in groups:

        subset = df[df[sensitive_col] == g]

        rate = subset[target_col].mean()

        explanation.append(
            f"{sensitive_col} = {g} → Positive rate: {rate:.2f}"
        )

    return explanation


# =========================
# UI START
# =========================

st.title("⚖️ Unbiased AI Decision System")

st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset (CSV)",
    type=["csv"]
)

model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Random Forest",
        "Decision Tree"
    ]
)

show_heatmap = st.sidebar.checkbox(
    "Show Correlation Heatmap"
)

# =========================
# DATA LOADING
# =========================

if uploaded_file is None:

    st.info("Upload a dataset to begin.")
    st.stop()

df = load_data(uploaded_file)

if df.empty:

    st.error("Dataset is empty.")
    st.stop()

st.subheader("Dataset Preview")

st.dataframe(df.head())

# =========================
# SELECT TARGET
# =========================

target_column = st.sidebar.selectbox(
    "Select Target Column",
    df.columns
)

feature_columns = [
    c for c in df.columns
    if c != target_column
]

sensitive_column = st.sidebar.selectbox(
    "Sensitive Column (Fairness)",
    feature_columns
)

# =========================
# CLEANING
# =========================

df_clean = df.copy()

for col in df_clean.columns:

    if df_clean[col].dtype != "object":
        df_clean[col] = pd.to_numeric(
            df_clean[col],
            errors="coerce"
        )

# Impute numeric
numeric_cols = df_clean.select_dtypes(
    include=np.number
).columns

imputer = SimpleImputer(
    strategy="median"
)

df_clean[numeric_cols] = imputer.fit_transform(
    df_clean[numeric_cols]
)

# Encode
df_encoded, encoders = encode_data(
    df_clean
)

# =========================
# SPLIT DATA
# =========================

X = df_encoded[feature_columns]
y = df_encoded[target_column]

try:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

except:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

# =========================
# TRAIN MODEL
# =========================

model = get_model(model_name)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

st.success(
    f"Model Accuracy: {accuracy:.3f}"
)

# =========================
# BIAS DETECTION
# =========================

bias_before = compute_bias(
    y_test,
    y_pred,
    X_test[sensitive_column]
)

st.subheader("Bias Detection")

col1, col2 = st.columns(2)

col1.metric(
    "Bias Before Mitigation",
    f"{bias_before:.3f}"
)

fairness_score = max(
    0,
    1 - bias_before
)

col2.metric(
    "Fairness Score",
    f"{fairness_score:.3f}"
)

# =========================
# MITIGATION
# =========================

mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(
        max_iter=1000
    ),
    constraints=DemographicParity()
)

mitigator.fit(
    X_train,
    y_train,
    sensitive_features=X_train[
        sensitive_column
    ]
)

y_pred_mitigated = mitigator.predict(
    X_test
)

bias_after = compute_bias(
    y_test,
    y_pred_mitigated,
    X_test[sensitive_column]
)

st.subheader("Bias Mitigation")

col3, col4 = st.columns(2)

col3.metric(
    "Bias After Mitigation",
    f"{bias_after:.3f}"
)

improvement = bias_before - bias_after

col4.metric(
    "Bias Improvement",
    f"{improvement:.3f}"
)

# =========================
# GROUP EXPLANATION
# =========================

st.subheader("Group Bias Explanation")

group_info = explain_bias_groups(
    df_encoded,
    sensitive_column,
    target_column
)

for line in group_info:

    st.write(line)

# =========================
# BIAS VISUALIZATION
# =========================

st.subheader("Bias Visualization")

fig, ax = plt.subplots()

ax.bar(
    ["Before", "After"],
    [bias_before, bias_after]
)

ax.set_ylabel("Bias Score")

st.pyplot(fig)

# =========================
# CORRELATION HEATMAP
# =========================

if show_heatmap:

    st.subheader("Correlation Heatmap")

    corr = df_encoded.corr()

    fig2, ax2 = plt.subplots()

    cax = ax2.imshow(corr)

    plt.colorbar(cax)

    ax2.set_title("Feature Correlation")

    st.pyplot(fig2)

# =========================
# SMART AI INSIGHT
# =========================

st.subheader("AI Insight Summary")

if bias_after < bias_before:

    st.success(
        "Bias reduced successfully using fairness mitigation."
    )

else:

    st.warning(
        "Bias reduction limited. Consider data balancing."
    )

st.write(
"""
Suggested Improvements:

• Add more balanced training data  
• Review sensitive attributes  
• Compare multiple fairness models  
• Monitor fairness continuously  
"""
)
