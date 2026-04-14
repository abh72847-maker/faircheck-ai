import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from fairlearn.metrics import demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import DemographicParity


# Title
st.title("⚖️ FairCheck AI — Bias Detection System")

st.write(
    "Upload a dataset to detect and reduce bias automatically."
)


# Upload dataset
uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)


if uploaded_file:

    # Load data
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Clean data
    data = data.dropna()

    # Remove spaces
    data = data.apply(
        lambda x: x.str.strip()
        if x.dtype == "object"
        else x
    )

    # Encode text columns
    le = LabelEncoder()

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = le.fit_transform(data[col])

    # Target column = last column
    target_column = data.columns[-1]

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Detect bias for all columns
    bias_results = {}

    for col in X_test.columns:

        try:

            bias = demographic_parity_difference(
                y_true=y_test,
                y_pred=predictions,
                sensitive_features=X_test[col]
            )

            bias_results[col] = abs(bias)

        except:
            continue

    # Find most biased column
    most_biased_column = max(
        bias_results,
        key=bias_results.get
    )

    st.subheader("Most Biased Column")

    st.write(most_biased_column)

    # Bias mitigation
    mitigator = ExponentiatedGradient(
        LogisticRegression(max_iter=1000),
        constraints=DemographicParity()
    )

    sensitive_column = most_biased_column

    mitigator.fit(
        X_train,
        y_train,
        sensitive_features=X_train[sensitive_column]
    )

    mitigated_predictions = mitigator.predict(
        X_test
    )

    # Compare bias
    original_bias = demographic_parity_difference(
        y_true=y_test,
        y_pred=predictions,
        sensitive_features=X_test[sensitive_column]
    )

    new_bias = demographic_parity_difference(
        y_true=y_test,
        y_pred=mitigated_predictions,
        sensitive_features=X_test[sensitive_column]
    )

    before = abs(original_bias)
    after = abs(new_bias)

    st.subheader("Bias Comparison")

    st.write("Before Bias:", before)
    st.write("After Bias:", after)

    # Bias severity
    if before > 0.3:
        severity = "🔴 High Bias"
    elif before > 0.1:
        severity = "🟡 Medium Bias"
    else:
        severity = "🟢 Low Bias"

    st.subheader("Bias Severity")

    st.write(severity)

    # Visualization
    st.subheader("Bias Visualization")

    fig, ax = plt.subplots()

    ax.bar(
        bias_results.keys(),
        bias_results.values()
    )

    plt.xticks(rotation=90)

    st.pyplot(fig)

    # Explanation
    st.subheader("AI Explanation")

    if after < before:
        result = "Bias was successfully reduced."
    else:
        result = "Bias reduction needs improvement."

    explanation = f"""
📊 Bias Explanation

Before Bias: {before}
After Bias: {after}

Result:
{result}

Why Bias Matters:
AI systems with bias may unfairly
treat certain groups.

Suggested Improvements:
• Balance dataset
• Monitor sensitive features
• Retrain models regularly
"""

    st.write(explanation)

    # Download report
    report_text = f"""
Bias Report

Most Biased Column: {most_biased_column}

Before Bias: {before}
After Bias: {after}

Bias Severity: {severity}

Explanation:
{result}
"""

    st.download_button(
        label="📥 Download Bias Report",
        data=report_text,
        file_name="bias_report.txt",
        mime="text/plain"
    )
