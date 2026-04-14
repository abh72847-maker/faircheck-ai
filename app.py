import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from fairlearn.metrics import demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import DemographicParity


# Page settings
st.set_page_config(
    page_title="FairCheck AI",
    page_icon="⚖️",
    layout="wide"
)

# Title
st.title("⚖️ FairCheck AI — Smart Bias Detection System")

st.markdown(
"""
Upload your dataset and detect unfair bias automatically.
"""
)

# Sidebar
st.sidebar.title("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

show_heatmap = st.sidebar.checkbox(
    "Show Correlation Heatmap"
)

show_accuracy = st.sidebar.checkbox(
    "Show Model Accuracy"
)


if uploaded_file:

    data = pd.read_csv(uploaded_file)

    # Dataset preview
    st.subheader("📂 Dataset Preview")

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(data.head())

    with col2:
        st.write("Dataset Shape:", data.shape)
        st.write("Columns:", list(data.columns))

    # Cleaning
    data = data.dropna()

    data = data.apply(
        lambda x: x.str.strip()
        if x.dtype == "object"
        else x
    )

    # Encoding
    le = LabelEncoder()

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = le.fit_transform(data[col])

    # Target column
    target_column = data.columns[-1]

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Model
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Accuracy
    if show_accuracy:

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        st.subheader("📈 Model Accuracy")

        st.success(f"Accuracy: {accuracy:.2f}")

    # Bias detection
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

    # Most biased column
    most_biased_column = max(
        bias_results,
        key=bias_results.get
    )

    st.subheader("⚠️ Most Biased Column")

    st.error(most_biased_column)

    # Bias mitigation
    mitigator = ExponentiatedGradient(
        LogisticRegression(max_iter=1000),
        constraints=DemographicParity()
    )

    mitigator.fit(
        X_train,
        y_train,
        sensitive_features=X_train[most_biased_column]
    )

    mitigated_predictions = mitigator.predict(
        X_test
    )

    # Bias comparison
    original_bias = demographic_parity_difference(
        y_true=y_test,
        y_pred=predictions,
        sensitive_features=X_test[most_biased_column]
    )

    new_bias = demographic_parity_difference(
        y_true=y_test,
        y_pred=mitigated_predictions,
        sensitive_features=X_test[most_biased_column]
    )

    before = abs(original_bias)
    after = abs(new_bias)

    st.subheader("📉 Bias Comparison")

    col1, col2 = st.columns(2)

    col1.metric(
        "Before Bias",
        f"{before:.3f}"
    )

    col2.metric(
        "After Bias",
        f"{after:.3f}"
    )

    # Severity
    if before > 0.3:
        severity = "🔴 High Bias"
    elif before > 0.1:
        severity = "🟡 Medium Bias"
    else:
        severity = "🟢 Low Bias"

    st.subheader("🚦 Bias Severity")

    st.write(severity)

    # Visualization
    st.subheader("📊 Bias Visualization")

    fig, ax = plt.subplots()

    ax.bar(
        bias_results.keys(),
        bias_results.values()
    )

    plt.xticks(rotation=90)

    st.pyplot(fig)

    # Heatmap
    if show_heatmap:

        st.subheader("🔥 Correlation Heatmap")

        fig2, ax2 = plt.subplots()

        sns.heatmap(
            data.corr(),
            cmap="coolwarm"
        )

        st.pyplot(fig2)

    # Smart Explanation Section
st.subheader("🧠 Bias Explanation & Recommendations")

if after < before:
    result = "✅ Bias successfully reduced after mitigation."
else:
    result = "⚠️ Bias reduction needs improvement."

# Generate explanation
explanation = f"""
📊 Bias Analysis Summary

Most Biased Column:
➡️ {most_biased_column}

Before Bias Score:
{before:.3f}

After Bias Score:
{after:.3f}

Result:
{result}

🔍 Why Bias Occurs:
Bias happens when certain groups in data 
(such as gender, age, or income group) 
receive unfair outcomes compared to others.

⚠️ Risk Identified:
Column '{most_biased_column}' shows the highest 
difference between groups, meaning it may 
influence unfair predictions.

🛠️ Recommended Actions to Reduce Bias:

1️⃣ Balance Dataset  
Add more samples for underrepresented groups.

2️⃣ Review Sensitive Features  
Check if '{most_biased_column}' should be 
used in prediction.

3️⃣ Apply Fairness Constraints  
Use fairness-aware models like 
Demographic Parity (already applied).

4️⃣ Monitor Model Regularly  
Re-check fairness after retraining.

📈 Impact:
Reducing bias improves fairness, trust, 
and real-world usability of AI systems.
"""

st.info(explanation)
    # Download report
    report_text = f"""
FairCheck AI Report

Most Biased Column: {most_biased_column}

Before Bias: {before}
After Bias: {after}

Severity: {severity}
"""

    st.download_button(
        "📥 Download Report",
        data=report_text,
        file_name="bias_report.txt"
    )    # Clean data
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
