import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from fairlearn.metrics import demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import DemographicParity

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# Page config
st.set_page_config(
    page_title="FairCheck AI Dashboard",
    page_icon="⚖️",
    layout="wide"
)


# Title
st.title("⚖️ FairCheck AI — Bias Detection Dashboard")

st.markdown(
"""
Detect, analyze, and reduce unfair bias in machine learning datasets.
"""
)


# Sidebar
st.sidebar.title("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

show_accuracy = st.sidebar.checkbox(
    "Show Model Accuracy"
)

show_heatmap = st.sidebar.checkbox(
    "Show Correlation Heatmap"
)


# PDF Report Function
def create_pdf_report(
    most_biased_column,
    before,
    after,
    severity
):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    story = []

    title = Paragraph(
        "FairCheck AI - Bias Detection Report",
        styles["Title"]
    )

    story.append(title)

    story.append(Spacer(1, 12))

    content = f"""
    <b>Most Biased Column:</b> {most_biased_column} <br/><br/>

    <b>Bias Before Mitigation:</b> {before:.3f} <br/><br/>

    <b>Bias After Mitigation:</b> {after:.3f} <br/><br/>

    <b>Bias Severity:</b> {severity} <br/><br/>

    <b>Fairness Score:</b> {1-before:.3f} <br/><br/>

    <b>Recommendations:</b><br/>
    1. Balance dataset samples.<br/>
    2. Review sensitive columns.<br/>
    3. Apply fairness-aware models.<br/>
    4. Monitor bias regularly.<br/>
    """

    paragraph = Paragraph(
        content,
        styles["BodyText"]
    )

    story.append(paragraph)

    doc.build(story)

    buffer.seek(0)

    return buffer


# Main logic
if uploaded_file:

    data = pd.read_csv(uploaded_file)

    st.subheader("📂 Dataset Preview")

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(data.head())

    with col2:
        st.write("Dataset Shape:", data.shape)
        st.write("Columns:", list(data.columns))

    # Data cleaning
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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Model training
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

    # Dashboard metrics
    st.subheader("📊 Bias Dashboard")

    col1, col2, col3 = st.columns(3)

    fairness_score = 1 - before

    col1.metric(
        "Most Biased Column",
        most_biased_column
    )

    col2.metric(
        "Bias Score",
        f"{before:.3f}"
    )

    col3.metric(
        "Fairness Score",
        f"{fairness_score:.3f}"
    )

    # Alerts
    if before > 0.3:
        st.error("🚨 High Bias Detected — Immediate Action Required!")

    elif before > 0.1:
        st.warning("⚠️ Medium Bias Detected — Monitor Carefully.")

    else:
        st.success("✅ Low Bias Detected — System Fair.")

    # Top biased columns table
    st.subheader("📋 Top Biased Columns")

    bias_df = pd.DataFrame(
        bias_results.items(),
        columns=["Column", "Bias Score"]
    )

    bias_df = bias_df.sort_values(
        by="Bias Score",
        ascending=False
    )

    st.dataframe(bias_df)

    # Bias chart
    st.subheader("📈 Bias Distribution")

    fig, ax = plt.subplots()

    ax.barh(
        bias_df["Column"],
        bias_df["Bias Score"]
    )

    ax.set_xlabel("Bias Score")

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

    # Recommendations
    st.subheader("🧠 Bias Reduction Recommendations")

    recommendation_text = f"""
Most bias detected in column:
➡️ {most_biased_column}

Recommended Actions:

1️⃣ Balance Dataset  
Add more samples for underrepresented groups.

2️⃣ Review Sensitive Feature  
Check whether '{most_biased_column}'
should influence predictions.

3️⃣ Use Fairness Constraints  
Apply fairness-aware algorithms.

4️⃣ Monitor Regularly  
Re-test fairness after updates.
"""

    st.info(recommendation_text)

    # PDF download
    pdf_file = create_pdf_report(
        most_biased_column,
        before,
        after,
        severity="Auto-calculated"
    )

    st.download_button(
        label="📄 Download Full PDF Report",
        data=pdf_file,
        file_name="FairCheck_Bias_Report.pdf",
        mime="application/pdf"
    )
