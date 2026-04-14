import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from fairlearn.metrics import demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import DemographicParity

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="FairCheck AI",
    page_icon="⚖️",
    layout="wide"
)


# =========================
# TITLE
# =========================

st.title("⚖️ FairCheck AI — Smart Bias Detection Dashboard")

st.markdown(
"""
### Detect Bias • Improve Fairness • Generate Reports  
Upload datasets and analyze fairness using AI models.
"""
)


# =========================
# SIDEBAR
# =========================

st.sidebar.title("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression",
     "Random Forest",
     "Decision Tree"]
)

show_heatmap = st.sidebar.checkbox(
    "Show Correlation Heatmap"
)

show_accuracy = st.sidebar.checkbox(
    "Show Model Accuracy"
)


# =========================
# PDF FUNCTION
# =========================

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

    story.append(
        Paragraph(
            "FairCheck AI Bias Report",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 12))

    content = f"""
    <b>Most Biased Column:</b> {most_biased_column}<br/><br/>

    <b>Bias Before:</b> {before:.3f}<br/><br/>

    <b>Bias After:</b> {after:.3f}<br/><br/>

    <b>Severity:</b> {severity}<br/><br/>

    <b>Fairness Score:</b> {1-before:.3f}<br/><br/>

    <b>Recommendations:</b><br/>
    1. Balance dataset samples<br/>
    2. Review sensitive columns<br/>
    3. Apply fairness models<br/>
    4. Monitor fairness regularly<br/>
    """

    story.append(
        Paragraph(
            content,
            styles["BodyText"]
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer


# =========================
# MAIN LOGIC
# =========================

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    st.subheader("📂 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(data.head())

    with col2:
        st.write("Shape:", data.shape)
        st.write("Columns:", list(data.columns))


    # =========================
    # CLEANING
    # =========================

    data = data.dropna()

    data = data.apply(
        lambda x: x.str.strip()
        if x.dtype == "object"
        else x
    )

    le = LabelEncoder()

    for col in data.columns:

        if data[col].dtype == "object":
            data[col] = le.fit_transform(
                data[col]
            )


    # =========================
    # TARGET
    # =========================

    target_column = data.columns[-1]

    X = data.drop(
        target_column,
        axis=1
    )

    y = data[target_column]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    # =========================
    # MODEL SELECTION ⭐
    # =========================

    if model_choice == "Logistic Regression":

        model = LogisticRegression(
            max_iter=1000
        )

    elif model_choice == "Random Forest":

        model = RandomForestClassifier()

    else:

        model = DecisionTreeClassifier()


    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )


    # =========================
    # ACCURACY
    # =========================

    if show_accuracy:

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        st.subheader("📈 Model Accuracy")

        st.success(
            f"Accuracy: {accuracy:.2f}"
        )


    # =========================
    # BIAS DETECTION
    # =========================

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


    most_biased_column = max(
        bias_results,
        key=bias_results.get
    )


    # =========================
    # BIAS MITIGATION
    # =========================

    mitigator = ExponentiatedGradient(
        LogisticRegression(max_iter=1000),
        constraints=DemographicParity()
    )

    mitigator.fit(
        X_train,
        y_train,
        sensitive_features=X_train[
            most_biased_column
        ]
    )

    mitigated_predictions = mitigator.predict(
        X_test
    )


    original_bias = demographic_parity_difference(
        y_true=y_test,
        y_pred=predictions,
        sensitive_features=X_test[
            most_biased_column
        ]
    )

    new_bias = demographic_parity_difference(
        y_true=y_test,
        y_pred=mitigated_predictions,
        sensitive_features=X_test[
            most_biased_column
        ]
    )


    before = abs(original_bias)

    after = abs(new_bias)

    fairness_score = 1 - before


    # =========================
    # SEVERITY
    # =========================

    if before > 0.3:

        severity = "🔴 High Bias"

    elif before > 0.1:

        severity = "🟡 Medium Bias"

    else:

        severity = "🟢 Low Bias"


    # =========================
    # DASHBOARD CARDS ⭐
    # =========================

    st.subheader("📊 Bias Dashboard")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Most Biased Column",
        most_biased_column
    )

    c2.metric(
        "Bias Score",
        f"{before:.3f}"
    )

    c3.metric(
        "Fairness Score",
        f"{fairness_score:.3f}"
    )


    # =========================
    # ALERTS
    # =========================

    if before > 0.3:

        st.error(
            "🚨 High Bias Detected!"
        )

    elif before > 0.1:

        st.warning(
            "⚠️ Medium Bias Detected!"
        )

    else:

        st.success(
            "✅ Low Bias Detected!"
        )


    # =========================
    # TOP BIAS TABLE
    # =========================

    st.subheader("📋 Top Biased Columns")

    bias_df = pd.DataFrame(
        bias_results.items(),
        columns=[
            "Column",
            "Bias Score"
        ]
    )

    bias_df = bias_df.sort_values(
        by="Bias Score",
        ascending=False
    )

    st.dataframe(bias_df)


    # =========================
    # BIAS CHART
    # =========================

    st.subheader("📈 Bias Distribution")

    fig, ax = plt.subplots()

    ax.barh(
        bias_df["Column"],
        bias_df["Bias Score"]
    )

    st.pyplot(fig)


    # =========================
    # HEATMAP
    # =========================

    if show_heatmap:

        st.subheader(
            "🔥 Correlation Heatmap"
        )

        fig2, ax2 = plt.subplots()

        sns.heatmap(
            data.corr(),
            cmap="coolwarm"
        )

        st.pyplot(fig2)


    # =========================
    # EXPLANATION ⭐
    # =========================

    st.subheader(
        "🧠 Bias Explanation & Fixes"
    )

    explanation = f"""
Most bias detected in:

➡️ {most_biased_column}

Bias Before:
{before:.3f}

Bias After:
{after:.3f}

Recommended Fixes:

1️⃣ Balance Dataset  
2️⃣ Remove biased feature  
3️⃣ Use fairness algorithms  
4️⃣ Monitor fairness regularly  
"""

    st.info(explanation)


    # =========================
    # PDF DOWNLOAD ⭐
    # =========================

    pdf_file = create_pdf_report(
        most_biased_column,
        before,
        after,
        severity
    )

    st.download_button(
        label="📄 Download Full PDF Report",
        data=pdf_file,
        file_name="FairCheck_Bias_Report.pdf",
        mime="application/pdf"
    )
