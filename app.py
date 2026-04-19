import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from fairlearn.metrics import demographic_parity_difference
from fairlearn.reductions import DemographicParity, ExponentiatedGradient

st.set_page_config(
    page_title="Unbiased AI Decision System",
    page_layout="wide",
    page_icon="⚖️"
)

# =========================
# HELPER FUNCTIONS
# =========================

@st.cache_data
def load_data(file):
    """Caches the dataset so Streamlit doesn't reload it constantly."""
    return pd.read_csv(file)

def get_model(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    else:
        return DecisionTreeClassifier(max_depth=6, random_state=42)

def compute_bias(y_true, y_pred, sensitive_feature):
    bias = demographic_parity_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )
    return abs(float(bias))

# =========================
# UI & SIDEBAR CONFIG
# =========================

st.title("⚖️ Unbiased AI Decision System")
st.markdown("Detect and mitigate hidden biases in machine learning models to ensure fair decisions in hiring, loans, and healthcare.")

st.sidebar.header("1. Upload & Configure")

uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("👋 Welcome! Please upload a CSV dataset in the sidebar to begin.")
    st.stop()

# Load Data
df = load_data(uploaded_file)

with st.expander("🔍 View Raw Data Preview"):
    st.dataframe(df.head())

# =========================
# PIPELINE CONFIGURATION FORM
# =========================
# Using a form prevents Streamlit from rerunning on every single dropdown change!
with st.sidebar.form("config_form"):
    st.header("2. Model Settings")
    
    target_column = st.selectbox("Select Target Column (What to predict)", df.columns)
    
    feature_columns = [c for c in df.columns if c != target_column]
    sensitive_column = st.selectbox("Sensitive Column (e.g., Race, Gender, Age)", feature_columns)
    
    model_name = st.selectbox("Select Base ML Model", ["Logistic Regression", "Random Forest", "Decision Tree"])
    
    submit_button = st.form_submit_button(label="🚀 Run Fairness Pipeline")

# =========================
# MAIN EXECUTION PIPELINE
# =========================

if submit_button:
    with st.spinner('Training models and analyzing bias...'):
        
        # 1. PREPROCESSING
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode target if categorical
        if y.dtype == "object":
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)

        # Handle Categorical features via One-Hot Encoding
        X = pd.get_dummies(X, drop_first=True)

        # Ensure the sensitive column is preserved as a distinct array for Fairlearn
        sensitive_features_full = df[sensitive_column]

        # 2. SPLIT DATA (Done before imputation to prevent data leakage)
        X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
            X, y, sensitive_features_full, test_size=0.2, random_state=42
        )

        # 3. IMPUTE MISSING VALUES
        imputer = SimpleImputer(strategy="median")
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

        # 4. TRAIN STANDARD BIASED MODEL
        base_model = get_model(model_name)
        base_model.fit(X_train, y_train)
        y_pred_standard = base_model.predict(X_test)

        acc_standard = accuracy_score(y_test, y_pred_standard)
        bias_standard = compute_bias(y_test, y_pred_standard, sens_test)

        # 5. TRAIN MITIGATED FAIR MODEL
        # We pass the dynamically selected base model into the mitigator!
        mitigator = ExponentiatedGradient(
            estimator=get_model(model_name), 
            constraints=DemographicParity(),
            max_iter=50 # Kept low for fast hackathon generation
        )
        mitigator.fit(X_train, y_train, sensitive_features=sens_train)
        y_pred_fair = mitigator.predict(X_test)

        acc_fair = accuracy_score(y_test, y_pred_fair)
        bias_fair = compute_bias(y_test, y_pred_fair, sens_test)

        # =========================
        # RESULTS DASHBOARD
        # =========================
        st.divider()
        st.subheader("📊 Model Performance & Fairness Results")
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Original Accuracy", f"{acc_standard:.2%}")
        col2.metric("Fair Model Accuracy", f"{acc_fair:.2%}", delta=f"{(acc_fair - acc_standard):.2%}", delta_color="normal")
        
        col3.metric("Original Bias Score", f"{bias_standard:.3f}")
        col4.metric("Fair Model Bias Score", f"{bias_fair:.3f}", delta=f"{(bias_fair - bias_standard):.3f}", delta_color="inverse")

        # =========================
        # VISUALIZATIONS
        # =========================
        st.divider()
        st.subheader("📈 Trade-off Analysis")
        
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            # Bar Chart comparing Bias
            fig_bias = px.bar(
                x=["Before Mitigation", "After Mitigation"], 
                y=[bias_standard, bias_fair],
                labels={'x': 'Model Phase', 'y': 'Demographic Parity Difference (Lower is better)'},
                title="Bias Reduction",
                color=["Before Mitigation", "After Mitigation"],
                color_discrete_sequence=["#EF553B", "#00CC96"]
            )
            st.plotly_chart(fig_bias, use_container_width=True)
            
        with fig_col2:
            # Bar Chart comparing Accuracy
            fig_acc = px.bar(
                x=["Standard Model", "Fair Model"], 
                y=[acc_standard, acc_fair],
                labels={'x': 'Model Phase', 'y': 'Accuracy'},
                title="Accuracy Trade-off",
                color=["Standard Model", "Fair Model"],
                color_discrete_sequence=["#636EFA", "#AB63FA"]
            )
            # Zoom the Y-axis in so the change is actually visible
            fig_acc.update_yaxes(range=[min(acc_standard, acc_fair) - 0.05, 1.0])
            st.plotly_chart(fig_acc, use_container_width=True)

        # AI INSIGHT
        st.success("✅ **Analysis Complete:** The Exponentiated Gradient algorithm successfully adjusted the decision boundaries to minimize demographic disparity while attempting to preserve overall predictive accuracy.")
