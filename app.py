import io
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from fairlearn.metrics import demographic_parity_difference
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(
    page_title="FairCheck AI",
    page_icon="⚖️",
    layout="wide",
)


def apply_theme(dark_mode: bool) -> None:
    """Apply a polished custom theme for light and dark mode."""
    if dark_mode:
        bg = "#07111f"
        bg_soft = "#0f1c2e"
        surface = "rgba(15, 23, 42, 0.88)"
        surface_strong = "#111c31"
        sidebar = "rgba(8, 15, 29, 0.95)"
        text = "#f8fafc"
        muted = "#cbd5e1"
        border = "rgba(148, 163, 184, 0.24)"
        accent = "#19c37d"
        accent_2 = "#38bdf8"
        accent_soft = "rgba(25, 195, 125, 0.14)"
        shadow = "0 20px 50px rgba(2, 6, 23, 0.45)"
    else:
        bg = "#f5f7fb"
        bg_soft = "#e8eef8"
        surface = "rgba(255, 255, 255, 0.88)"
        surface_strong = "#ffffff"
        sidebar = "rgba(241, 245, 249, 0.92)"
        text = "#0f172a"
        muted = "#475569"
        border = "rgba(15, 23, 42, 0.12)"
        accent = "#0f62fe"
        accent_2 = "#0891b2"
        accent_soft = "rgba(15, 98, 254, 0.10)"
        shadow = "0 20px 50px rgba(15, 23, 42, 0.08)"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=DM+Sans:wght@400;500;700&display=swap');

        :root {{
            --fc-bg: {bg};
            --fc-bg-soft: {bg_soft};
            --fc-surface: {surface};
            --fc-surface-strong: {surface_strong};
            --fc-sidebar: {sidebar};
            --fc-text: {text};
            --fc-muted: {muted};
            --fc-border: {border};
            --fc-accent: {accent};
            --fc-accent-2: {accent_2};
            --fc-accent-soft: {accent_soft};
            --fc-shadow: {shadow};
        }}

        .stApp {{
            background:
                radial-gradient(circle at top left, var(--fc-accent-soft), transparent 28%),
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.12), transparent 24%),
                linear-gradient(135deg, var(--fc-bg) 0%, var(--fc-bg-soft) 100%);
            color: var(--fc-text);
            font-family: "DM Sans", sans-serif;
        }}

        section[data-testid="stSidebar"] {{
            background: var(--fc-sidebar);
            border-right: 1px solid var(--fc-border);
            backdrop-filter: blur(18px);
        }}

        h1, h2, h3, h4 {{
            color: var(--fc-text);
            font-family: "Space Grotesk", sans-serif;
            letter-spacing: -0.03em;
        }}

        h5, h6, p, div, span, label {{
            color: var(--fc-text);
        }}

        .stMarkdown, .stText, .stCaption, small {{
            color: var(--fc-muted);
        }}

        .stDataFrame, div[data-testid="stTable"] {{
            background-color: var(--fc-surface-strong);
            border: 1px solid var(--fc-border);
            border-radius: 18px;
            padding: 0.4rem;
            box-shadow: var(--fc-shadow);
        }}

        div[data-testid="metric-container"] {{
            background: linear-gradient(180deg, var(--fc-surface) 0%, var(--fc-surface-strong) 100%);
            border: 1px solid var(--fc-border);
            border-radius: 20px;
            padding: 1rem;
            box-shadow: var(--fc-shadow);
            backdrop-filter: blur(14px);
        }}

        div[data-testid="stAlert"] {{
            background: var(--fc-accent-soft);
            color: var(--fc-text);
            border: 1px solid var(--fc-border);
            border-radius: 18px;
        }}

        .stButton > button, .stDownloadButton > button {{
            background: linear-gradient(135deg, var(--fc-accent) 0%, var(--fc-accent-2) 100%);
            color: white;
            border: none;
            border-radius: 999px;
            padding: 0.7rem 1.2rem;
            font-weight: 600;
            box-shadow: 0 14px 30px rgba(8, 145, 178, 0.22);
        }}

        .stSelectbox div[data-baseweb="select"],
        .stMultiSelect div[data-baseweb="select"],
        .stTextInput input,
        .stNumberInput input {{
            background: var(--fc-surface-strong);
            color: var(--fc-text);
            border: 1px solid var(--fc-border);
            border-radius: 14px;
        }}

        .stCheckbox label, .stRadio label, .stFileUploader label {{
            color: var(--fc-text);
        }}

        div[data-testid="stFileUploader"] {{
            background: var(--fc-surface);
            border: 1px dashed var(--fc-border);
            border-radius: 18px;
            padding: 0.9rem;
            backdrop-filter: blur(8px);
        }}

        div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] > div:has(> .premium-card) {{
            margin-bottom: 0.5rem;
        }}

        .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }}

        .premium-hero {{
            background:
                radial-gradient(circle at top left, var(--fc-accent-soft), transparent 35%),
                linear-gradient(135deg, var(--fc-surface) 0%, var(--fc-surface-strong) 100%);
            border: 1px solid var(--fc-border);
            border-radius: 28px;
            padding: 2rem 2rem 1.6rem 2rem;
            box-shadow: var(--fc-shadow);
            backdrop-filter: blur(18px);
            margin-bottom: 1.25rem;
        }}

        .premium-kicker {{
            color: var(--fc-accent);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }}

        .premium-title {{
            color: var(--fc-text);
            font-family: "Space Grotesk", sans-serif;
            font-size: clamp(2rem, 4vw, 3.8rem);
            line-height: 0.95;
            font-weight: 700;
            margin: 0;
        }}

        .premium-subtitle {{
            color: var(--fc-muted);
            font-size: 1.02rem;
            line-height: 1.7;
            max-width: 780px;
            margin-top: 0.9rem;
        }}

        .premium-card {{
            background: linear-gradient(180deg, var(--fc-surface) 0%, var(--fc-surface-strong) 100%);
            border: 1px solid var(--fc-border);
            border-radius: 22px;
            padding: 1.15rem 1.2rem;
            box-shadow: var(--fc-shadow);
            backdrop-filter: blur(12px);
        }}

        .premium-card h3 {{
            margin: 0 0 0.45rem 0;
            font-size: 1.05rem;
        }}

        .premium-card p {{
            margin: 0;
            color: var(--fc-muted);
            line-height: 1.6;
        }}

        .section-label {{
            color: var(--fc-muted);
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.78rem;
            font-weight: 700;
            margin: 1.4rem 0 0.65rem 0;
        }}

        @media (max-width: 900px) {{
            .premium-hero {{
                padding: 1.35rem;
                border-radius: 22px;
            }}

            .premium-title {{
                font-size: 2.15rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def generate_ai_insight(most_biased_column: str, before: float, after: float) -> str:
    status = "Bias reduced successfully." if after < before else "Bias reduction was limited."
    return f"""
AI Insight Summary

Most biased column: {most_biased_column}
Bias before mitigation: {before:.3f}
Bias after mitigation: {after:.3f}
Status: {status}

Why this matters:
Bias can create unfair outcomes for different groups in your data.

Suggested next steps:
1. Improve class balance and group balance.
2. Review sensitive or proxy features.
3. Compare multiple fairness-aware models.
4. Monitor accuracy and fairness together after retraining.
"""


def build_pdf_report(
    dataset_shape: Tuple[int, int],
    target_column: str,
    model_name: str,
    accuracy: Optional[float],
    most_biased_column: str,
    before: float,
    after: float,
    insight_text: str,
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("FairCheck AI Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Dataset shape: {dataset_shape[0]} rows x {dataset_shape[1]} columns", styles["BodyText"]),
        Spacer(1, 8),
        Paragraph(f"Target column: {target_column}", styles["BodyText"]),
        Spacer(1, 8),
        Paragraph(f"Selected model: {model_name}", styles["BodyText"]),
        Spacer(1, 8),
        Paragraph(
            f"Accuracy: {accuracy:.3f}" if accuracy is not None else "Accuracy: Not calculated",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph(f"Most biased column: {most_biased_column}", styles["BodyText"]),
        Spacer(1, 8),
        Paragraph(f"Bias before mitigation: {before:.3f}", styles["BodyText"]),
        Spacer(1, 8),
        Paragraph(f"Bias after mitigation: {after:.3f}", styles["BodyText"]),
        Spacer(1, 12),
    ]

    for line in insight_text.strip().splitlines():
        if line.strip():
            story.append(Paragraph(line.strip(), styles["BodyText"]))
            story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def sanitize_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    for col in cleaned.select_dtypes(include="object").columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        cleaned[col] = cleaned[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return cleaned


def encode_dataframe(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    encoded = data.copy()
    encoders: Dict[str, LabelEncoder] = {}
    for col in encoded.columns:
        if encoded[col].dtype == "object" or str(encoded[col].dtype).startswith("category"):
            encoder = LabelEncoder()
            encoded[col] = encoder.fit_transform(encoded[col].astype(str))
            encoders[col] = encoder
    return encoded, encoders


def get_model(model_name: str):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    return DecisionTreeClassifier(max_depth=6, random_state=42)


def calculate_bias_scores(
    y_true: pd.Series, y_pred: pd.Series, features: pd.DataFrame
) -> Dict[str, float]:
    bias_results: Dict[str, float] = {}
    for col in features.columns:
        sensitive_feature = features[col]
        if sensitive_feature.nunique(dropna=True) < 2:
            continue
        try:
            bias = demographic_parity_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_feature,
            )
            bias_results[col] = abs(float(bias))
        except Exception:
            continue
    return bias_results


def render_bar_chart(bias_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    palette = sns.color_palette("blend:#0f62fe,#19c37d", n_colors=len(bias_df))
    ax.barh(bias_df["Column"], bias_df["Bias Score"], color=palette)
    ax.set_xlabel("Bias Score")
    ax.set_ylabel("Column")
    ax.set_title("Bias Distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.25)
    ax.spines["bottom"].set_alpha(0.25)
    ax.grid(axis="x", linestyle="--", alpha=0.15)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_heatmap(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    sns.heatmap(
        data.corr(numeric_only=True),
        cmap=sns.diverging_palette(220, 28, as_cmap=True),
        annot=False,
        linewidths=0.5,
        linecolor=(1, 1, 1, 0.06),
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def validate_target(y: pd.Series) -> Tuple[bool, str]:
    unique_values = y.nunique(dropna=True)
    if unique_values < 2:
        return False, "The target column must contain at least 2 classes."
    if unique_values > 2:
        return False, "This app currently supports binary classification targets only."
    return True, ""


def choose_sensitive_column(default_columns: List[str]) -> Optional[str]:
    if not default_columns:
        return None
    return st.sidebar.selectbox("Sensitive column for fairness mitigation", default_columns)


def render_hero() -> None:
    st.markdown(
        """
        <div class="premium-hero">
            <div class="premium-kicker">Responsible AI Analytics</div>
            <h1 class="premium-title">FairCheck AI</h1>
            <p class="premium-subtitle">
                Upload a dataset, train a model, detect demographic bias, compare mitigation impact,
                and export a report from a cleaner, more polished fairness dashboard.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_intro_cards() -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="premium-card">
                <h3>Bias Detection</h3>
                <p>Identify which feature groups show the strongest demographic parity gaps.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="premium-card">
                <h3>Fairness Mitigation</h3>
                <p>Apply a fairness-aware reduction strategy and compare the shift in bias scores.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="premium-card">
                <h3>Exportable Reporting</h3>
                <p>Generate a concise PDF summary for presentations, demos, and project reviews.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


apply_theme(False)

dark_mode = st.sidebar.toggle("Dark Mode", value=False)
apply_theme(dark_mode)
render_hero()
render_intro_cards()

st.sidebar.markdown("### Workspace Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
model_choice = st.sidebar.selectbox(
    "Select model",
    ["Logistic Regression", "Random Forest", "Decision Tree"],
)
show_heatmap = st.sidebar.checkbox("Show correlation heatmap", value=False)
show_accuracy = st.sidebar.checkbox("Show model accuracy", value=True)


if not uploaded_file:
    st.info("Upload a CSV file from the sidebar to begin the analysis.")
    st.stop()


try:
    raw_data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if raw_data.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

data = sanitize_dataframe(raw_data)

section_label("Dataset")
st.subheader("Dataset Overview")
left, right = st.columns(2)
with left:
    st.dataframe(data.head(), use_container_width=True)
with right:
    st.markdown(
        f"""
        <div class="premium-card">
            <h3>Dataset Snapshot</h3>
            <p><strong>Rows:</strong> {data.shape[0]}</p>
            <p><strong>Columns:</strong> {data.shape[1]}</p>
            <p><strong>Fields:</strong> {", ".join(list(data.columns))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if data.shape[1] < 2:
    st.error("The dataset must have at least one feature column and one target column.")
    st.stop()

target_column = st.sidebar.selectbox("Target column", list(data.columns), index=len(data.columns) - 1)
feature_columns = [col for col in data.columns if col != target_column]
sensitive_column = choose_sensitive_column(feature_columns)

if sensitive_column is None:
    st.error("Please upload a dataset with at least one feature column.")
    st.stop()

encoded_data = data.copy()
for col in encoded_data.columns:
    if encoded_data[col].dtype != "object":
        encoded_data[col] = pd.to_numeric(encoded_data[col], errors="coerce")

object_cols = encoded_data.select_dtypes(include="object").columns.tolist()
encoded_data[object_cols] = encoded_data[object_cols].fillna("Missing")

numeric_cols = [col for col in encoded_data.columns if col not in object_cols]
if numeric_cols:
    numeric_imputer = SimpleImputer(strategy="median")
    encoded_data[numeric_cols] = numeric_imputer.fit_transform(encoded_data[numeric_cols])

encoded_data, _ = encode_dataframe(encoded_data)

X = encoded_data[feature_columns]
y = encoded_data[target_column]

is_valid_target, validation_message = validate_target(y)
if not is_valid_target:
    st.error(validation_message)
    st.stop()

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

model = get_model(model_choice)

try:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

accuracy: Optional[float] = None
if show_accuracy:
    accuracy = accuracy_score(y_test, predictions)
    st.success(f"Model accuracy: {accuracy:.3f}")

bias_results = calculate_bias_scores(y_test, predictions, X_test)
if not bias_results:
    st.error("Could not compute bias metrics. Check whether the dataset has valid feature groups with at least 2 values.")
    st.stop()

most_biased_column = max(bias_results, key=bias_results.get)

mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(max_iter=1000),
    constraints=DemographicParity(),
)

try:
    mitigator.fit(
        X_train,
        y_train,
        sensitive_features=X_train[sensitive_column],
    )
    mitigated_predictions = mitigator.predict(X_test)
except Exception:
    # If the user-selected sensitive column fails, fall back to the most biased one from the baseline model.
    mitigator.fit(
        X_train,
        y_train,
        sensitive_features=X_train[most_biased_column],
    )
    mitigated_predictions = mitigator.predict(X_test)
    sensitive_column = most_biased_column

original_bias = demographic_parity_difference(
    y_true=y_test,
    y_pred=predictions,
    sensitive_features=X_test[sensitive_column],
)
new_bias = demographic_parity_difference(
    y_true=y_test,
    y_pred=mitigated_predictions,
    sensitive_features=X_test[sensitive_column],
)

before = abs(float(original_bias))
after = abs(float(new_bias))
fairness_score = max(0.0, 1 - before)
mitigation_delta = before - after

section_label("Analysis")
st.subheader("Bias Dashboard")
metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Most biased column", most_biased_column)
metric_2.metric("Bias score", f"{before:.3f}")
metric_3.metric("Fairness score", f"{fairness_score:.3f}")

detail_1, detail_2, detail_3 = st.columns(3)
detail_1.markdown(
    f"""
    <div class="premium-card">
        <h3>Mitigation Focus</h3>
        <p>{sensitive_column}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
detail_2.markdown(
    f"""
    <div class="premium-card">
        <h3>Bias After Mitigation</h3>
        <p>{after:.3f}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
detail_3.markdown(
    f"""
    <div class="premium-card">
        <h3>Bias Improvement</h3>
        <p>{mitigation_delta:.3f}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

bias_df = pd.DataFrame(bias_results.items(), columns=["Column", "Bias Score"]).sort_values(
    by="Bias Score",
    ascending=False,
)

section_label("Breakdown")
st.subheader("Top Biased Columns")
st.dataframe(bias_df, use_container_width=True)

st.subheader("Bias Distribution")
render_bar_chart(bias_df)

if show_heatmap:
    section_label("Relationships")
    st.subheader("Correlation Heatmap")
    render_heatmap(encoded_data)

section_label("Narrative")
st.subheader("AI Insights")
ai_text = generate_ai_insight(sensitive_column, before, after)
st.info(ai_text)

report_bytes = build_pdf_report(
    dataset_shape=data.shape,
    target_column=target_column,
    model_name=model_choice,
    accuracy=accuracy,
    most_biased_column=sensitive_column,
    before=before,
    after=after,
    insight_text=ai_text,
)

st.download_button(
    label="Download PDF Report",
    data=report_bytes,
    file_name="faircheck_ai_report.pdf",
    mime="application/pdf",
)
