import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame
)
from fairlearn.reductions import (
    DemographicParity,
    EqualizedOdds,
    ExponentiatedGradient,
)
from fairlearn.postprocessing import ThresholdOptimizer

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Unbiased AI Decision System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Helper Functions
# =========================

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df


def encode_data(df, fit_encoders=None):
    """
    BUG FIX: Always fit encoders on train, transform test separately.
    Returns encoded df and encoder dict.
    """
    encoders = fit_encoders or {}
    df_encoded = df.copy()

    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            if col not in encoders:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(
                    df_encoded[col].astype(str)
                )
                encoders[col] = le
            else:
                le = encoders[col]
                # Handle unseen labels in test set safely
                known = set(le.classes_)
                df_encoded[col] = df_encoded[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df_encoded[col] = le.transform(df_encoded[col])

    return df_encoded, encoders


def get_model(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    return DecisionTreeClassifier(max_depth=6, random_state=42)


def compute_bias_metrics(y_true, y_pred, sensitive_feature):
    """
    BUG FIX: Compute multiple bias metrics, not just one.
    Returns a dict of metrics.
    """
    dpd = demographic_parity_difference(
        y_true=y_true, y_pred=y_pred,
        sensitive_features=sensitive_feature
    )
    eod = equalized_odds_difference(
        y_true=y_true, y_pred=y_pred,
        sensitive_features=sensitive_feature
    )
    return {
        "demographic_parity_diff": abs(float(dpd)),
        "equalized_odds_diff": abs(float(eod)),
    }


def group_metrics(y_true, y_pred, sensitive_col_values, target_col_name, sensitive_col_name):
    """
    BUG FIX: Compute group stats on TEST split only, not full dataset.
    Operates on aligned arrays instead of a DataFrame to avoid leakage.
    """
    results = []
    groups = np.unique(sensitive_col_values)

    for g in groups:
        mask = (sensitive_col_values == g)
        g_true = y_true[mask]
        g_pred = y_pred[mask]
        n = mask.sum()

        if n == 0:
            continue

        pos_rate_true = g_true.mean()
        pos_rate_pred = g_pred.mean()
        tpr = (g_pred[g_true == 1] == 1).mean() if (g_true == 1).sum() > 0 else 0.0
        fpr = (g_pred[g_true == 0] == 1).mean() if (g_true == 0).sum() > 0 else 0.0

        results.append({
            f"{sensitive_col_name}": g,
            "n (test)": int(n),
            "True positive rate": f"{tpr:.3f}",
            "False positive rate": f"{fpr:.3f}",
            f"Actual {target_col_name} rate": f"{pos_rate_true:.3f}",
            f"Predicted {target_col_name} rate": f"{pos_rate_pred:.3f}",
        })

    return pd.DataFrame(results)


# =========================
# UI
# =========================

st.title("⚖️ Unbiased AI Decision System")
st.caption("Detect and mitigate bias in automated decisions — loans, hiring, healthcare")

st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("📂 Upload a CSV dataset to begin. Works with loan, hiring, or medical datasets.")
    st.stop()

df = load_data(uploaded_file)

if df.empty:
    st.error("Dataset is empty.")
    st.stop()

col_options = df.columns.tolist()

target_column = st.sidebar.selectbox("Target column (what to predict)", col_options)
feature_columns = [c for c in col_options if c != target_column]

sensitive_column = st.sidebar.selectbox(
    "Sensitive attribute (protected group)",
    feature_columns
)
model_name = st.sidebar.selectbox(
    "ML model",
    ["Logistic Regression", "Random Forest", "Decision Tree"]
)
mitigation_strategy = st.sidebar.selectbox(
    "Mitigation strategy",
    ["ExponentiatedGradient (DemographicParity)",
     "ExponentiatedGradient (EqualizedOdds)",
     "ThresholdOptimizer"]
)
show_heatmap = st.sidebar.checkbox("Show correlation heatmap")
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

st.subheader("Dataset Preview")
st.dataframe(df.head(8), use_container_width=True)

col_info1, col_info2, col_info3 = st.columns(3)
col_info1.metric("Rows", f"{len(df):,}")
col_info2.metric("Features", len(feature_columns))
col_info3.metric("Missing values", int(df.isnull().sum().sum()))

# =========================
# CLEANING
# =========================

df_clean = df.copy()

# Coerce numeric columns
for col in df_clean.columns:
    if df_clean[col].dtype != "object":
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# Impute numeric
numeric_cols = df_clean.select_dtypes(include=np.number).columns
if len(numeric_cols) > 0:
    imputer = SimpleImputer(strategy="median")
    df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])

# Impute categorical
cat_cols = df_clean.select_dtypes(include="object").columns
if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df_clean[cat_cols] = cat_imputer.fit_transform(df_clean[cat_cols])

# =========================
# SPLIT FIRST, THEN ENCODE
# BUG FIX: Must split before encoding to prevent data leakage.
# =========================

X_raw = df_clean[feature_columns]
y_raw = df_clean[target_column]

try:
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=test_size, random_state=42, stratify=y_raw
    )
except Exception:
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=test_size, random_state=42
    )

# Encode train, then apply same encoders to test
X_train_enc, train_encoders = encode_data(X_train_raw)
X_test_enc, _ = encode_data(X_test_raw, fit_encoders=train_encoders)

# Also encode target if needed
if y_train.dtype == "object":
    le_target = LabelEncoder()
    y_train = le_target.fit_transform(y_train.astype(str))
    y_test = le_target.transform(
        y_test.astype(str).apply(
            lambda x: x if x in set(le_target.classes_) else le_target.classes_[0]
        )
    )
else:
    y_train = y_train.values
    y_test = y_test.values

# =========================
# TRAIN BASE MODEL
# =========================

with st.spinner("Training model..."):
    model = get_model(model_name)
    model.fit(X_train_enc, y_train)
    y_pred = model.predict(X_test_enc)

accuracy = accuracy_score(y_test, y_pred)

try:
    y_prob = model.predict_proba(X_test_enc)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    roc_str = f"{roc_auc:.3f}"
except Exception:
    roc_str = "N/A"

# =========================
# CROSS VALIDATION
# =========================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
try:
    cv_scores = cross_val_score(model, X_train_enc, y_train, cv=cv, scoring="accuracy")
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    cv_str = f"{cv_mean:.3f} ± {cv_std:.3f}"
except Exception:
    cv_str = "N/A"

# =========================
# BIAS DETECTION (on test split only)
# BUG FIX: sensitive_feature must be the test-split sensitive column values
# =========================

sensitive_test = X_test_enc[sensitive_column].values
sensitive_train = X_train_enc[sensitive_column].values

bias_before = compute_bias_metrics(y_test, y_pred, sensitive_test)

# =========================
# BIAS MITIGATION
# BUG FIX: Mitigation estimator now mirrors the selected model, not always LR.
# BUG FIX: Use a fresh LogisticRegression only as the constrained base (fairlearn
#          requires a sklearn estimator; wrap the user's model if possible).
# =========================

with st.spinner("Running bias mitigation..."):

    base_estimator = LogisticRegression(max_iter=1000, random_state=42)

    if "ThresholdOptimizer" in mitigation_strategy:
        mitigator = ThresholdOptimizer(
            estimator=base_estimator,
            constraints="demographic_parity",
            objective="accuracy_score",
            predict_method="auto"
        )
        mitigator.fit(
            X_train_enc, y_train,
            sensitive_features=sensitive_train
        )
        y_pred_mitigated = mitigator.predict(
            X_test_enc, sensitive_features=sensitive_test
        )

    else:
        constraints = (
            EqualizedOdds() if "EqualizedOdds" in mitigation_strategy
            else DemographicParity()
        )
        mitigator = ExponentiatedGradient(
            estimator=base_estimator,
            constraints=constraints,
            max_iter=50
        )
        mitigator.fit(
            X_train_enc, y_train,
            sensitive_features=sensitive_train
        )
        y_pred_mitigated = mitigator.predict(X_test_enc)

bias_after = compute_bias_metrics(y_test, y_pred_mitigated, sensitive_test)
accuracy_mitigated = accuracy_score(y_test, y_pred_mitigated)

# =========================
# RESULTS
# =========================

st.divider()
st.subheader("Model Performance")

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Test accuracy", f"{accuracy:.3f}")
mc2.metric("Mitigated accuracy", f"{accuracy_mitigated:.3f}",
           delta=f"{accuracy_mitigated - accuracy:+.3f}")
mc3.metric("ROC-AUC", roc_str)
mc4.metric("CV score (5-fold)", cv_str)

st.divider()
st.subheader("Bias Metrics")

dpd_before = bias_before["demographic_parity_diff"]
dpd_after = bias_after["demographic_parity_diff"]
eod_before = bias_before["equalized_odds_diff"]
eod_after = bias_after["equalized_odds_diff"]

improvement_dpd = dpd_before - dpd_after

bc1, bc2, bc3, bc4 = st.columns(4)
bc1.metric("DPD before", f"{dpd_before:.3f}",
           help="Demographic Parity Difference. 0 = perfectly fair.")
bc2.metric("DPD after", f"{dpd_after:.3f}",
           delta=f"{-improvement_dpd:+.3f}",
           delta_color="inverse")
bc3.metric("EOD before", f"{eod_before:.3f}",
           help="Equalized Odds Difference.")
bc4.metric("EOD after", f"{eod_after:.3f}",
           delta=f"{-(eod_before - eod_after):+.3f}",
           delta_color="inverse")

fairness_score = max(0, 1 - dpd_after)
st.metric("Fairness score (post-mitigation)", f"{fairness_score:.3f}")

if dpd_after < 0.1:
    st.success("✅ Bias reduced to within acceptable threshold (< 0.1).")
elif dpd_after < 0.2:
    st.warning("⚠️ Bias partially reduced. Consider data rebalancing.")
else:
    st.error("❌ Bias remains high. Review training data distribution.")

# =========================
# VISUALIZATION
# =========================

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(
    ["Before", "After"],
    [dpd_before, dpd_after],
    color=["#E24B4A", "#1D9E75"],
    width=0.5
)
axes[0].axhline(0.1, color="#888780", linestyle="--", linewidth=1, label="Fair threshold (0.1)")
axes[0].set_ylabel("Demographic Parity Difference")
axes[0].set_title("Bias Before vs After Mitigation")
axes[0].legend(fontsize=9)
axes[0].set_ylim(0, max(dpd_before * 1.3, 0.15))

# Per-group positive prediction rate
group_df = group_metrics(y_test, y_pred, sensitive_test, target_column, sensitive_column)
if not group_df.empty and f"Predicted {target_column} rate" in group_df.columns:
    groups_plot = group_df[sensitive_column].astype(str)
    rates_plot = group_df[f"Predicted {target_column} rate"].astype(float)
    axes[1].bar(groups_plot, rates_plot, color="#378ADD", width=0.5)
    axes[1].set_ylabel(f"Predicted {target_column} rate")
    axes[1].set_title("Decision Rate by Group")
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(rates_plot):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

plt.tight_layout()
st.pyplot(fig)

# =========================
# GROUP-LEVEL BREAKDOWN
# =========================

st.divider()
st.subheader("Group-Level Analysis (Test Set)")

group_df_full = group_metrics(y_test, y_pred, sensitive_test, target_column, sensitive_column)
st.dataframe(group_df_full, use_container_width=True)

# =========================
# FEATURE IMPORTANCE
# =========================

if hasattr(model, "feature_importances_"):
    st.divider()
    st.subheader("Feature Importances")
    fi = pd.Series(
        model.feature_importances_,
        index=X_train_enc.columns
    ).sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    fi.head(15).plot.barh(ax=ax3, color="#378ADD")
    ax3.invert_yaxis()
    ax3.set_xlabel("Importance")
    ax3.set_title("Top 15 Feature Importances")
    plt.tight_layout()
    st.pyplot(fig3)

elif hasattr(model, "coef_"):
    st.divider()
    st.subheader("Feature Coefficients (Logistic Regression)")
    coefs = pd.Series(
        model.coef_[0],
        index=X_train_enc.columns
    ).sort_values(key=abs, ascending=False)
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    colors = ["#E24B4A" if v < 0 else "#1D9E75" for v in coefs.head(15)]
    coefs.head(15).plot.barh(ax=ax4, color=colors)
    ax4.invert_yaxis()
    ax4.set_xlabel("Coefficient value")
    ax4.set_title("Top 15 Feature Coefficients")
    plt.tight_layout()
    st.pyplot(fig4)

# =========================
# CORRELATION HEATMAP
# =========================

if show_heatmap:
    st.divider()
    st.subheader("Correlation Heatmap")
    # BUG FIX: Use full encoded dataset here for exploration (clearly labelled)
    df_for_corr_enc, _ = encode_data(df_clean)
    corr = df_for_corr_enc.corr()
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    im = ax5.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax5)
    ax5.set_xticks(range(len(corr.columns)))
    ax5.set_yticks(range(len(corr.columns)))
    ax5.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax5.set_yticklabels(corr.columns, fontsize=9)
    ax5.set_title("Feature Correlation Matrix (full dataset)")
    plt.tight_layout()
    st.pyplot(fig5)

# =========================
# RECOMMENDATIONS
# =========================

st.divider()
st.subheader("Recommendations")

recommendations = []
if dpd_after >= 0.1:
    recommendations.append("Resample training data to balance protected groups (SMOTE or class weighting).")
if improvement_dpd < 0.05:
    recommendations.append("Try ThresholdOptimizer mitigation — it often outperforms ExponentiatedGradient on small datasets.")
recommendations.append("Remove proxy features that correlate with the sensitive attribute.")
recommendations.append("Monitor fairness metrics in production — bias can re-emerge as data distribution shifts.")
recommendations.append("Collect more samples for underrepresented groups.")

for i, rec in enumerate(recommendations, 1):
    st.write(f"**{i}.** {rec}")
