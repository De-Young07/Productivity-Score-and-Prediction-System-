import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# ================================
# PATHS
# ================================

ROOT = Path(__file__).resolve().parents[1]

MODELS = ROOT / "models"
RAW_DATA = ROOT / "Datasets" / "raw"
PROCESSED_DATA = ROOT / "Datasets" / "processed"
REPORTS = ROOT / "reports"

# ================================
# LOAD MODELS
# ================================

rf_pipeline = joblib.load(MODELS / "random_forest.pkl")
xgb_pipeline = joblib.load(MODELS / "xgboost.pkl")

# ================================
# LOAD DATA
# ================================

raw_df = pd.read_csv(RAW_DATA / "social_media_vs_productivity.csv")
raw_df.columns = raw_df.columns.str.strip().str.lower()

train_df = pd.read_csv(PROCESSED_DATA / "train.csv")
test_df = pd.read_csv(PROCESSED_DATA / "test.csv")

performance = pd.read_csv(REPORTS / "tables" / "model_performance.csv")

target = "actual_productivity_score"

X_test = test_df.drop(columns=[target])
y_test = test_df[target]

# ================================
# SAFE COLUMN ACCESS
# ================================

def get_unique(df, col):
    if col in df.columns:
        return sorted(df[col].dropna().unique())
    else:
        return []

# ================================
# PAGE CONFIG
# ================================

st.set_page_config(page_title="Productivity ML Dashboard", layout="wide")

st.title("Productivity Prediction System")

# ================================
# SIDEBAR INPUTS (FROM RAW DATA)
# ================================

st.sidebar.header("Input Variables")

age = st.sidebar.slider("Age", 18, 65, 30)

gender = st.sidebar.selectbox(
    "Gender",
    get_unique(raw_df, "gender")
)

job_type = st.sidebar.selectbox(
    "Job Type",
    get_unique(raw_df, "job_type")
)

daily_social_media_time = st.sidebar.slider("Daily Social Media Time", 0.0, 10.0, 3.0)

social_platform_preference = st.sidebar.selectbox(
    "Preferred Platform",
    get_unique(raw_df, "social_platform_preference")
)

number_of_notifications = st.sidebar.slider("Notifications Per Day", 0, 200, 50)

work_hours_per_day = st.sidebar.slider("Work Hours Per Day", 0.0, 16.0, 8.0)

perceived_productivity_score = st.sidebar.slider("Perceived Productivity", 0.0, 10.0, 5.0)

stress_level = st.sidebar.slider("Stress Level", 0.0, 10.0, 5.0)

sleep_hours = st.sidebar.slider("Sleep Hours", 0.0, 12.0, 7.0)

screen_time_before_sleep = st.sidebar.slider("Screen Time Before Sleep", 0.0, 5.0, 1.0)

breaks_during_work = st.sidebar.slider("Breaks During Work", 0, 10, 3)

uses_focus_apps = st.sidebar.selectbox(
    "Uses Focus Apps",
    get_unique(raw_df, "uses_focus_apps")
)

has_digital_wellbeing_enabled = st.sidebar.selectbox(
    "Digital Wellbeing Enabled",
    get_unique(raw_df, "has_digital_wellbeing_enabled")
)

coffee_consumption_per_day = st.sidebar.slider("Coffee Per Day", 0, 10, 2)

days_feeling_burnout_per_month = st.sidebar.slider("Burnout Days", 0, 30, 5)

weekly_offline_hours = st.sidebar.slider("Weekly Offline Hours", 0.0, 50.0, 10.0)

job_satisfaction_score = st.sidebar.slider("Job Satisfaction", 0.0, 10.0, 5.0)

# ================================
# FEATURE ENGINEERING
# ================================

digital_distraction_score = (
    daily_social_media_time
    + screen_time_before_sleep
    + (number_of_notifications / 100)
)

work_life_balance = weekly_offline_hours / max(work_hours_per_day, 1)

# ================================
# BUILD INPUT DATAFRAME
# ================================

input_df = pd.DataFrame({

    "age":[age],
    "gender":[gender],
    "job_type":[job_type],
    "daily_social_media_time":[daily_social_media_time],
    "social_platform_preference":[social_platform_preference],
    "number_of_notifications":[number_of_notifications],
    "work_hours_per_day":[work_hours_per_day],
    "perceived_productivity_score":[perceived_productivity_score],
    "stress_level":[stress_level],
    "sleep_hours":[sleep_hours],
    "screen_time_before_sleep":[screen_time_before_sleep],
    "breaks_during_work":[breaks_during_work],
    "uses_focus_apps":[uses_focus_apps],
    "has_digital_wellbeing_enabled":[has_digital_wellbeing_enabled],
    "coffee_consumption_per_day":[coffee_consumption_per_day],
    "days_feeling_burnout_per_month":[days_feeling_burnout_per_month],
    "weekly_offline_hours":[weekly_offline_hours],
    "job_satisfaction_score":[job_satisfaction_score],
    "digital_distraction_score":[digital_distraction_score],
    "work_life_balance":[work_life_balance]

})

st.subheader("Input Data")
st.dataframe(input_df)

# ================================
# PREDICTIONS (SAFE)
# ================================

st.subheader("Model Predictions")

try:
    rf_pred = rf_pipeline.predict(input_df)[0]
    xgb_pred = xgb_pipeline.predict(input_df)[0]

    col1, col2 = st.columns(2)
    col1.metric("Random Forest", round(rf_pred, 3))
    col2.metric("XGBoost", round(xgb_pred, 3))

except Exception as e:
    st.error(f"Prediction Error: {e}")

# ================================
# PERFORMANCE
# ================================

st.subheader("Model Performance")

st.dataframe(performance)

fig, ax = plt.subplots()
ax.bar(performance["model"], performance["RMSE"])
st.pyplot(fig)

# ================================
# FEATURE IMPORTANCE
# ================================

st.subheader("Feature Importance")

rf_model = rf_pipeline.named_steps["model"]

features = rf_pipeline.named_steps["preprocessor"].get_feature_names_out()

importance_df = pd.DataFrame({
    "feature": features,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False).head(15)

fig2, ax2 = plt.subplots()
ax2.barh(importance_df["feature"], importance_df["importance"])
ax2.invert_yaxis()

st.pyplot(fig2)

# ================================
# SHAP
# ================================

st.subheader("SHAP Explainability")

try:
    sample = X_test.sample(200)

    transformed = rf_pipeline.named_steps["preprocessor"].transform(sample)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(transformed)

    fig3 = plt.figure()
    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=features,
        show=False
    )

    st.pyplot(fig3)

except Exception as e:
    st.warning(f"SHAP error: {e}")

# ================================
# DISTRIBUTION
# ================================

st.subheader("Target Distribution")

fig4, ax4 = plt.subplots()
sns.histplot(raw_df[target], bins=30, kde=True, ax=ax4)

st.pyplot(fig4)