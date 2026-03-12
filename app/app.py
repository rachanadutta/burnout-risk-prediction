import streamlit as st
import sys
import os
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# ===============================
# Load Model
# ===============================

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "burnout_model.pkl")
model_pipeline = joblib.load(model_path)

rf_model = model_pipeline.named_steps["model"]
explainer = shap.TreeExplainer(rf_model)

# ===============================
# Allow importing from src
# ===============================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import predict_burnout

# ===============================
# Load Dataset
# ===============================

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(base_dir, "data", "processed", "clean_student_lifestyle.csv")

df = pd.read_csv(data_path)

departments = sorted(df["Department"].unique())
genders = sorted(df["Gender"].unique())

# ===============================
# App Header
# ===============================

st.title("BurnoutGuard")

st.caption(
    "An Explainable Machine Learning System for Early Detection of Student Burnout."
)
st.markdown("---")

st.write(
    "Adjust the values below to estimate your burnout risk for the current week."
)


# ===============================
# Layout (Two Columns)
# ===============================

col1, spacer, col2 = st.columns([1, 0.2, 1])

with col1:

    st.header("Student Inputs")

    age = st.slider("Age", 18, 30, 21)

    cgpa = st.slider("CGPA", 0.0, 4.0, 3.0)

    sleep = st.slider("Sleep Duration (hours per night)", 0.0, 10.0, 7.0)

    study = st.slider("Study Hours per day", 0.0, 12.0, 5.0)

    social_media = st.slider("Social Media Hours per day", 0.0, 10.0, 2.0)

    exercise = st.slider(
        "Physical Activity (minutes per day)",
        0,
        120,
        30
    )

    gender = st.selectbox("Gender", genders)

    department = st.selectbox("Department", departments)

    st.markdown("### ")


# ===============================
# Prediction
# ===============================

if st.button("Estimate Burnout Risk"):

    input_data = {
        "Age": age,
        "CGPA": cgpa,
        "Sleep_Duration": sleep,
        "Study_Hours": study,
        "Social_Media_Hours": social_media,
        "Physical_Activity": exercise,
        "Gender": gender,
        "Department": department
    }

    risk_percent, risk_level = predict_burnout(input_data)

    input_df = pd.DataFrame([input_data])

    # ===============================
    # SHAP Explanation
    # ===============================

    processed_input = model_pipeline.named_steps["preprocessor"].transform(input_df)

    shap_values = explainer.shap_values(processed_input)

    feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()

    # SHAP output handling (works for both formats)

    # SHAP output handling

    

    if isinstance(shap_values, list):
        shap_importance = shap_values[1][0]
    else:
        shap_importance = shap_values

    # flatten to 1D
    shap_importance = np.array(shap_importance).flatten()

    # Ensure lengths match
    min_len = min(len(feature_names), len(shap_importance))

    shap_df = pd.DataFrame({
        "feature": feature_names[:min_len],
        "importance": shap_importance[:min_len]
    })

    shap_df["feature"] = (
    shap_df["feature"]
    .str.replace("num__", "")
    .str.replace("cat__", "")
    .str.replace("_", " ")
)

    # Attach actual input values
    shap_df["value"] = processed_input[0]

    # Remove encoded features that are zero
    shap_df = shap_df[shap_df["value"] != 0]

    # Sort by importance
    top_features = shap_df.reindex(
    shap_df["importance"].abs().sort_values(ascending=False).index
).head(5)

    # ===============================
    # Prediction Results (Column 2)
    # ===============================

    with col2:
        st.header("Burnout Risk Assessment") 

        st.metric("Burnout Risk", f"{risk_percent:.2f}%")

        # Burnout Risk Gauge
        st.markdown("### Burnout Risk Meter")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_percent,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 30], 'color': "#2ecc71"},   # green
                    {'range': [30, 60], 'color': "#f1c40f"},  # yellow
                    {'range': [60, 100], 'color': "#e74c3c"}  # red
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        if risk_level == "Low":
            st.success(f"Risk Level: {risk_level}")

        elif risk_level == "Moderate":
            st.warning(f"Risk Level: {risk_level}")

        else:
            st.error(f"Risk Level: {risk_level}")

        # ===============================
        # Lifestyle Balance Score
        # ===============================

        lifestyle_score = (
            (sleep / 10) * 25 +
            (1 - study / 12) * 25 +
            (exercise / 120) * 25 +
            (1 - social_media / 10) * 25
        )

        st.subheader("Lifestyle Balance Score")

        st.metric("Balance Score", f"{int(lifestyle_score)}/100")

        # ===============================
        # Top Factors
        # ===============================

        st.subheader("Top Factors Influencing Burnout Risk")

        for _, row in top_features.iterrows():
            st.write(f"• {row['feature']}")

        # ===============================
        # SHAP Bar Chart
        # ===============================

        st.subheader("Feature Contribution Visualization")

        fig, ax = plt.subplots()

        colors = ["#e74c3c" if x > 0 else "#2ecc71" for x in top_features["importance"]]

        ax.barh(top_features["feature"], top_features["importance"], color=colors)

        ax.axvline(0, color="black", linewidth=1)
        ax.invert_yaxis()

        ax.set_xlabel("Impact on Burnout Risk")
        ax.set_ylabel("Feature")

        st.pyplot(fig)

        # ===============================
        # Model-Driven Recommendations
        # ===============================

        st.subheader("Suggested Improvements")

        for feature in top_features["feature"]:

            f = feature.lower()

            if "sleep" in f:
                st.write("• Try increasing sleep duration to at least 7 hours.")

            elif "study" in f:
                st.write("• Consider taking breaks during long study sessions.")

            elif "physical" in f:
                st.write("• Increasing daily physical activity may reduce burnout risk.")

            elif "social media" in f:
                st.write("• Reducing social media usage may improve focus and wellbeing.")


# ===============================
# Dataset Insights
# ===============================

st.markdown("---")
st.markdown("### ")

st.subheader("Global Drivers of Burnout Risk")

# Create centered layout
left_space, chart_col, right_space = st.columns([1,2,1])

with chart_col:

    importances = rf_model.feature_importances_
    feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()

    global_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    global_df["feature"] = (
        global_df["feature"]
        .str.replace("num__", "")
        .str.replace("cat__", "")
        .str.replace("_", " ")
    )

    global_df = global_df.sort_values(by="importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(6,4))

    ax.barh(global_df["feature"], global_df["importance"])

    ax.invert_yaxis()

    ax.set_title("Top Features Affecting Burnout")
    ax.set_xlabel("Importance Score")

    st.pyplot(fig)

st.subheader("Dataset Insights")

st.bar_chart(df["burnout"].value_counts())

# ===============================
# Model Information
# ===============================

with st.expander("About the Model"):

    st.write("""
    **Model Type:** Random Forest Classifier  

    **Training Dataset:** 100,000 student lifestyle records  

    **Key Features Used:**
    - Sleep Duration
    - Study Hours
    - Social Media Usage
    - Physical Activity
    - CGPA
    - Demographic information

    **Evaluation Metric:**
    ROC-AUC ≈ 0.88

    **Explainability:**
    SHAP values are used to highlight the factors contributing most to burnout risk.
    """)

# ===============================
# Disclaimer
# ===============================

st.caption(
    "Disclaimer: This tool estimates behavioral burnout risk and does not provide medical diagnosis."
)