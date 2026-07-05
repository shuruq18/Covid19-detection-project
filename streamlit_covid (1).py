import streamlit as st
import pandas as pd
import joblib
import base64

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="COVID-19 Detection",
    layout="centered"
)

# ===============================
# Background Image
# ===============================
def set_background(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except:
        pass

set_background("background.png")

# ===============================
# Load Files
# ===============================
model = joblib.load("covid_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

# ===============================
# Title
# ===============================
st.title("🦠 COVID-19 Detection System")
st.markdown("### Machine Learning Based Diagnosis")

# ===============================
# Features
# ===============================
features = [
    "Breathing Problem",
    "Fever",
    "Dry Cough",
    "Sore throat",
    "Running Nose",
    "Asthma",
    "Chronic Lung Disease",
    "Headache",
    "Heart Disease",
    "Diabetes",
    "Hyper Tension",
    "Fatigue ",
    "Gastrointestinal ",
    "Abroad travel",
    "Contact with COVID Patient",
    "Attended Large Gathering",
    "Visited Public Exposed Places",
    "Family working in Public Exposed Places",
    "Wearing Masks",
    "Sanitization from Market"
]

# ===============================
# Input Form
# ===============================
st.subheader("Patient Symptoms")

user_input = {}

for feature in features:
    user_input[feature] = st.selectbox(
        feature.strip(),
        ["No", "Yes"],
        key=feature
    )

# ===============================
# Prediction
# ===============================
if st.button("Predict COVID-19"):

    try:
        input_df = pd.DataFrame([user_input])

        # Encode categorical features
        for col, encoder in encoders.items():

            if col in input_df.columns:

                # تحويل القيم بنفس طريقة التدريب
                input_df[col] = encoder.transform(
                    input_df[col].astype(str)
                )

        # ترتيب الأعمدة بنفس ترتيب التدريب
        input_df = input_df[features]

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.error("⚠️ COVID-19 POSITIVE")
        else:
            st.success("✅ COVID-19 NEGATIVE")

    except Exception as e:
        st.error(f"Error: {e}")