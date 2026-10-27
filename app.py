
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction")
st.write(
    "Enter student details to estimate the expected exam score "
    "using a trained Machine Learning model."
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "student_performance_model.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please upload the .pkl file to GitHub.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Read the feature columns saved by the training pipeline
try:
    preprocessor = model.named_steps["preprocessor"]
    feature_columns = list(preprocessor.feature_names_in_)
except Exception:
    st.error(
        "Could not identify model input columns. "
        "Please check the model saved in Kaggle."
    )
    st.stop()

st.subheader("Student Details")

# Defaults for common columns in this dataset
defaults = {
    "Hours_Studied": (0.0, 24.0, 5.0),
    "Attendance": (0.0, 100.0, 80.0),
    "Sleep_Hours": (0.0, 24.0, 7.0),
    "Previous_Scores": (0.0, 100.0, 70.0),
    "Tutoring_Sessions": (0.0, 20.0, 1.0),
    "Physical_Activity": (0.0, 24.0, 3.0),
    "Age": (5.0, 100.0, 16.0),
}

input_data = {}

with st.form("student_form"):
    for col in feature_columns:
        label = col.replace("_", " ")

        if col in defaults:
            minimum, maximum, default = defaults[col]
            input_data[col] = st.number_input(
                label,
                min_value=minimum,
                max_value=maximum,
                value=default
            )
        else:
            input_data[col] = st.text_input(
                label,
                value=""
            )

    submitted = st.form_submit_button("Predict Exam Score")

if submitted:
    input_df = pd.DataFrame([input_data], columns=feature_columns)

    try:
        prediction = float(model.predict(input_df)[0])

        st.success(f"Predicted Exam Score: {prediction:.2f}")
        st.caption(
            "This is an estimate from a machine learning model, "
            "not a guaranteed exam result."
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Please check the input values and ensure all categorical "
            "fields match the format used in the training dataset."
        )
