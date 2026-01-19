import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance Prediction")

st.title("Student Performance Prediction")
st.write("Predict a student's final grade based on academic inputs.")

model = joblib.load("student_performance_model.pkl")
features = joblib.load("model_features.pkl")

st.subheader("Enter Student Details")

G1 = st.number_input("First Period Grade (G1)", min_value=0, max_value=20, value=10)
G2 = st.number_input("Second Period Grade (G2)", min_value=0, max_value=20, value=10)
studytime = st.number_input("Weekly Study Time (1–4)", min_value=1, max_value=4, value=2)
failures = st.number_input("Number of Past Failures", min_value=0, max_value=3, value=0)
absences = st.number_input("Number of Absences", min_value=0, max_value=100, value=5)

if st.button("Predict Final Grade"):
    input_data = {
        "G1": G1,
        "G2": G2,
        "studytime": studytime,
        "failures": failures,
        "absences": absences
    }

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=features, fill_value=0)

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Final Grade: {prediction:.2f}")
