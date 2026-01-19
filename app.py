import streamlit as st
import pandas as pd
import joblib

model = joblib.load("student_performance_model.pkl")
features = joblib.load("model_features.pkl")

st.title("Student Performance Prediction")

st.write("Enter student details to predict final grade")

input_data = {}

for feature in features:
    input_data[feature] = st.number_input(feature, value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Final Grade: {prediction:.2f}")
