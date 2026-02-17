import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💓 Heart Failure Risk Prediction")
st.write("Enter patient details to predict the risk of heart failure.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
anaemia = st.selectbox("Anaemia (1 = Yes, 0 = No)", [0, 1])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=0, value=100)
diabetes = st.selectbox("Diabetes (1 = Yes, 0 = No)", [0, 1])
ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=1, max_value=100, value=40)
high_blood_pressure = st.selectbox("High Blood Pressure (1 = Yes, 0 = No)", [0, 1])
platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=0, value=250000)
serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, value=1.0)
serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=100, max_value=200, value=135)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
smoking = st.selectbox("Smoking (1 = Yes, 0 = No)", [0, 1])
time = st.number_input("Follow-up Period (days)", min_value=0, value=100)

if st.button("Predict"):
    features = np.array([[age, anaemia, creatinine_phosphokinase, diabetes,
                          ejection_fraction, high_blood_pressure, platelets,
                          serum_creatinine, serum_sodium, sex, smoking, time]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"⚠ High Risk of Heart Failure ({probability*100:.2f}% probability)")
    else:
        st.success(f"✅ Low Risk of Heart Failure ({probability*100:.2f}% probability)")
