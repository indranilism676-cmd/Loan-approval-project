import streamlit as st
import pandas as pd
import joblib
model = joblib.load("loan_model3.pkl")
scaler = joblib.load("scaler3.pkl")
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered")
st.title("🏦 Loan Approval Prediction System")
st.write("Enter the applicant details below and click Predict.")
st.markdown("---")
person_age = st.number_input(
    "Person Age",
    min_value=18,
    max_value=100,
    value=30)
person_income = st.number_input(
    "Person Income",
    min_value=1000.0,
    value=50000.0)
person_emp_exp = st.number_input(
    "Employment Experience (Years)",
    min_value=0,
    max_value=50,
    value=5)
loan_amnt = st.number_input(
    "Loan Amount",
    min_value=1000,
    value=10000)
loan_int_rate = st.number_input(
    "Loan Interest Rate (%)",
    min_value=0.0,
    max_value=40.0,
    value=10.0)
loan_percent_income = st.number_input(
    "Loan Percent Income",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.01)
cb_person_cred_hist_length = st.number_input(
    "Credit History Length",
    min_value=0.0,
    value=10.0)
credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=700)
st.markdown("---")

if st.button("Predict Loan Status"):

    input_df = pd.DataFrame({
        "person_age": [person_age],
        "person_income": [person_income],
        "person_emp_exp": [person_emp_exp],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length],
        "credit_score": [credit_score]
    })

   
    input_scaled = scaler.transform(input_df)

    
    prediction = model.predict(input_scaled)[0]

   
    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")

    if( prediction == 1:)
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.subheader("Prediction Probability")

    st.metric(
        label="Approval Probability",
        value=f"{probability[1]*100:.2f}%"
    )

    st.metric(
        label="Rejection Probability",
        value=f"{probability[0]*100:.2f}%"
    )

    st.progress(float(probability[1]))

    st.subheader("Applicant Details")

    st.dataframe(input_df)
