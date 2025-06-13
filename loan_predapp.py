import streamlit as st
import numpy as np
import joblib
import pandas as pd
 
features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History']

joblib.dump(features, 'features.pkl')

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
 
st.title("Loan Approval Prediction App")
 
def user_input_features():
    gender = st.selectbox('Gender',['Male', 'Female'])
    married = st.selectbox('Married',['Yes', 'No'])
    dependents = st.selectbox('Dependents',['0', '1', '2', '3+'])
    education = st.selectbox('Education',['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed',['Yes', 'No'])
    applicant_income = st.number_input('Applicant Income')
    coapplicant_income = st.number_input('Coapplicant Income')
    loan_amount = st.number_input('Loan Amount')
    loan_term = st.number_input('Loan Amount Term')
    credit_history = st.selectbox('Credit History',[0.0, 1.0])
 
    data = {
        'Gender': 0 if gender == 'Male' else 1,
        'Married': 1 if married == 'Yes' else 0,
        'Dependents': int(dependents[0]),
        'Education': 0 if education == 'Graduate' else 1,
        'Self_Employed': 1 if self_employed == 'Yes' else 0,
        'ApplicantIncome':applicant_income,
        'CoapplicantIncome':coapplicant_income,
        'LoanAmount':loan_amount,
        'Loan_Amount_Term':loan_term,
        'Credit_History':credit_history,
    }
 
    return pd.DataFrame([data])

input_df = user_input_features()
 
if st.button('Predict'):
    try:
        input_df = input_df[features]  
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        result = 'Approved' if prediction[0]==1 else 'Rejected'
        st.success(f'Loan Status:{result}')
    except ValueError as e:
        st.error(f'Input error:{e}')