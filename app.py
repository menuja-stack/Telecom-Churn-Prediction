
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. LOAD THE SAVED MODELS
# ==========================================
# We use st.cache so the app is fast and doesn't reload the model every time
@st.cache_resource
def load_data():
    # Make sure these .pkl files are in the same folder as this app.py file!
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns

model, scaler, model_columns = load_data()

# ==========================================
# 2. APP TITLE & DESIGN
# ==========================================
st.title("📉 Telco Customer Churn Predictor")
st.write("Enter customer details below to predict if they will leave.")

# ==========================================
# 3. SIDEBAR INPUTS
# ==========================================
st.sidebar.header('Customer Profile')

# Numeric Inputs
tenure = st.sidebar.slider('Tenure (Months)', 0, 72, 12)
monthly_charges = st.sidebar.number_input('Monthly Charges ($)', min_value=18.0, max_value=120.0, value=70.0)
total_charges = st.sidebar.number_input('Total Charges ($)', min_value=0.0, max_value=9000.0, value=monthly_charges * tenure)

# Categorical Inputs
contract = st.sidebar.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
internet = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
payment = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
tech_support = st.sidebar.selectbox('Tech Support?', ['No', 'Yes', 'No internet service'])

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
if st.button('Predict Churn Risk'):
    
    # A. Create a row of zeros (the default "empty" customer)
    # This ensures we have all 30 columns the model expects
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # B. Fill in the numerical values
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    input_data['TotalCharges'] = total_charges
    
    # C. Handle Categorical Inputs (Manual One-Hot Encoding)
    # We set the specific column to 1 if the user selected it
    
    # Contract
    if contract == 'One year': input_data['Contract_One year'] = 1
    if contract == 'Two year': input_data['Contract_Two year'] = 1
    
    # Internet
    if internet == 'Fiber optic': input_data['InternetService_Fiber optic'] = 1
    if internet == 'No': input_data['InternetService_No'] = 1
        
    # Payment
    if payment == 'Electronic check': input_data['PaymentMethod_Electronic check'] = 1
    if payment == 'Credit card (automatic)': input_data['PaymentMethod_Credit card (automatic)'] = 1
    if payment == 'Mailed check': input_data['PaymentMethod_Mailed check'] = 1
        
    # Tech Support
    if tech_support == 'Yes': input_data['TechSupport_Yes'] = 1
    if tech_support == 'No internet service': input_data['TechSupport_No internet service'] = 1

    # D. Scale the Numerical Columns (CRITICAL)
    # The model expects "scaled" numbers, not raw dollars
    input_data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(input_data[['tenure', 'MonthlyCharges', 'TotalCharges']])

    # ==========================================
    # 5. DISPLAY RESULTS
    # ==========================================
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader('Prediction Result')
    
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK: Customer is likely to CHURN.")
        st.metric(label="Churn Probability", value=f"{probability:.2%}")
        st.write("Suggestion: Offer a 1-year contract discount immediately.")
    else:
        st.success(f"✅ LOW RISK: Customer is likely to STAY.")
        st.metric(label="Churn Probability", value=f"{probability:.2%}")