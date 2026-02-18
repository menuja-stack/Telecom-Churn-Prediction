
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
# 5. PREDICTION & SMART SUGGESTION LOGIC
# ==========================================
    # 1. Make Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # 2. Display Result
    st.subheader('Prediction Result')
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK: Customer is likely to CHURN.")
        st.metric(label="Churn Probability", value=f"{probability:.2%}")
        
        # ==========================================
        # 6. EXPLAINABLE AI (Smart Suggestions)
        # ==========================================
        st.subheader("💡 Why is this customer leaving?")
        
        # A. Get the Model Coefficients (The "Weights")
        # This tells us how much the model hates/loves each feature generally
        feature_names = model_columns
        coefficients = model.coef_[0]
        
        # B. Calculate Impact for THIS Customer
        # Impact = Coefficient * Customer_Value
        # This shows us exactly what is driving the score for THIS specific person
        input_values = input_data.values[0]
        impacts = coefficients * input_values
        
        # C. Create a DataFrame to sort them
        impact_df = pd.DataFrame({
            'Feature': feature_names,
            'Impact': impacts
        })
        
        # D. Get the Top 3 "Pain Points" (Positive Impact = Pushing towards Churn)
        top_drivers = impact_df.sort_values(by='Impact', ascending=False).head(3)
        
        # E. The "Strategy Dictionary" - Mapping Problems to Solutions
        suggestions_dict = {
            'MonthlyCharges': "💰 **Price Sensitivity:** This customer is paying a high monthly rate. Suggest a **15% discount** or a downgrade to a cheaper                plan.",
            'TotalCharges': "💳 **Long-term Cost:** They have spent a lot over time. Recognize their loyalty with a **'Platinum Member' perk**.",
            'tenure': "👶 **New Customer Risk:** They are new and haven't formed a habit yet. Call them to ensure onboarding went smoothly.",
            'InternetService_Fiber optic': "⚡ **Service Quality:** Fiber customers often leave due to outages or high cost. Check their **technical support                logs** immediately.",
            'Contract_Month-to-month': "📅 **Contract Risk:** They are on a rolling contract. Offer a **free month** if they switch to a 1-year term.",
            'PaymentMethod_Electronic check': "📝 **Payment Friction:** Electronic checks fail often. Suggest switching to **Auto-Pay (Credit Card)** for a $5              Zdiscount.",
            'TechSupport_No': "🛠️ **Lack of Support:** They have no tech support. Offer a **free 3-month trial** of Premium Support."
        }
        
        # F. Display the Dynamic Suggestions
        for index, row in top_drivers.iterrows():
            feature = row['Feature']
            
            # Only show if it's actually pushing them to churn (Impact > 0)
            if row['Impact'] > 0:
                # 1. Check if we have a specific rule for this feature
                if feature in suggestions_dict:
                    st.info(suggestions_dict[feature])
                
                # 2. Special Logic for One-Hot Encoded features (e.g., Contract_One year)
                elif feature.startswith('Contract_') and 'One year' not in feature and 'Two year' not in feature:
                     st.info(suggestions_dict['Contract_Month-to-month'])
                     
                elif feature.startswith('InternetService_') and 'Fiber' in feature:
                     st.info(suggestions_dict['InternetService_Fiber optic'])
                     
                else:
                    # Fallback for other features
                    st.write(f"⚠️ **Factor:** {feature} is contributing to the risk.")

    else:
        st.success(f"✅ LOW RISK: Customer is likely to STAY.")
        st.metric(label="Churn Probability", value=f"{probability:.2%}")
        st.write("Suggestion: Keep engaging with standard loyalty newsletters.")