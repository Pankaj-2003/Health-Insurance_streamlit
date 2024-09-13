import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('best_lgbm_model.joblib')

model = load_model()

# Streamlit app
st.title('Insurance Prediction App')

# Create input fields for features
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=18, max_value=100, value=30)
driving_license = st.selectbox('Driving License', [0, 1])
region_code = st.number_input('Region Code', min_value=0, value=28)
previously_insured = st.selectbox('Previously Insured', [0, 1])
annual_premium = st.number_input('Annual Premium', min_value=0, value=30000)
policy_sales_channel = st.number_input('Policy Sales Channel', min_value=1, max_value=163, value=26)
vintage = st.number_input('Vintage (in days)', min_value=0, max_value=300, value=150)
vehicle_age_lt_1_year = st.selectbox('Vehicle Age < 1 Year', [0, 1])
vehicle_age_gt_2_years = st.selectbox('Vehicle Age > 2 Years', [0, 1])
vehicle_damage_yes = st.selectbox('Vehicle Damage', [0, 1])

# When the user clicks the predict button
if st.button('Predict'):
    # Create a dataframe from the input
    input_data = pd.DataFrame([[gender, age, driving_license, region_code, previously_insured,
                                annual_premium, policy_sales_channel, vintage,
                                vehicle_age_lt_1_year, vehicle_age_gt_2_years, vehicle_damage_yes]],
                              columns=['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
                                       'Annual_Premium', 'Policy_Sales_Channel', 'Vintage',
                                       'Vehicle_Age_lt_1_Year', 'Vehicle_Age_gt_2_Years',
                                       'Vehicle_Damage_Yes'])
    
    # Convert 'Gender' to numeric (assuming 'Male' = 0, 'Female' = 1)
    input_data['Gender'] = input_data['Gender'].map({'Male': 0, 'Female': 1})
    
    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    # Display the prediction
    st.write(f'Prediction: {"Interested" if prediction[0] == 1 else "Not Interested"}')
    st.write(f'Probability of being interested: {probability:.2f}')

# Add some information about the features
st.sidebar.header('Feature Information')
st.sidebar.write("""
- Gender: Male or Female
- Age: Age of the customer
- Driving_License: 0 for No, 1 for Yes
- Region_Code: Unique code for the region
- Previously_Insured: 0 for No, 1 for Yes
- Annual_Premium: The amount customer needs to pay as premium in a year
- Policy_Sales_Channel: Anonymized Code for the channel of outreaching to the customer
- Vintage: Number of Days, Customer has been associated with the company
- Vehicle_Age_lt_1_Year: 1 if Vehicle Age < 1 Year, else 0
- Vehicle_Age_gt_2_Years: 1 if Vehicle Age > 2 Years, else 0
- Vehicle_Damage_Yes: 1 if vehicle has been damaged in the past, else 0
""")