import streamlit as st
import numpy as np
import joblib

# Load the trained logistic regression model
logreg = joblib.load('logistic_regression_model.pkl')
st.image("engage_jooust_branding.png",caption="Engage brands")
st.markdown("[ENGAGE Program](https://engange.uonbi.ac.ke)")

# Define the Streamlit app
st.title('Cholera Outbreak Risk Prediction App')

st.write("""
This application predicts the risk of cholera occurrence based on various predictors.
Please provide the input data to get a prediction.
""")

# Define categorical to numeric mappings
subcounty_map = {'South': 0, 'East': 1, 'West': 2, 'North': 3, 'North East': 4}
water_map = {'piped': 0, 'borehole': 1, 'public well': 2, 'other': 3}
sanitation_map = {'private toilet': 0, 'private latrine': 1, 'public latrine': 2, 'other': 3}
income_map = {'lowest': 0, 'low': 1, 'average': 2, 'high': 3, 'highest': 4}
informal_settlement_map = {'yes': 1, 'no': 0}

# Create input fields for the user to enter data
sex = st.selectbox("Sex (0 for Female, 1 for Male)", [0, 1])
age = st.slider("Age", min_value=0, max_value=100, value=0)
subcounty = st.selectbox("Subcounty", list(subcounty_map.keys()))
water = st.selectbox("Water Source", list(water_map.keys()))
sanitation = st.selectbox("Sanitation", list(sanitation_map.keys()))
income = st.selectbox("Income", list(income_map.keys()))
informal_settlement = st.selectbox("Informal Settlement", list(informal_settlement_map.keys()))

# Convert categorical inputs to numeric using the mappings
subcounty_numeric = subcounty_map[subcounty]
water_numeric = water_map[water]
sanitation_numeric = sanitation_map[sanitation]
income_numeric = income_map[income]
informal_settlement_numeric = informal_settlement_map[informal_settlement]

# Prepare input data for prediction
input_data = np.array([sex, age, subcounty_numeric, water_numeric, sanitation_numeric, income_numeric, informal_settlement_numeric]).reshape(1, -1)

# Make prediction and display results
if st.button('Predict'):
    prediction = logreg.predict(input_data)
    if prediction[0] == 1:
        st.write("Prediction: Positive")
    else:
        st.write("Prediction: Negative")
