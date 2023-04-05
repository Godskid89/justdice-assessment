import streamlit as st
import pandas as pd
import numpy as np
from model import train_model
from data_preprocessing import preprocess_data

# File location
datafile = 'data/rev_prediction.csv'
# Train model
model = train_model(datafile)

data = pd.read_csv(datafile)

def predict_revenue(network_id, country_id, month_year, total_adspend):

    # Prepare input data
    input_df = pd.DataFrame({
        'network_id': [network_id],
        'country_id': [country_id],
        'month_year': [month_year],
        'total_adspend': [total_adspend]
    })
    input_df = preprocess_data(input_df)
    
    # Make prediction
    prediction = model.predict(input_df)
    return prediction[0]


# Streamlit app
st.title('Total Revenue Predictor')

# Input form
network_id = st.selectbox('Network ID', data['network_id'].unique())
country_id = st.selectbox('Country ID', data['country_id'].unique())
month = st.selectbox('Month', data['month_year'].unique())
total_adspend = st.number_input('Total Ad Spend', value=1000, step=100)

# Predict revenue
prediction = predict_revenue(network_id, country_id, month, total_adspend)

# Display prediction
st.write('Estimated Total Revenue:', round(prediction, 2))
