import streamlit as st
from flight_price_app import flight_price_app
from passenger_satisfaction_app import passenger_satisfaction_app

# Streamlit page configuration
#st.set_page_config(page_title="Flight & Passenger Prediction", layout="wide")

# Sidebar navigation
st.sidebar.title("🔍 Select a Prediction Criteria")
page = st.sidebar.radio("Go to:", ["✈️ Flight Price Predictions", "😊 Passenger Satisfaction Predictions"])

# Load the appropriate page based on selection
if page == "✈️ Flight Price Predictions":
    flight_price_app()
elif page == "😊 Passenger Satisfaction Predictions":
    passenger_satisfaction_app()
