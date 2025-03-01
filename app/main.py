import streamlit as st
from flight_price_app import flight_price_app
from passenger_satisfaction_app import passenger_satisfaction_app

# Streamlit page configuration
st.set_page_config(page_title="Flight & Passenger Prediction", layout="wide")

# Sidebar navigation
st.sidebar.title("🔍 Select a Prediction Task")
page = st.sidebar.radio("Go to:", ["✈️ Flight Price Prediction", "😊 Passenger Satisfaction Prediction"])

# Load the appropriate page based on selection
if page == "✈️ Flight Price Prediction":
    flight_price_app()
elif page == "😊 Passenger Satisfaction Prediction":
    passenger_satisfaction_app()
