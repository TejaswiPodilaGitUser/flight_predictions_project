import streamlit as st
import numpy as np
import joblib  # Load the trained model

# Load the trained flight price model
model = joblib.load("models/best_flight_price_model.pkl")

# Prediction function
def predict_price(features):
    features = np.array(features).reshape(1, -1)
    return model.predict(features)[0]

def flight_price_app():
    st.title("✈️ Flight Price Prediction")

    st.sidebar.subheader("Flight Details")

    # Move inputs to sidebar
    airline = st.sidebar.selectbox("Airline", ["Air India", "IndiGo", "SpiceJet", "Vistara"])
    source = st.sidebar.selectbox("Source", ["Delhi", "Mumbai", "Bangalore", "Kolkata"])
    destination = st.sidebar.selectbox("Destination", ["Hyderabad", "Chennai", "Goa", "Pune"])
    route = st.sidebar.text_input("Route (e.g., 'BLR → DEL')")
    stops = st.sidebar.selectbox("Total Stops", ["Non-stop", "1 Stop", "2 Stops", "3+ Stops"])
    additional_info = st.sidebar.selectbox("Additional Info", ["No info", "In-flight meal", "Extra baggage", "Other"])
    journey_day = st.sidebar.slider("Journey Day", 1, 31, 15)
    journey_month = st.sidebar.slider("Journey Month", 1, 12, 6)
    dep_hour = st.sidebar.slider("Departure Hour", 0, 23, 10)
    dep_minute = st.sidebar.slider("Departure Minute", 0, 59, 30)
    arr_hour = st.sidebar.slider("Arrival Hour", 0, 23, 12)
    arr_minute = st.sidebar.slider("Arrival Minute", 0, 59, 45)
    duration_minutes = st.sidebar.number_input("Duration (minutes)", min_value=30, max_value=1440, step=10)

    # Encode categorical inputs
    airline_encoded = ["Air India", "IndiGo", "SpiceJet", "Vistara"].index(airline)
    source_encoded = ["Delhi", "Mumbai", "Bangalore", "Kolkata"].index(source)
    destination_encoded = ["Hyderabad", "Chennai", "Goa", "Pune"].index(destination)
    stops_encoded = ["Non-stop", "1 Stop", "2 Stops", "3+ Stops"].index(stops)
    additional_info_encoded = ["No info", "In-flight meal", "Extra baggage", "Other"].index(additional_info)
    
    # Encode Route (basic handling: count number of stops in the route)
    route_encoded = len(route.split("→")) - 1 if route else 0  # Convert route string into number of stops

    # Predict button in sidebar
    if st.sidebar.button("Predict Price"):
        features = [
            airline_encoded, source_encoded, destination_encoded, route_encoded, stops_encoded, 
            additional_info_encoded, journey_day, journey_month, dep_hour, dep_minute, 
            arr_hour, arr_minute, duration_minutes
        ]

        # Ensure correct feature count
        prediction = predict_price(features)
        st.subheader(f"💰 Predicted Flight Price: ₹{round(prediction, 2)}")
