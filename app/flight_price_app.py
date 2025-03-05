import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plots_flight  # ✅ Import the new plots.py file

# Load the trained flight price model
model = joblib.load("models/best_flight_price_model.pkl")

# Load dataset dynamically
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned/Cleaned_Flight_Price.csv")
    return df

df = load_data()

# Extract unique values dynamically
unique_airlines = df["Airline"].unique().tolist()
unique_sources = df["Source"].unique().tolist()
unique_destinations = df["Destination"].unique().tolist()
unique_stops = df["Total_Stops"].unique().tolist()
unique_additional_info = df["Additional_Info"].unique().tolist()

# Prediction function
def predict_price(features):
    features = np.array(features).reshape(1, -1)
    return model.predict(features)[0]

# ✅ Flight Price App
def flight_price_app():
    st.markdown("<h3 style='text-align: center;'>✈️ Flight Price Prediction</h3>", unsafe_allow_html=True)

    st.sidebar.subheader("Flight Details")

    # ✅ User Input
    airline = st.sidebar.selectbox("Airline", unique_airlines)
    source = st.sidebar.selectbox("Source", unique_sources)
    destination = st.sidebar.selectbox("Destination", unique_destinations)
    stops = st.sidebar.selectbox("Total Stops", unique_stops)
    additional_info = st.sidebar.selectbox("Additional Info", unique_additional_info)

    # Other Inputs
    route = st.sidebar.text_input("Route (e.g., 'BLR → DEL')")
    journey_day = st.sidebar.slider("Journey Day", 1, 31, 15)
    journey_month = st.sidebar.slider("Journey Month", 1, 12, 6)
    dep_hour = st.sidebar.slider("Departure Hour", 0, 23, 10)
    dep_minute = st.sidebar.slider("Departure Minute", 0, 59, 30)
    arr_hour = st.sidebar.slider("Arrival Hour", 0, 23, 12)
    arr_minute = st.sidebar.slider("Arrival Minute", 0, 59, 45)
    duration_minutes = st.sidebar.number_input("Duration (minutes)", min_value=30, max_value=1440, step=10)

    # ✅ Encode categorical inputs dynamically
    airline_encoded = unique_airlines.index(airline)
    source_encoded = unique_sources.index(source)
    destination_encoded = unique_destinations.index(destination)
    stops_encoded = unique_stops.index(stops)
    additional_info_encoded = unique_additional_info.index(additional_info)
    
    # Encode Route (basic handling: count number of stops in the route)
    route_encoded = len(route.split("→")) - 1 if route else 0  

    # ✅ Predict button
    if st.sidebar.button("Predict Price"):
        features = [
            airline_encoded, source_encoded, destination_encoded, route_encoded, stops_encoded, 
            additional_info_encoded, journey_day, journey_month, dep_hour, dep_minute, 
            arr_hour, arr_minute, duration_minutes
        ]
        prediction = predict_price(features)
        st.subheader(f"💰 Predicted Flight Price: ₹{round(prediction, 2)}")

    # ✅ Display All Plots Without Selection
    st.markdown("### 📊 Flight Price Insights")

    col1, col2 = st.columns(2)
    with col1:
        plots_flight.plot_airline_vs_price(df)
    with col2:
        plots_flight.plot_source_vs_destination(df)

    col3, col4 = st.columns(2)
    with col3:
        plots_flight.plot_stops_vs_price(df)
    with col4:
        plots_flight.plot_dep_hour_vs_price(df)

    st.markdown("## ⌛ Flight Duration vs Price")
    plots_flight.plot_duration_vs_price(df)

if __name__ == "__main__":
    flight_price_app()
