import streamlit as st
import numpy as np
import joblib  # Load the trained model

# Load the trained passenger satisfaction model
model = joblib.load("models/best_passenger_model.pkl")

# Prediction function
def predict_satisfaction(features):
    features = np.array(features).reshape(1, -1)
    return model.predict(features)[0]

def passenger_satisfaction_app():
    st.title("😊 Passenger Satisfaction Prediction")

    st.sidebar.subheader("Passenger Details")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    customer_type = st.sidebar.selectbox("Customer Type", ["Loyal", "New"])
    age = st.sidebar.number_input("Age", min_value=5, max_value=100, step=1)
    travel_type = st.sidebar.selectbox("Type of Travel", ["Business", "Personal"])
    flight_class = st.sidebar.selectbox("Class", ["Economy", "Business", "First"])
    flight_distance = st.sidebar.number_input("Flight Distance (km)", min_value=100, max_value=10000, step=50)

    inflight_wifi = st.sidebar.slider("Inflight Wifi Service (1-5)", 1, 5, 3)
    dep_arr_convenience = st.sidebar.slider("Departure/Arrival Convenience (1-5)", 1, 5, 3)
    ease_of_booking = st.sidebar.slider("Ease of Online Booking (1-5)", 1, 5, 3)
    gate_location = st.sidebar.slider("Gate Location (1-5)", 1, 5, 3)
    food_drink = st.sidebar.slider("Food and Drink (1-5)", 1, 5, 3)
    online_boarding = st.sidebar.slider("Online Boarding (1-5)", 1, 5, 3)
    seat_comfort = st.sidebar.slider("Seat Comfort (1-5)", 1, 5, 3)
    inflight_entertainment = st.sidebar.slider("Inflight Entertainment (1-5)", 1, 5, 3)
    onboard_service = st.sidebar.slider("On-board Service (1-5)", 1, 5, 3)
    legroom_service = st.sidebar.slider("Leg Room Service (1-5)", 1, 5, 3)
    baggage_handling = st.sidebar.slider("Baggage Handling (1-5)", 1, 5, 3)
    checkin_service = st.sidebar.slider("Check-in Service (1-5)", 1, 5, 3)
    inflight_service = st.sidebar.slider("Inflight Service (1-5)", 1, 5, 3)
    cleanliness = st.sidebar.slider("Cleanliness (1-5)", 1, 5, 3)

    dep_delay = st.sidebar.number_input("Departure Delay (Minutes)", min_value=0, max_value=180, step=1)
    arr_delay = st.sidebar.number_input("Arrival Delay (Minutes)", min_value=0, max_value=180, step=1)

    # Encode categorical inputs
    gender_encoded = 0 if gender == "Male" else 1
    customer_encoded = 0 if customer_type == "Loyal" else 1
    travel_encoded = 0 if travel_type == "Business" else 1
    class_encoded = ["Economy", "Business", "First"].index(flight_class)

    # Predict button
    if st.sidebar.button("Predict Satisfaction"):
        features = [
            gender_encoded, customer_encoded, age, travel_encoded, class_encoded, flight_distance,
            inflight_wifi, dep_arr_convenience, ease_of_booking, gate_location, food_drink,
            online_boarding, seat_comfort, inflight_entertainment, onboard_service,
            legroom_service, baggage_handling, checkin_service, inflight_service,
            cleanliness, dep_delay, arr_delay
        ]

        # Ensure correct feature count
        if len(features) != 22:
            st.error(f"Feature mismatch: Expected 22, got {len(features)}")
        else:
            prediction = predict_satisfaction(features)
            satisfaction = "Satisfied 😊" if prediction == 1 else "Not Satisfied 😞"
            st.success(f"🛫 Passenger is: {satisfaction}")
