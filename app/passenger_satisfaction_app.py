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

    # User Inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    customer_type = st.selectbox("Customer Type", ["Loyal", "New"])
    age = st.number_input("Age", min_value=5, max_value=100, step=1)
    travel_type = st.selectbox("Type of Travel", ["Business", "Personal"])
    flight_class = st.selectbox("Class", ["Economy", "Business", "First"])
    flight_distance = st.number_input("Flight Distance (km)", min_value=100, max_value=10000, step=50)

    inflight_wifi = st.slider("Inflight Wifi Service (1-5)", 1, 5, 3)
    dep_arr_convenience = st.slider("Departure/Arrival Convenience (1-5)", 1, 5, 3)
    ease_of_booking = st.slider("Ease of Online Booking (1-5)", 1, 5, 3)
    gate_location = st.slider("Gate Location (1-5)", 1, 5, 3)
    food_drink = st.slider("Food and Drink (1-5)", 1, 5, 3)
    online_boarding = st.slider("Online Boarding (1-5)", 1, 5, 3)
    seat_comfort = st.slider("Seat Comfort (1-5)", 1, 5, 3)
    inflight_entertainment = st.slider("Inflight Entertainment (1-5)", 1, 5, 3)
    onboard_service = st.slider("On-board Service (1-5)", 1, 5, 3)
    legroom_service = st.slider("Leg Room Service (1-5)", 1, 5, 3)
    baggage_handling = st.slider("Baggage Handling (1-5)", 1, 5, 3)
    checkin_service = st.slider("Check-in Service (1-5)", 1, 5, 3)
    inflight_service = st.slider("Inflight Service (1-5)", 1, 5, 3)
    cleanliness = st.slider("Cleanliness (1-5)", 1, 5, 3)

    dep_delay = st.number_input("Departure Delay (Minutes)", min_value=0, max_value=180, step=1)
    arr_delay = st.number_input("Arrival Delay (Minutes)", min_value=0, max_value=180, step=1)

    # **Missing Feature (Added here to reach 24 features)**
    loyalty_points = st.number_input("Loyalty Points Earned", min_value=0, max_value=10000, step=100)  # New Feature

    satisfaction_score = st.slider("Satisfaction Score (1-5)", 1, 5, 3)  # Previously added missing feature

    # Encode categorical inputs
    gender_encoded = 0 if gender == "Male" else 1
    customer_encoded = 0 if customer_type == "Loyal" else 1
    travel_encoded = 0 if travel_type == "Business" else 1
    class_encoded = ["Economy", "Business", "First"].index(flight_class)

    # Predict button
    if st.button("Predict Satisfaction"):
        features = [
            gender_encoded, customer_encoded, age, travel_encoded, class_encoded, flight_distance,
            inflight_wifi, dep_arr_convenience, ease_of_booking, gate_location, food_drink,
            online_boarding, seat_comfort, inflight_entertainment, onboard_service,
            legroom_service, baggage_handling, checkin_service, inflight_service,
            cleanliness, dep_delay, arr_delay, loyalty_points, satisfaction_score  # **Now 24 features**
        ]

        # Ensure correct feature count
        if len(features) != 24:
            st.error(f"Feature mismatch: Expected 24, got {len(features)}")
        else:
            prediction = predict_satisfaction(features)
            satisfaction = "Satisfied 😊" if prediction == 1 else "Not Satisfied 😞"
            st.success(f"🛫 Passenger is: {satisfaction}")
