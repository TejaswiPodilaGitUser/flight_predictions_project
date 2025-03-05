import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plots_passenger  # ✅ Import the new plots_passenger.py file

# ✅ Load the trained model
model = joblib.load("models/best_passenger_model.pkl")

# ✅ Load dataset dynamically
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned/Cleaned_Passenger_Satisfaction.csv")
    return df

df = load_data()

# ✅ Extract unique values dynamically
unique_genders = df["Gender"].unique().tolist()
unique_customer_types = df["Customer Type"].unique().tolist()
unique_travel_types = df["Type of Travel"].unique().tolist()
unique_flight_classes = df["Class"].unique().tolist()

# ✅ Prediction function
def predict_satisfaction(features):
    features = np.array(features).reshape(1, -1)
    return model.predict(features)[0]

# ✅ Streamlit App
def passenger_satisfaction_app():
    #st.title("😊 Passenger Satisfaction Prediction")
    st.markdown("<h3 style='text-align: center;'>😊 Passenger Satisfaction Prediction</h3>", unsafe_allow_html=True)


    st.sidebar.subheader("Passenger Details")

    # ✅ User Input
    gender = st.sidebar.selectbox("Gender", unique_genders)
    customer_type = st.sidebar.selectbox("Customer Type", unique_customer_types)
    age = st.sidebar.number_input("Age", min_value=5, max_value=100, step=1, value=25)
    travel_type = st.sidebar.selectbox("Type of Travel", unique_travel_types)
    flight_class = st.sidebar.selectbox("Class", unique_flight_classes)
    flight_distance = st.sidebar.number_input("Flight Distance (km)", min_value=100, max_value=10000, step=50)

    # ✅ Sliders for ratings
    inflight_wifi = st.sidebar.slider("Inflight Wifi Service (1-5)", 1, 5, 4)
    dep_arr_convenience = st.sidebar.slider("Departure/Arrival Convenience (1-5)", 1, 5, 4)
    ease_of_booking = st.sidebar.slider("Ease of Online Booking (1-5)", 1, 5, 4)
    gate_location = st.sidebar.slider("Gate Location (1-5)", 1, 5, 4)
    food_drink = st.sidebar.slider("Food and Drink (1-5)", 1, 5, 3)
    online_boarding = st.sidebar.slider("Online Boarding (1-5)", 1, 5, 4)
    seat_comfort = st.sidebar.slider("Seat Comfort (1-5)", 1, 5, 3)
    inflight_entertainment = st.sidebar.slider("Inflight Entertainment (1-5)", 1, 5, 3)
    onboard_service = st.sidebar.slider("On-board Service (1-5)", 1, 5, 4)
    legroom_service = st.sidebar.slider("Leg Room Service (1-5)", 1, 5, 3)
    baggage_handling = st.sidebar.slider("Baggage Handling (1-5)", 1, 5, 3)
    checkin_service = st.sidebar.slider("Check-in Service (1-5)", 1, 5, 4)
    inflight_service = st.sidebar.slider("Inflight Service (1-5)", 1, 5, 4)
    cleanliness = st.sidebar.slider("Cleanliness (1-5)", 1, 5, 3)

    dep_delay = st.sidebar.number_input("Departure Delay (Minutes)", min_value=0, max_value=180, step=1)
    arr_delay = st.sidebar.number_input("Arrival Delay (Minutes)", min_value=0, max_value=180, step=1)

    # ✅ Encode categorical inputs dynamically
    gender_encoded = unique_genders.index(gender)
    customer_encoded = unique_customer_types.index(customer_type)
    travel_encoded = unique_travel_types.index(travel_type)
    class_encoded = unique_flight_classes.index(flight_class)

    # ✅ Predict button
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

    # ✅ Display all plots in a 2x2 grid
    st.subheader("📊 Data Insights")

    col1, col2 = st.columns(2)  # Create two columns

    with col1:
        st.subheader("📌 Age Distribution")
        plots_passenger.plot_age_distribution(df)

    with col2:
        st.subheader("📌 Flight Distance vs Satisfaction")
        plots_passenger.plot_flight_distance_vs_satisfaction(df)

    col3, col4 = st.columns(2)  # Second row of plots

    with col3:
        st.subheader("📌 Departure Delay vs Satisfaction")
        plots_passenger.plot_delay_vs_satisfaction(df)

    with col4:
        st.subheader("📌 Class-wise Satisfaction")
        plots_passenger.plot_class_vs_satisfaction(df)

    # ✅ Feature Importance in a full row
    st.subheader("📌 Feature Importance")
    plots_passenger.plot_feature_importance("models/tuned_passenger_satisfaction_model.pkl",df)

if __name__ == "__main__":
    passenger_satisfaction_app()