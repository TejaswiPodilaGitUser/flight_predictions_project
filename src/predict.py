import pandas as pd
import joblib
import logging
import numpy as np
import os

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Model Paths
model_dir = "models"
model_path = os.path.join(model_dir, "flight_price_model.pkl")
scaler_path = os.path.join(model_dir, "flight_price_scaler.pkl")
feature_order_path = os.path.join(model_dir, "feature_order.pkl")

# Check if all required model files exist
missing_files = [path for path in [model_path, scaler_path, feature_order_path] if not os.path.exists(path)]
if missing_files:
    logging.error(f"❌ Missing required files: {missing_files}. Ensure all model files exist in the 'models' directory.")
    raise FileNotFoundError(f"❌ Missing required files: {missing_files}. Ensure all model files exist.")

# Load Models and Feature Order
flight_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
expected_features = joblib.load(feature_order_path)
logging.info("✅ Models loaded successfully.")

def preprocess_input(sample_data):
    """
    Ensures input data matches the trained model's feature order, applies categorical encoding, and scales numeric data.
    """
    df = pd.DataFrame([sample_data])

    # **Ensure feature names match exactly (case-sensitive)**
    df = df.rename(columns=lambda x: x.strip())  # Remove spaces

    # **Ensure all expected features are present**
    missing_cols = set(expected_features) - set(df.columns)
    if missing_cols:
        logging.error(f"❌ Missing features: {missing_cols}")
        raise ValueError(f"Missing required features: {missing_cols}")

    extra_cols = set(df.columns) - set(expected_features)
    if extra_cols:
        logging.warning(f"⚠️ Unexpected extra columns: {extra_cols}. They will be ignored.")

    # **Reorder columns to match training order**
    df = df[expected_features]

    # **Ensure categorical encoding is consistent**
    categorical_cols = ["Airline", "Source", "Destination", "Additional_Info", "Route"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.as_ordered()

    # **Apply Scaling**
    df_scaled = scaler.transform(df)

    # Debugging logs
    logging.info(f"🛠️ Final Feature Order Before Scaling: {df.columns.tolist()}")
    logging.info(f"🔍 Model Expected Order: {expected_features}")

    return df_scaled

def predict_flight_price(sample_data):
    """
    Predicts flight price given a sample data dictionary.
    """
    try:
        processed_data = preprocess_input(sample_data)
        prediction = flight_model.predict(processed_data)[0]
        logging.info(f"✈️ Predicted Flight Price: ₹{prediction:.2f}")
        return prediction
    except Exception as e:
        logging.error(f"❌ Prediction failed: {e}")
        return None

# Example Test Case
sample_flight_data = {
    "Airline": "IndiGo", "Source": "Delhi", "Destination": "Mumbai",
    "Dep_Hour": 10, "Dep_Minute": 30, "Arr_Hour": 13, "Arr_Minute": 15,
    "Duration_Minutes": 165, "Total_Stops": 1, "Additional_Info": "No Info",
    "Journey_Day": 14, "Journey_Month": 2, "Route": "DEL → BOM"
}

if __name__ == "__main__":
    price = predict_flight_price(sample_flight_data)
    if price is None:
        logging.error("❌ Prediction could not be completed.")
