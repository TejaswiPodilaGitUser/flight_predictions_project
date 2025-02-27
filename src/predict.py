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
expected_features = joblib.load(feature_order_path)  # Ensure this is a list of feature names
logging.info("✅ Models and feature order loaded successfully.")

def preprocess_input(sample_data):
    """
    Ensures input data matches the trained model's feature order, applies categorical encoding, and scales numeric data.
    """
    logging.info(f"✈️ Received input data for prediction: {sample_data}")

    df = pd.DataFrame([sample_data])

    # **Ensure feature names match exactly (case-sensitive, trimmed)**
    df.columns = df.columns.str.strip()
    expected_features_cleaned = [col.strip() for col in expected_features]

    # **Step 1: Validate Features**
    missing_cols = set(expected_features_cleaned) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_features_cleaned)

    if missing_cols:
        logging.error(f"❌ Missing required features: {missing_cols}")
        raise ValueError(f"❌ Missing required features: {missing_cols}")

    if extra_cols:
        logging.warning(f"⚠️ Extra features detected (will be ignored): {extra_cols}")

    # **Step 2: Reorder columns strictly to match training order**
    df = df.reindex(columns=expected_features_cleaned)  # Ensures exact order

    logging.info(f"🛠️ Final Feature Order Before Encoding: {df.columns.tolist()}")

    # **Step 3: Handle categorical variables**
    categorical_cols = ["Airline", "Source", "Destination", "Additional_Info", "Route"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes  
            logging.info(f"🛠️ Converted '{col}' to numeric codes.")

    logging.info(f"🛠️ Feature Order After Encoding: {df.columns.tolist()}")

    # **Final Check Before Scaling**
    if df.isnull().any().any():
        logging.error("❌ Reindexing introduced NaN values. Check for mismatched feature names.")
        raise ValueError("❌ Feature order mismatch or missing values detected.")

    # **Step 4: Apply Scaling**
    try:
        df_scaled = scaler.transform(df)
        logging.info(f"🛠️ Successfully applied scaling.")
    except ValueError as e:
        logging.error(f"❌ Feature mismatch error: {e}")
        logging.error(f"🔄 Expected Features: {list(scaler.feature_names_in_)}")
        logging.error(f"🔄 Provided Features: {df.columns.tolist()}")
        raise ValueError(f"❌ Issue with scaling: Feature names must exactly match those used in training.")

    return df_scaled

def predict_flight_price(sample_data):
    """
    Predicts flight price given a sample data dictionary.
    """
    try:
        processed_data = preprocess_input(sample_data)
        logging.info(f"🛠️ Processed Data (Before Prediction): {processed_data}")

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
