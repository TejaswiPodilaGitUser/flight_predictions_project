import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load Flight Price Data
flight_data_path = "data/Cleaned_Processed_Flight_Price.csv"

if not os.path.exists(flight_data_path):
    raise FileNotFoundError(f"Dataset not found at {flight_data_path}")

df_flight = pd.read_csv(flight_data_path)
if df_flight.empty:
    raise ValueError("The flight price dataset is empty!")

logging.info("✅ Flight dataset loaded successfully.")

# Clean column names
df_flight.columns = df_flight.columns.str.strip()

# Identify categorical columns
categorical_cols = df_flight.select_dtypes(include=['object']).columns.tolist()

# Target column check
target_col_flight = "Price"
if target_col_flight not in df_flight.columns:
    raise KeyError(f"Target column '{target_col_flight}' not found in the dataset.")

# Prepare features and target
X_flight = df_flight.drop(columns=[target_col_flight])
y_flight = df_flight[target_col_flight]

# Encode categorical features
for col in categorical_cols:
    encoder = LabelEncoder()
    X_flight[col] = encoder.fit_transform(X_flight[col])

# Train-test split
X_train_flight, X_test_flight, y_train_flight, y_test_flight = train_test_split(
    X_flight, y_flight, test_size=0.2, random_state=42
)

# Feature scaling
scaler_flight = StandardScaler()
X_train_flight_scaled = scaler_flight.fit_transform(X_train_flight)
X_test_flight_scaled = scaler_flight.transform(X_test_flight)

# ✅ Save the scaler for future predictions
scaler_filename = "models/flight_price_scaler.pkl"
joblib.dump(scaler_flight, scaler_filename)
logging.info(f"✅ Flight Price Scaler saved at {scaler_filename}")

# Train flight price prediction model
flight_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, tree_method="hist", random_state=42)
flight_model.fit(X_train_flight_scaled, y_train_flight)

# Save flight price model
flight_model_filename = "models/flight_price_model.pkl"
joblib.dump(flight_model, flight_model_filename)
logging.info(f"✅ Flight Price Model saved at {flight_model_filename}")

# Model Evaluation
y_pred_flight = flight_model.predict(X_test_flight_scaled)
r2 = r2_score(y_test_flight, y_pred_flight)
logging.info(f"✈️ Flight Price Model R² Score: {r2:.2f}")
