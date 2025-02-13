import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report

# Ensure model directory exists
model_dir = "models/"
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory '{model_dir}' does not exist. Train models first.")

# ---------------- FLIGHT PRICE PREDICTION EVALUATION ---------------- #
flight_data_path = "data/Cleaned_Processed_Flight_Price.csv"
flight_model_path = os.path.join(model_dir, "flight_price_model.pkl")

if not os.path.exists(flight_data_path):
    raise FileNotFoundError(f"Flight price dataset not found at {flight_data_path}")
if not os.path.exists(flight_model_path):
    raise FileNotFoundError(f"Flight price model not found at {flight_model_path}")

# Load dataset
df_flight = pd.read_csv(flight_data_path)
df_flight.columns = df_flight.columns.str.strip()  # Ensure clean column names

# Ensure dataset is not empty
if df_flight.empty:
    raise ValueError("The flight price dataset is empty!")

# Identify categorical columns and target column
target_col_flight = "Price"
if target_col_flight not in df_flight.columns:
    raise KeyError(f"Target column '{target_col_flight}' not found in the dataset.")

X_flight = df_flight.drop(columns=[target_col_flight])
y_flight = df_flight[target_col_flight]

# Encode categorical features
categorical_cols = X_flight.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    X_flight[col] = X_flight[col].astype('category').cat.codes

# Load trained model
flight_model = joblib.load(flight_model_path)

# Predictions
y_pred_flight = flight_model.predict(X_flight)

# Evaluation metrics
mae_flight = mean_absolute_error(y_flight, y_pred_flight)
mse_flight = mean_squared_error(y_flight, y_pred_flight)
r2_flight = r2_score(y_flight, y_pred_flight)

print("\n🚀 Flight Price Prediction Model Evaluation:")
print(f"📌 Mean Absolute Error (MAE): {mae_flight:.2f}")
print(f"📌 Mean Squared Error (MSE): {mse_flight:.2f}")
print(f"📌 R² Score: {r2_flight:.4f}")

# ---------------- CUSTOMER SATISFACTION PREDICTION EVALUATION ---------------- #

satisfaction_data_path = "data/Cleaned_Processed_Passenger_Satisfaction.csv"
satisfaction_model_path = os.path.join(model_dir, "passenger_satisfaction_model.pkl")

if not os.path.exists(satisfaction_data_path):
    raise FileNotFoundError(f"Passenger satisfaction dataset not found at {satisfaction_data_path}")
if not os.path.exists(satisfaction_model_path):
    raise FileNotFoundError(f"Passenger satisfaction model not found at {satisfaction_model_path}")

# Load dataset
df_satisfaction = pd.read_csv(satisfaction_data_path)
df_satisfaction.columns = df_satisfaction.columns.str.strip()  # Ensure clean column names

# Ensure dataset is not empty
if df_satisfaction.empty:
    raise ValueError("The customer satisfaction dataset is empty!")

# Identify categorical columns and target column
target_col_satisfaction = "satisfaction"
if target_col_satisfaction not in df_satisfaction.columns:
    raise KeyError(f"Target column '{target_col_satisfaction}' not found in the dataset.")

X_satisfaction = df_satisfaction.drop(columns=[target_col_satisfaction])
y_satisfaction = df_satisfaction[target_col_satisfaction]

# Encode categorical features
categorical_cols_satisfaction = X_satisfaction.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols_satisfaction:
    X_satisfaction[col] = X_satisfaction[col].astype('category').cat.codes

# Load trained model
satisfaction_model = joblib.load(satisfaction_model_path)

# Predictions
y_pred_satisfaction = satisfaction_model.predict(X_satisfaction)

# Evaluation metrics
accuracy = accuracy_score(y_satisfaction, y_pred_satisfaction)
classification_rep = classification_report(y_satisfaction, y_pred_satisfaction)

print("\n🚀 Customer Satisfaction Prediction Model Evaluation:")
print(f"📌 Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)
