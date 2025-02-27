import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

# Ensure model directory exists
model_dir = "models/"
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory '{model_dir}' does not exist. Train models first.")

# ---------------- FLIGHT PRICE PREDICTION EVALUATION ---------------- #
flight_data_path = "data/Cleaned_Processed_Flight_Price.csv"
flight_model_path = os.path.join(model_dir, "flight_price_model.pkl")
feature_names_path = os.path.join(model_dir, "flight_price_features.pkl")

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

# Encode categorical features using One-Hot Encoding
X_flight = pd.get_dummies(X_flight, drop_first=True)

# Train-test split
X_flight_train, X_flight_test, y_flight_train, y_flight_test = train_test_split(X_flight, y_flight, test_size=0.2, random_state=42)

# Load trained model
flight_model = joblib.load(flight_model_path)

# Ensure feature consistency
expected_features = joblib.load(feature_names_path)
X_flight_test = X_flight_test.reindex(columns=expected_features, fill_value=0)

# Predictions
y_pred_flight = flight_model.predict(X_flight_test)

# Evaluation metrics
mae_flight = mean_absolute_error(y_flight_test, y_pred_flight)
mse_flight = mean_squared_error(y_flight_test, y_pred_flight)
r2_flight = r2_score(y_flight_test, y_pred_flight)

print("\n🚀 Flight Price Prediction Model Evaluation:")
print(f"📌 Mean Absolute Error (MAE): {mae_flight:.2f}")
print(f"📌 Mean Squared Error (MSE): {mse_flight:.2f}")
print(f"📌 R² Score: {r2_flight:.4f}")

# ---------------- CUSTOMER SATISFACTION PREDICTION EVALUATION ---------------- #

satisfaction_data_path = "data/Cleaned_Processed_Passenger_Satisfaction.csv"
satisfaction_model_path = os.path.join(model_dir, "passenger_satisfaction_model.pkl")
feature_names_satisfaction_path = os.path.join(model_dir, "passenger_satisfaction_features.pkl")

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
X_satisfaction = pd.get_dummies(X_satisfaction, drop_first=True)

# Train-test split
X_satisfaction_train, X_satisfaction_test, y_satisfaction_train, y_satisfaction_test = train_test_split(X_satisfaction, y_satisfaction, test_size=0.2, random_state=42)

# Load trained model
satisfaction_model = joblib.load(satisfaction_model_path)

# Ensure feature consistency
expected_features_satisfaction = joblib.load(feature_names_satisfaction_path)
X_satisfaction_test = X_satisfaction_test.reindex(columns=expected_features_satisfaction, fill_value=0)

# Predictions
y_pred_satisfaction = satisfaction_model.predict(X_satisfaction_test)

# Evaluation metrics
accuracy = accuracy_score(y_satisfaction_test, y_pred_satisfaction)
classification_rep = classification_report(y_satisfaction_test, y_pred_satisfaction)
conf_matrix = confusion_matrix(y_satisfaction_test, y_pred_satisfaction)

print("\n🚀 Customer Satisfaction Prediction Model Evaluation:")
print(f"📌 Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Dissatisfied", "Satisfied"], yticklabels=["Dissatisfied", "Satisfied"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Passenger Satisfaction")
plt.show()
