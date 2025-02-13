import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, r2_score
from xgboost import XGBRegressor

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure necessary directories exist
os.makedirs("models", exist_ok=True)

def load_dataset(file_path, target_col):
    """Load dataset and check for errors."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"The dataset at {file_path} is empty!")
    
    logging.info(f"✅ Dataset loaded successfully from {file_path}")
    df.columns = df.columns.str.strip()
    
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in the dataset.")
    
    return df

def preprocess_data(df, target_col):
    """Preprocess dataset: Encode categorical features and scale numerical values."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_flight_price_model(X_train, X_test, y_train, y_test):
    """Train and save flight price prediction model."""
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, tree_method="hist", random_state=42)
    model.fit(X_train, y_train)
    
    model_filename = "models/flight_price_model.pkl"
    joblib.dump(model, model_filename)
    logging.info(f"✅ Flight Price Model saved at {model_filename}")
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"✈️ Flight Price Model R² Score: {r2:.2f}")

def train_satisfaction_model(X_train, X_test, y_train, y_test):
    """Train and save customer satisfaction prediction model."""
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    model_filename = "models/passenger_satisfaction_model.pkl"
    joblib.dump(model, model_filename)
    logging.info(f"✅ Customer Satisfaction Model saved at {model_filename}")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"😊 Customer Satisfaction Model Accuracy: {accuracy:.2f}")
    logging.info("\n" + classification_report(y_test, y_pred))

# Flight Price Prediction
flight_data_path = "data/Cleaned_Processed_Flight_Price.csv"
df_flight = load_dataset(flight_data_path, "Price")
X_train_flight, X_test_flight, y_train_flight, y_test_flight = preprocess_data(df_flight, "Price")
train_flight_price_model(X_train_flight, X_test_flight, y_train_flight, y_test_flight)

# Customer Satisfaction Prediction
satisfaction_data_path = "data/Cleaned_Processed_Passenger_Satisfaction.csv"
df_satisfaction = load_dataset(satisfaction_data_path, "satisfaction")
X_train_satisfaction, X_test_satisfaction, y_train_satisfaction, y_test_satisfaction = preprocess_data(df_satisfaction, "satisfaction")
train_satisfaction_model(X_train_satisfaction, X_test_satisfaction, y_train_satisfaction, y_test_satisfaction)