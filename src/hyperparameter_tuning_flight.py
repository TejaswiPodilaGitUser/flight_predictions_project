import pandas as pd
import numpy as np
import joblib
import os
import logging
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load Dataset
flight_data_path = "data/processed/Cleaned_Processed_Flight_Price.csv"
if not os.path.exists(flight_data_path):
    raise FileNotFoundError(f"Dataset not found at {flight_data_path}")

df = pd.read_csv(flight_data_path)
if df.empty:
    raise ValueError("The dataset is empty!")

logging.info("✅ Flight dataset loaded successfully.")

# Clean column names
df.columns = df.columns.str.strip()

# Target column check
target_col = "Price"
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found in the dataset.")

# Prepare features and target
X = df.drop(columns=[target_col])
y = df[target_col].astype(np.float64)  # Ensure target is float64

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Load label encoders
label_encoders = joblib.load("models/flight_price_label_encoders.pkl")

for col in categorical_cols:
    if col in label_encoders:
        X[col] = label_encoders[col].transform(X[col])
    else:
        raise KeyError(f"Missing label encoder for {col}")

# Convert integer columns to float64 in X
to_convert = X.select_dtypes(include=['int']).columns
X[to_convert] = X[to_convert].astype(np.float64)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load Scaler
scaler = joblib.load("models/flight_price_scaler.pkl")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

logging.info("✅ Data Preprocessing completed.")

# Define Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    "Support Vector Machine": SVR(kernel='rbf', C=100, gamma=0.1)
}

# MLflow Experiment
experiment_name = "Flight_Price_Predictions"
mlflow.set_experiment(experiment_name)

# Results Storage
results = []

for name, model in models.items():
    logging.info(f"🚀 Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    logging.info(f"🏆 {name} -> R²: {r2:.4f}, MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")
    
    # Save Model
    model_path = f"models/{name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, model_path)
    
    # Log in MLflow
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model", name)
        mlflow.log_metric("R2_Score", r2)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.sklearn.log_model(model, f"{name}_model", input_example=X_test_scaled.iloc[:5])
    
    results.append([name, r2, mae, mse, rmse])

# Save Results
results_df = pd.DataFrame(results, columns=["Model", "R2_Score", "MAE", "MSE", "RMSE"])
results_df.to_csv("results/flight_price_model_comparison.csv", index=False)
logging.info("📊 All model scores saved and logged in MLflow!")
