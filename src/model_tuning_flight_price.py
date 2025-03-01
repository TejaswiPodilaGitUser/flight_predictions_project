import pandas as pd
import numpy as np
import joblib
import os
import logging
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
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

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Target column check
target_col = "Price"
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found in the dataset.")

# Prepare features and target
X = df.drop(columns=[target_col])
y = df[target_col].astype(np.float64)  # Ensure target is float64

# Load label encoders
label_encoders = joblib.load("models/flight_price_label_encoders.pkl")
for col in categorical_cols:
    if col in label_encoders:
        X[col] = label_encoders[col].transform(X[col])

# Convert integer columns to float64 to prevent MLflow warnings
for col in X.select_dtypes(include=['int']).columns:
    X[col] = X[col].astype(np.float64)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load Scaler
scaler = joblib.load("models/flight_price_scaler.pkl")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to retain feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

logging.info("✅ Data Preprocessing completed.")

# Load Best Model for Tuning
best_model_path = "models/best_flight_price_model.pkl"
if not os.path.exists(best_model_path):
    raise FileNotFoundError("Best trained model not found!")

best_model = joblib.load(best_model_path)
logging.info("✅ Best model loaded for tuning.")

# Define Hyperparameter Grids
param_grid = {}

if isinstance(best_model, XGBRegressor):
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0]
    }
elif isinstance(best_model, RandomForestRegressor):
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
else:
    raise TypeError("Best model is not XGBoost or RandomForest. Tuning not supported!")

# Perform Hyperparameter Tuning
grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring="r2", verbose=2, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Get Best Model
tuned_model = grid_search.best_estimator_
best_params = grid_search.best_params_
logging.info(f"✅ Best hyperparameters found: {best_params}")

# Evaluate Model
y_pred = tuned_model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

logging.info(f"🏆 Tuned Model Performance -> R²: {r2:.4f}, MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")

# Save Tuned Model
tuned_model_path = "models/tuned_flight_price_model.pkl"
joblib.dump(tuned_model, tuned_model_path)
logging.info(f"✅ Tuned Model saved at {tuned_model_path}")

# Log to MLflow
with mlflow.start_run(run_name="Tuned_FlightPrice"):
    mlflow.log_params(best_params)
    mlflow.log_metric("R2_Score", r2)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.sklearn.log_model(tuned_model, "tuned_flight_price_model", input_example=X_test_scaled.iloc[:5])

# Save Updated Results
results_df = pd.DataFrame([[type(tuned_model).__name__, r2, mae, mse, rmse]],
                          columns=["Model", "R2_Score", "MAE", "MSE", "RMSE"])
results_df.to_csv("results/tuned_flight_price_model_scores.csv", index=False)
logging.info("📊 Tuned model scores saved.")
