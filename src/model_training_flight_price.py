import pandas as pd
import numpy as np
import joblib
import os
import logging
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure necessary directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Initialize MLflow
mlflow.set_experiment("Flight Price Prediction")

# Load dataset
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
y = df[target_col].astype(np.float64)  # Convert target column to float64

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])
    label_encoders[col] = encoder

# Convert integer columns to float64 to prevent MLflow warnings
for col in X.select_dtypes(include=['int']).columns:
    X[col] = X[col].astype(np.float64)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to retain feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Save the scaler and encoders
joblib.dump(scaler, "models/flight_price_scaler.pkl")
joblib.dump(label_encoders, "models/flight_price_label_encoders.pkl")
logging.info("✅ Flight Price Scaler and Label Encoders saved.")

# Define models
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel="rbf"),
}

best_model = None
best_score = -np.inf
results = []

# Train and evaluate models
for model_name, model in models.items():
    with mlflow.start_run(run_name=f"{model_name}_FlightPrice"):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate performance metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Log metrics in MLflow
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)

        # Save model
        model_filename = f"models/{model_name}_flight_price.pkl"
        joblib.dump(model, model_filename)
        mlflow.sklearn.log_model(model, model_name, input_example=X_test_scaled.iloc[:5])
        logging.info(f"✅ {model_name} Model saved with R² Score: {r2:.4f}")

        # Store results
        results.append([model_name, r2, mae, mse, rmse])

        # Update best model
        if r2 > best_score:
            best_score = r2
            best_model = model

# Save best model
joblib.dump(best_model, "models/best_flight_price_model.pkl")
logging.info(f"🏆 Best Flight Price Model saved with R² Score: {best_score:.4f}")

# Save results to CSV with all metrics
results_df = pd.DataFrame(results, columns=["Model", "R2_Score", "MAE", "MSE", "RMSE"])
results_df.to_csv("results/flight_price_model_scores.csv", index=False)
logging.info("📊 Model scores saved to results/flight_price_model_scores.csv")
