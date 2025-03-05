import pandas as pd
import numpy as np
import joblib
import os
import logging
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load Dataset
passenger_data_path = "data/processed/Cleaned_Processed_Passenger_Satisfaction.csv"
if not os.path.exists(passenger_data_path):
    raise FileNotFoundError(f"Dataset not found at {passenger_data_path}")

df = pd.read_csv(passenger_data_path)
if df.empty:
    raise ValueError("The dataset is empty!")

logging.info("✅ Passenger dataset loaded successfully.")

# Clean column names
df.columns = df.columns.str.strip()

# Ensure correct target column
target_col = "satisfaction"
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found in the dataset. Available columns: {df.columns.tolist()}")

# Prepare features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the scaled data back to DataFrame with original column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Save Scaler
joblib.dump(scaler, "models/passenger_satisfaction_scaler.pkl")

logging.info("✅ Data Preprocessing completed and saved.")

# Load Best Model for Tuning
best_model_path = "models/best_passenger_satisfaction_model.pkl"
if not os.path.exists(best_model_path):
    raise FileNotFoundError("Best trained model not found!")

best_model = joblib.load(best_model_path)
logging.info("✅ Best model loaded for tuning.")

# Define Hyperparameter Grids for different models
param_grids = {
    XGBClassifier: {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "scale_pos_weight": [1, 1.5, 2]  # Adjust the class weight to prevent bias
    },
    RandomForestClassifier: {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    LGBMClassifier: {
        "num_leaves": [31, 50, 100],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200, 300]
    },
    CatBoostClassifier: {
        "iterations": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [3, 5, 7]
    },
    LogisticRegression: {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
    }
}

# Check if the model is in the param_grids dictionary, otherwise raise an error
if type(best_model) not in param_grids:
    raise TypeError(f"Best model {type(best_model).__name__} not supported for hyperparameter tuning!")

# Perform Hyperparameter Tuning using GridSearchCV
param_grid = param_grids[type(best_model)]
grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring="accuracy", verbose=2, n_jobs=-1)

logging.info(f"✅ Starting Grid Search for {type(best_model).__name__}")
grid_search.fit(X_train_scaled, y_train)

# Get Best Model and Hyperparameters
tuned_model = grid_search.best_estimator_
best_params = grid_search.best_params_
logging.info(f"✅ Best hyperparameters found: {best_params}")

# Evaluate Tuned Model
y_pred = tuned_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

logging.info(f"🏆 Tuned Model Performance -> Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
logging.info(f"📊 Confusion Matrix:\n{conf_matrix}")
logging.info(f"📜 Classification Report:\n{class_report}")

# Save Tuned Model
tuned_model_path = "models/tuned_passenger_satisfaction_model.pkl"
joblib.dump(tuned_model, tuned_model_path)
logging.info(f"✅ Tuned Model saved at {tuned_model_path}")

# Log results to MLflow
mlflow.set_experiment("PassengerSatisfaction_Tuning")
with mlflow.start_run(run_name="Tuned_PassengerSatisfaction"):
    mlflow.log_params(best_params)
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("F1_Score", f1)
    mlflow.sklearn.log_model(tuned_model, "tuned_passenger_satisfaction_model", input_example=pd.DataFrame(X_test_scaled[:5], columns=X.columns))

# Save Updated Results to CSV
results_df = pd.DataFrame([[type(tuned_model).__name__, accuracy, f1]],
                          columns=["Model", "Accuracy", "F1_Score"])
results_df.to_csv("results/tuned_passenger_satisfaction_model_scores.csv", index=False)
logging.info("📊 Tuned model scores saved.")

