import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load dataset
data_path = "data/processed/Cleaned_Processed_Passenger_Satisfaction.csv"
df = pd.read_csv(data_path)
logging.info("✅ Passenger satisfaction dataset loaded successfully.")

# Handle missing values
df.fillna(df.median(), inplace=True)  # Replace NaNs with median values


# Fix column names
df.columns = [re.sub(r"[^\w]", "_", col) for col in df.columns]

# Convert all numeric columns to float64 to prevent MLflow warnings
df = df.astype(np.float64)

# Define features and target
target_col = "satisfaction"
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logging.info("✅ Passenger Scaler and Label Encoders saved.")

# Convert back to DataFrame
X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

# MLflow experiment setup
mlflow.set_experiment("Passenger_Satisfaction")

models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(force_col_wise=True, random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

results = []

with mlflow.start_run():
    for name, model in models.items():
        logging.info(f"🏁 Training {name} started...")

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Use a valid input example for MLflow logging (Avoid NaNs)
        input_example = X_train.sample(1).to_dict(orient="records")[0]

        mlflow.sklearn.log_model(model, name, input_example=input_example)
        mlflow.log_metric(f"{name}_accuracy", accuracy)
        mlflow.log_metric(f"{name}_f1_score", f1)

        results.append((name, accuracy, f1))
        logging.info(f"✅ {name} Model saved with Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

# Save results to CSV
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1_Score"])
results_df.to_csv("results/passenger_model_scores.csv", index=False)

# Find best model based on F1 score
best_model = max(results, key=lambda x: x[2])  # Prioritize F1 Score
logging.info(f"🏆 Best Passenger Satisfaction Model: {best_model[0]} with F1 Score: {best_model[2]:.4f}")
