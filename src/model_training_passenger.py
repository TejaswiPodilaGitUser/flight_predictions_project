import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load dataset
logging.info("🚀 Loading passenger dataset...")
data = pd.read_csv("data/processed/Cleaned_Processed_Passenger_Satisfaction.csv")
X = data.drop(columns=["satisfaction"])
y = data["satisfaction"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed test data
joblib.dump(X_test_scaled, "models/X_test_passenger.pkl")
joblib.dump(y_test, "models/y_test_passenger.pkl")

# Initialize MLflow
mlflow.set_experiment("Passenger Satisfaction Predictions")

models = {
    "LogisticRegression": LogisticRegression(max_iter=500, solver='lbfgs'),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(iterations=200, verbose=0, random_seed=42)
}

results = []

for name, model in models.items():
    logging.info(f"🏁 Training {name} started...")

    # Convert back to DataFrame for LightGBM
    if name == "LightGBM":
        X_train_final = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_final = pd.DataFrame(X_test_scaled, columns=X.columns)
    else:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled

    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append([name, acc, f1])

    with mlflow.start_run():
        mlflow.sklearn.log_model(model, name, input_example=X_test_final[:5])
        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("F1 Score", f1)

    joblib.dump(model, f"models/{name}_passenger.pkl")
    logging.info(f"✅ {name} Model saved with Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

# Identify best model
best_model_name, best_acc, best_f1 = max(results, key=lambda x: x[1])  # Choose based on accuracy
best_model_path = f"models/{best_model_name}_passenger.pkl"

# Save the best model separately
best_model = joblib.load(best_model_path)
joblib.dump(best_model, "models/best_passenger_satisfaction_model.pkl")

logging.info(f"🏆 Best Model: {best_model_name} saved as 'best_passenger_satisfaction_model.pkl' with Accuracy: {best_acc:.4f}, F1 Score: {best_f1:.4f}")


# Save results
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 Score"])
results_df.to_csv("results/passenger_model_results.csv", index=False)
logging.info("✅ Model training completed and results saved.")
