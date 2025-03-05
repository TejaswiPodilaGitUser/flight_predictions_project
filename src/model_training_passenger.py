import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load dataset
logging.info("🚀 Loading passenger dataset...")
data = pd.read_csv("data/processed/Cleaned_Processed_Passenger_Satisfaction.csv")

X = data.drop(columns=["satisfaction"])
y = data["satisfaction"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Before SMOTE: Log the count of the original class distribution
positive_before = y_train.value_counts()[1] if 1 in y_train.value_counts() else 0
negative_before = y_train.value_counts()[0] if 0 in y_train.value_counts() else 0
# Before SMOTE: Log the count of the original class distribution
logging.info("🚀 Original satisfaction distribution ")
logging.info("--------------------------------------")
logging.info(f"🚀 Satisfied: {positive_before}")
logging.info(f"🚀 Not Satisfied: {negative_before}")
logging.info("--------------------------------------")

# Handle class imbalance using SMOTE
logging.info("🚀 Handling class imbalance using SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# After SMOTE: Log the count of the resampled class distribution
positive_after = y_train_resampled.value_counts()[1] if 1 in y_train_resampled.value_counts() else 0
negative_after = y_train_resampled.value_counts()[0] if 0 in y_train_resampled.value_counts() else 0
# After SMOTE: Log the count of the resampled class distribution
logging.info("✅ Resampled satisfaction distribution")
logging.info("--------------------------------------")
logging.info(f"✅ Satisfied: {positive_after}")
logging.info(f"✅ Not Satisfied: {negative_after}")
logging.info("--------------------------------------")

# Standardize features AFTER resampling
logging.info("🚀 Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)
logging.info("✅ Features scaled.")

# Save processed test data
joblib.dump(X_test_scaled, "models/X_test_passenger.pkl")
joblib.dump(y_test, "models/y_test_passenger.pkl")


# Initialize MLflow
mlflow.set_experiment("Passenger Satisfaction Predictions")

# Models dictionary (Top 5 Models)
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]), random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(iterations=200, verbose=0, random_seed=42)
}

results = []
feature_importance_dict = {}

# Train each model
for name, model in models.items():
    logging.info(f"🏁 Training {name} started...")

    # Convert back to DataFrame for LightGBM
    if name == "LightGBM":
        X_train_final = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_final = pd.DataFrame(X_test_scaled, columns=X.columns)
    else:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled

    model.fit(X_train_final, y_train_resampled)
    y_probs = model.predict_proba(X_test_final)[:, 1]
    threshold = 0.4
    y_pred_adjusted = (y_probs >= threshold).astype(int)

    # Evaluate model with balanced metrics
    acc = accuracy_score(y_test, y_pred_adjusted)
    f1 = f1_score(y_test, y_pred_adjusted)
    precision = precision_score(y_test, y_pred_adjusted)
    recall = recall_score(y_test, y_pred_adjusted)
    train_acc = accuracy_score(y_train_resampled, model.predict(X_train_final))

    logging.info(f"{name} - Train Accuracy: {train_acc:.4f}, Test Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    results.append([name, acc, f1, precision, recall])

    with mlflow.start_run():
        mlflow.sklearn.log_model(model, name, input_example=X_test_final[:5])
        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("F1 Score", f1)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)

    joblib.dump(model, f"models/{name}_passenger.pkl")

    if hasattr(model, "feature_importances_"):
        feature_importance_dict[name] = model.feature_importances_

    logging.info(f"✅ {name} Model saved with Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

# Select best model based on F1 Score
best_model_name, best_acc, best_f1, best_precision, best_recall = max(results, key=lambda x: x[2])
best_model_path = f"models/{best_model_name}_passenger.pkl"
best_model = joblib.load(best_model_path)

# Save the best model separately
joblib.dump(best_model, "models/best_passenger_satisfaction_model.pkl")
logging.info(f"🏆 Best Model: {best_model_name} saved as 'best_passenger_satisfaction_model.pkl' with Accuracy: {best_acc:.4f}, F1 Score: {best_f1:.4f}")

# Save feature importance & top 10 features
if best_model_name in feature_importance_dict:
    feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance_dict[best_model_name]})
    top_10_features = feature_importance_df.sort_values(by="Importance", ascending=False)["Feature"].head(10).tolist()
else:
    logging.warning(f"⚠️ No feature importance available for {best_model_name}, selecting default top 10 features.")
    top_10_features = X.columns[:10].tolist()

joblib.dump(top_10_features, "models/top_10_features.pkl")
joblib.dump(best_model, "models/best_passenger_model.pkl")

logging.info(f"✅ Top 10 Features Selected: {top_10_features}")

# Save results
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 Score", "Precision", "Recall"])
results_df.to_csv("results/passenger_model_results.csv", index=False)
logging.info("✅ Model training completed and results saved.")
