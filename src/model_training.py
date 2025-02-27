import pandas as pd
import numpy as np
import joblib
import os
import logging
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure necessary directories exist
os.makedirs("models", exist_ok=True)

# Initialize MLflow
mlflow.set_experiment("Customer Satisfaction Prediction")

def load_dataset(file_path, target_col):
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

    # Save X_test and y_test for confusion matrix generation
    joblib.dump(X_test_scaled, "models/X_test.pkl")
    joblib.dump(y_test, "models/y_test.pkl")
    logging.info("✅ X_test and y_test saved for confusion matrix generation.")

    return X_train_scaled, X_test_scaled, y_train, y_test

def train_satisfaction_models(X_train, X_test, y_train, y_test):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    best_model = None
    best_score = -np.inf
    model_results = []
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_Satisfaction"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            
            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            
            model_filename = f"models/{model_name}_satisfaction.pkl"
            joblib.dump(model, model_filename)
            mlflow.sklearn.log_model(model, model_name, input_example=X_test[:5])
            
            model_results.append([model_name, accuracy, f1])
            logging.info(f"✅ {model_name} Model saved at {model_filename} with Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
    
    joblib.dump(best_model, "models/best_satisfaction_model.pkl")
    logging.info(f"🏆 Best Satisfaction Model saved with Accuracy: {best_score:.4f}")
    return model_results

# Run training
df = load_dataset("data/processed/Cleaned_Processed_Passenger_Satisfaction.csv", "satisfaction")
X_train, X_test, y_train, y_test = preprocess_data(df, "satisfaction")
train_satisfaction_models(X_train, X_test, y_train, y_test)
