import pandas as pd
import joblib
import os
import logging
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, r2_score

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure necessary directories exist
os.makedirs("models", exist_ok=True)

# Initialize MLflow
mlflow.set_experiment("Model Tuning - Flight & Satisfaction")

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
    y = df[target_col].astype(float)  # Ensuring target is numeric

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def tune_models(X_train, X_test, y_train, y_test, model_type):
    """Perform hyperparameter tuning for multiple models and select the best."""
    models = {
        "XGBRegressor": (XGBRegressor(tree_method="hist", random_state=42), {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2]
        }),
        "RandomForest": (RandomForestClassifier(random_state=42), {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10]
        }),
        "GradientBoosting": (GradientBoostingClassifier(random_state=42), {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        }),
        "LogisticRegression": (LogisticRegression(max_iter=1000), {
            "C": [0.1, 1, 10],
            "solver": ["liblinear", "lbfgs"]
        }),
        "SVM": (SVC(probability=True), {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }),
        "DecisionTree": (DecisionTreeClassifier(random_state=42), {
            "max_depth": [5, 10, 20],
            "min_samples_split": [2, 5, 10]
        }),
        "KNN": (KNeighborsClassifier(), {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"]
        })
    }

    best_model = None
    best_score = -float("inf")
    best_model_name = ""

    for model_name, (model, param_grid) in models.items():
        logging.info(f"🔍 Tuning {model_name}...")
        search = RandomizedSearchCV(model, param_grid, n_iter=10, scoring="accuracy" if model_type == "classification" else "r2",
                                    cv=3, n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)
        
        best_model_current = search.best_estimator_
        y_pred = best_model_current.predict(X_test)

        if model_type == "classification":
            score = accuracy_score(y_test, y_pred)
        else:
            score = r2_score(y_test, y_pred)

        logging.info(f"✅ Best {model_name} Model - Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = best_model_current
            best_model_name = model_name

    logging.info(f"🏆 Best Model: {best_model_name} with Score: {best_score:.4f}")

    return best_model, best_model_name, best_score

# Flight Price Prediction - Regression Models
flight_data_path = "data/processed/Cleaned_Processed_Flight_Price.csv"
df_flight = load_dataset(flight_data_path, "Price")
X_train_flight, X_test_flight, y_train_flight, y_test_flight = preprocess_data(df_flight, "Price")
best_flight_model, best_flight_model_name, best_flight_score = tune_models(X_train_flight, X_test_flight, y_train_flight, y_test_flight, "regression")

# Save Best Flight Model
best_flight_model_filename = "models/best_flight_price_model.pkl"
joblib.dump(best_flight_model, best_flight_model_filename)
logging.info(f"✅ Best Flight Price Model ({best_flight_model_name}) saved at {best_flight_model_filename}")

# Customer Satisfaction Prediction - Classification Models
satisfaction_data_path = "data/processed/Cleaned_Processed_Passenger_Satisfaction.csv"
df_satisfaction = load_dataset(satisfaction_data_path, "satisfaction")
X_train_satisfaction, X_test_satisfaction, y_train_satisfaction, y_test_satisfaction = preprocess_data(df_satisfaction, "satisfaction")
best_satisfaction_model, best_satisfaction_model_name, best_satisfaction_score = tune_models(X_train_satisfaction, X_test_satisfaction, y_train_satisfaction, y_test_satisfaction, "classification")

# Save Best Satisfaction Model
best_satisfaction_model_filename = "models/best_satisfaction_model.pkl"
joblib.dump(best_satisfaction_model, best_satisfaction_model_filename)
logging.info(f"✅ Best Satisfaction Model ({best_satisfaction_model_name}) saved at {best_satisfaction_model_filename}")

# Store Model Performance in CSV
performance_data = {
    "Model": [best_flight_model_name, best_satisfaction_model_name],
    "Score": [best_flight_score, best_satisfaction_score]
}
performance_df = pd.DataFrame(performance_data)
performance_df.to_csv("models/model_performance.csv", index=False)
logging.info("📊 Model Performance saved in 'models/model_performance.csv'")
