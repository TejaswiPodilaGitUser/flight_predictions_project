import pandas as pd
import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure necessary directories exist
os.makedirs("models/confusion_matrices", exist_ok=True)

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

def plot_combined_eda(df, target_col, y_true=None, y_pred=None, model_name=None):
    """Plots Satisfaction Distribution & Confusion Matrix in Row 1, Correlation Heatmap in Row 2."""
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1]})

    # Plot 1 (Row 1, Col 1): Satisfaction Distribution
    sns.countplot(x=df[target_col], ax=axes[0, 0])
    axes[0, 0].set_title("Distribution of Satisfaction Levels")

    # Plot 2 (Row 1, Col 2): Confusion Matrix (if available)
    if y_true is not None and y_pred is not None and model_name is not None:
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Satisfied", "Satisfied"], yticklabels=["Not Satisfied", "Satisfied"], ax=axes[0, 1])
        axes[0, 1].set_xlabel("Predicted")
        axes[0, 1].set_ylabel("Actual")
        axes[0, 1].set_title(f"Confusion Matrix - {model_name}")
    else:
        axes[0, 1].axis("off")  # Hide plot if no confusion matrix available

    # Plot 3 (Row 2, Span 2 Columns): Correlation Heatmap (Top Features)
    top_corr_features = df.corr()["satisfaction"].abs().sort_values(ascending=False)[1:10].index
    sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1, 0])
    axes[1, 0].set_title("Top 10 Feature Correlations")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Hide the empty subplot in (Row 2, Col 2)
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("models/eda_confusion_matrix.png")
    plt.show()  # Show all plots in a single figure
    logging.info("📊 Combined EDA & Confusion Matrix saved.")

def generate_confusion_matrix():
    """Loads the best model and generates confusion matrix."""
    try:
        X_test = joblib.load("models/X_test.pkl")
        y_test = joblib.load("models/y_test.pkl")
    except FileNotFoundError:
        logging.error("❌ X_test or y_test not found. Run `model_training.py` first.")
        return None, None, None
    
    best_model_path = "models/best_satisfaction_model.pkl"
    if not os.path.exists(best_model_path):
        logging.error("❌ Best model not found. Run `model_training.py` first.")
        return None, None, None
    
    best_model = joblib.load(best_model_path)
    model_name = type(best_model).__name__
    
    y_pred = best_model.predict(X_test)
    return y_test, y_pred, model_name

if __name__ == "__main__":
    df = load_dataset("data/processed/Cleaned_Processed_Passenger_Satisfaction.csv", "satisfaction")
    y_test, y_pred, model_name = generate_confusion_matrix()
    plot_combined_eda(df, "satisfaction", y_test, y_pred, model_name)
