import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import joblib

# Set Streamlit page layout to wide
st.set_page_config(page_title="Passenger Satisfaction EDA", layout="wide")

def load_dataset(file_path):
    """Loads the dataset and returns a DataFrame."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df

def display_insights(df):
    """Generates key insights from the dataset and displays as a table."""
    total_passengers = len(df)
    satisfaction_counts = df["satisfaction"].value_counts().to_dict()
    
    insights_data = {
        "Metric": ["Total Passengers", "Satisfied Passengers", "Not Satisfied Passengers"],
        "Value": [total_passengers, satisfaction_counts.get(1, 0), satisfaction_counts.get(0, 0)]
    }
    
    insights_df = pd.DataFrame(insights_data)
    st.write("### 📋 Dataset Insights")
    st.table(insights_df)

def plot_satisfaction_distribution(df):
    """Plots the distribution of satisfaction levels."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=df["satisfaction"], ax=ax)
    return fig

def plot_feature_correlations(df):
    """Plots feature correlations with satisfaction."""
    fig, ax = plt.subplots(figsize=(6, 4))
    top_corr_features = df.corr()["satisfaction"].abs().sort_values(ascending=False)[1:10].index
    sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    plt.xticks(rotation=30, ha="right")
    return fig

def load_model_and_data():
    """Loads the trained model and test data."""
    try:
        X_test = joblib.load("models/X_test_passenger.pkl")
        y_test = joblib.load("models/y_test_passenger.pkl")
        best_model = joblib.load("models/best_passenger_satisfaction_model.pkl")
        model_name = type(best_model).__name__
        return X_test, y_test, best_model, model_name
    except FileNotFoundError:
        st.error("❌ Required model or test data files not found. Run `model_training_passenger.py` first.")
        return None, None, None, None

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plots the confusion matrix for the best model."""
    fig, ax = plt.subplots(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Satisfied", "Satisfied"], yticklabels=["Not Satisfied", "Satisfied"], ax=ax)
    return fig

def main():
    st.markdown("<h1 style='text-align: center;'>🛫 Passenger Satisfaction EDA</h1><br>", unsafe_allow_html=True)
    
    df = load_dataset("data/processed/Cleaned_Processed_Passenger_Satisfaction.csv")
    
    # FIRST ROW: Insights Table & Satisfaction Distribution
    spacer,col1, spacer, col2 , spacer= st.columns([0.3,0.5, 0.2, 0.6,0.2])  # Adjusting widths (0.2 is the space)
    with col1:
        display_insights(df)

    with col2:
        st.write("### 📊 Distribution of Satisfaction Levels")
        st.pyplot(plot_satisfaction_distribution(df))

    # Load Model & Data for Confusion Matrix
    X_test, y_test, best_model, model_name = load_model_and_data()
    
    if X_test is not None and y_test is not None and best_model is not None:
        y_pred = best_model.predict(X_test)

        # SECOND ROW: Feature Correlation & Confusion Matrix
        spacer,col3, spacer, col4 , spacer= st.columns([0.1,1, 0.2, 0.8,0.2])  # Adjusting widths (0.2 is the space)
        with col3:
            st.write("### 🔥 Feature Correlations with Satisfaction")
            st.pyplot(plot_feature_correlations(df))

        with col4:
            st.write("### 🤖 Confusion Matrix")
            st.pyplot(plot_confusion_matrix(y_test, y_pred, model_name))

if __name__ == "__main__":
    main()
