import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import pandas as pd


# ✅ Function to plot Age Distribution
def plot_age_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Age"], bins=20, kde=True, color="skyblue")
    plt.title("Distribution of Passenger Ages")
    plt.xlabel("Age")
    plt.ylabel("Count")
    st.pyplot(plt)


# ✅ Function to plot Flight Distance vs Satisfaction
def plot_flight_distance_vs_satisfaction(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["satisfaction"], y=df["Flight Distance"], palette="coolwarm")
    plt.title("Flight Distance vs Passenger Satisfaction")
    plt.xlabel("Satisfaction")
    plt.ylabel("Flight Distance (km)")
    st.pyplot(plt)


# ✅ Function to plot Departure Delay vs Satisfaction
def plot_delay_vs_satisfaction(df):
    plt.figure(figsize=(8, 5))
    sns.violinplot(x=df["satisfaction"], y=df["Departure Delay in Minutes"], palette="magma")
    plt.title("Departure Delay vs Satisfaction")
    plt.xlabel("Satisfaction")
    plt.ylabel("Departure Delay (mins)")
    st.pyplot(plt)


# ✅ Function to plot Class-wise Satisfaction
def plot_class_vs_satisfaction(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df["Class"], hue=df["satisfaction"], palette="Set1")
    plt.title("Class-wise Passenger Satisfaction")
    plt.xlabel("Class")
    plt.ylabel("Count")
    st.pyplot(plt)

# ✅ Function to plot Feature Importance Dynamically
def plot_feature_importance(model_path, df):
    try:
        # Load the trained model
        model = joblib.load(model_path)

        # Get feature names and importance values from the model
        model_features = model.feature_name_  # Keep original names
        importance_values = model.feature_importances_

        # ✅ Rename columns to match the model's expected format
        df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))

        # Check for mismatches
        extra_features = set(df.columns) - set(model_features)
        missing_features = set(model_features) - set(df.columns)

        #if extra_features or missing_features:
         #   st.warning(f"Feature mismatch detected! Extra in DataFrame: {extra_features}, Missing in DataFrame: {missing_features}")

        # Select only the model's features from the DataFrame
        df = df[model_features]

        # Create DataFrame for plotting
        feature_importance_df = pd.DataFrame({
            "Feature": model_features,
            "Importance": importance_values
        })

        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis")
        plt.title("Feature Importance in Model")
        plt.xlabel("Importance Score")
        st.pyplot(plt)

    except KeyError as e:
        st.error(f"Column mismatch error! Ensure DataFrame has correct feature names: {e}")
    except Exception as e:
        st.error(f"Error in plotting feature importance: {e}")