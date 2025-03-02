import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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

# ✅ Function to plot Feature Importance (Example Data)
def plot_feature_importance(df):
    feature_names = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]  # Example
    importance_values = [0.3, 0.4, 0.15, 0.15]  # Example
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importance_values, y=feature_names, palette="viridis")
    plt.title("Feature Importance in Model")
    plt.xlabel("Importance Score")
    st.pyplot(plt)
