import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
@st.cache_data
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df

# Visualization Functions
def plot_price_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["Price"], bins=50, kde=True, ax=ax)
    ax.set_title("Flight Price Distribution")
    plt.xticks(rotation=30, ha="right")
    return fig

def plot_airline_vs_price(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    airline_avg_price = df.groupby("Airline")["Price"].mean().sort_values()
    sns.barplot(x=airline_avg_price.index, y=airline_avg_price.values, ax=ax)
    ax.set_title("Average Price by Airline")
    ax.set_xlabel("Airline")
    ax.set_ylabel("Average Price")
    plt.xticks(rotation=30, ha="right")
    return fig

def plot_stops_vs_price(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=df["Total_Stops"], y=df["Price"], ax=ax)
    ax.set_title("Total Stops vs Price")
    ax.set_xlabel("Total Stops")
    ax.set_ylabel("Price")
    return fig

def plot_duration_vs_price(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=df["Duration_Minutes"], y=df["Price"], ax=ax)
    ax.set_title("Flight Duration vs Price")
    ax.set_xlabel("Duration (Minutes)")
    ax.set_ylabel("Price")
    return fig

def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    numeric_df = df.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.xticks(rotation=30, ha="right")
    return fig

def main():
    st.markdown("<h1 style='text-align: center;'>✈️ Flight Price EDA Dashboard</h1>", unsafe_allow_html=True)
    
    df = load_dataset("data/processed/Cleaned_Processed_Flight_Price.csv")
    
    st.markdown("<h3 style='text-align: center;'>📊 Visualizations</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.pyplot(plot_price_distribution(df))
        st.pyplot(plot_stops_vs_price(df))
    
    with col2:
        st.pyplot(plot_airline_vs_price(df))
        st.pyplot(plot_duration_vs_price(df))
        st.pyplot(plot_correlation_heatmap(df))
    
    st.markdown("<h3 style='text-align: center;'>💰 Top & Least 5 Prices</h3>", unsafe_allow_html=True)
    spacer, col1, spacer, col2 = st.columns([0.4,1, 0.2, 1])
    
    with col1:
        st.write("🔝 Top 5 Highest Prices")
        st.dataframe(df.nlargest(5, "Price")[["Airline", "Source", "Destination", "Price"]])
    with col2:
        st.write("🔻 Lowest 5 Prices")
        st.dataframe(df.nsmallest(5, "Price")[["Airline", "Source", "Destination", "Price"]])
    
    st.success("✅ EDA Analysis Completed!")

if __name__ == "__main__":
    main()