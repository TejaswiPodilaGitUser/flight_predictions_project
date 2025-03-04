import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Function to plot Airline vs Average Price
def plot_airline_vs_price(df):
    avg_price_per_airline = df.groupby("Airline")["Price"].mean().sort_values()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=avg_price_per_airline.index, y=avg_price_per_airline.values, palette="coolwarm")
    plt.xticks(rotation=45, ha="right")
    plt.title("✈️ Average Flight Price by Airline")
    plt.xlabel("Airline")
    plt.ylabel("Average Price (₹)")
    st.pyplot(plt)

# ✅ Function to plot Source vs Destination Flight Prices
def plot_source_vs_destination(df):
    avg_price_per_route = df.groupby(["Source", "Destination"])["Price"].mean().unstack()
    plt.figure(figsize=(8, 5))
    sns.heatmap(avg_price_per_route, cmap="Blues", annot=True, fmt=".0f")
    plt.title("🌍 Flight Price Heatmap (Source → Destination)")
    plt.xlabel("Destination")
    plt.ylabel("Source")
    st.pyplot(plt)

# ✅ Function to plot Stops vs Flight Price
def plot_stops_vs_price(df):
    avg_price_per_stop = df.groupby("Total_Stops")["Price"].mean()
    plt.figure(figsize=(8, 4))
    sns.barplot(x=avg_price_per_stop.index, y=avg_price_per_stop.values, palette="magma")
    plt.title("⏳ Flight Price by Number of Stops")
    plt.xlabel("Total Stops")
    plt.ylabel("Average Price (₹)")
    st.pyplot(plt)

# ✅ Function to plot Departure Hour vs Price
def plot_dep_hour_vs_price(df):
    avg_price_per_hour = df.groupby("Dep_Hour")["Price"].mean()
    plt.figure(figsize=(8, 4))
    sns.lineplot(x=avg_price_per_hour.index, y=avg_price_per_hour.values, marker="o", color="green")
    plt.title("🕒 Flight Price by Departure Hour")
    plt.xlabel("Departure Hour")
    plt.ylabel("Average Price (₹)")
    st.pyplot(plt)

# ✅ Function to plot Flight Duration vs Price
def plot_duration_vs_price(df):
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=df["Duration_Minutes"], y=df["Price"], alpha=0.5, color="purple")
    plt.title("⌛ Flight Price vs Duration")
    plt.xlabel("Flight Duration (minutes)")
    plt.ylabel("Price (₹)")
    st.pyplot(plt)

# ✅ Function to call all plots
def show_all_flight_price_plots(df):
    st.subheader("📊 Flight Price Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        plot_airline_vs_price(df)
        plot_stops_vs_price(df)

    with col2:
        plot_source_vs_destination(df)
        plot_dep_hour_vs_price(df)

    st.subheader("📉 Additional Insights")
    plot_duration_vs_price(df)
