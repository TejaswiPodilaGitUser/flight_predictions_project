import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set Streamlit page layout to wide
#st.set_page_config(page_title="Flight Price EDA", layout="wide")

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

def plot_route_vs_price(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    top_routes = df["Route"].value_counts().index[:10]
    df_filtered = df[df["Route"].isin(top_routes)]
    sns.boxplot(x="Route", y="Price", data=df_filtered, ax=ax)
    ax.set_title("Price Distribution Across Top Routes")
    plt.xticks(rotation=30, ha="right")
    return fig

def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    numeric_df = df.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.xticks(rotation=30, ha="right")
    return fig


def main():
    # Streamlit App
    st.markdown("<h1 style='text-align: center;'>✈️ Flight Price EDA Dashboard</h1>", unsafe_allow_html=True)


    df = load_dataset("data/processed/Cleaned_Processed_Flight_Price.csv")


    # Visualizations
    st.markdown("<h3 style='text-align: center;'>📊 Visualizations</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(plot_price_distribution(df))
        st.pyplot(plot_route_vs_price(df))

    with col2:
        st.pyplot(plot_airline_vs_price(df))
        st.pyplot(plot_correlation_heatmap(df))


    # Display top and bottom 5 prices
    st.markdown("<h3 style='text-align: center;'>💰 Top & Least 5 Prices</h3>", unsafe_allow_html=True)

    # Create empty space between columns using three columns
    spacer,col1, spacer, col2 = st.columns([0.4,1, 0.2, 1])  # Adjusting widths (0.2 is the space)

    with col1:
        st.write("🔝 Top 5 Highest Prices")
        st.dataframe(df.nlargest(5, "Price")[["Airline", "Route", "Price"]])
    with col2:
        st.write("🔻 Lowest 5 Prices")
        st.dataframe(df.nsmallest(5, "Price")[["Airline", "Route", "Price"]])

    st.success("✅ EDA Analysis Completed!")

if __name__ == "__main__":
    main()