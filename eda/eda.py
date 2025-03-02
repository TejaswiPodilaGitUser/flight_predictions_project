import streamlit as st
from eda_flight import load_dataset as load_flight_data, plot_price_distribution, plot_airline_vs_price, plot_route_vs_price, plot_correlation_heatmap
from eda_passenger import load_dataset as load_passenger_data, display_insights, plot_satisfaction_distribution, plot_feature_correlations, load_model_and_data, plot_confusion_matrix

# Set Streamlit page layout only in the main script
st.set_page_config(page_title="EDA Dashboard", layout="wide")

st.markdown("<h1 style='text-align: center;'>✈️ EDA Dashboard</h1>", unsafe_allow_html=True)

# 🔴 IMPORTANT: Ensure tabs are the FIRST elements in Streamlit
tab1, tab2 = st.tabs(["✈️ Flight Price EDA", "🛫 Passenger Satisfaction EDA"])

# Flight Price EDA Tab
with tab1:
    st.markdown("<h2 style='text-align: center;'>✈️ Flight Price EDA</h2>", unsafe_allow_html=True)
    
    df_flight = load_flight_data("data/processed/Cleaned_Processed_Flight_Price.csv")

    spacer,col1, spacer, col2 = st.columns([0.4,1, 0.2, 1])  # Adjusting widths (0.2 is the space)
    with col1:
        st.pyplot(plot_price_distribution(df_flight))
        st.pyplot(plot_route_vs_price(df_flight))

    with col2:
        st.pyplot(plot_airline_vs_price(df_flight))
        st.pyplot(plot_correlation_heatmap(df_flight))

# Passenger Satisfaction EDA Tab
with tab2:
    st.markdown("<h2 style='text-align: center;'>🛫 Passenger Satisfaction EDA</h2>", unsafe_allow_html=True)
    
    df_passenger = load_passenger_data("data/processed/Cleaned_Processed_Passenger_Satisfaction.csv")

    spacer,col1, spacer, col2 = st.columns([0.4,1, 0.2, 1])  # Adjusting widths (0.2 is the space)
    with col1:
        display_insights(df_passenger)

    with col2:
        st.write("### 📊 Distribution of Satisfaction Levels")
        st.pyplot(plot_satisfaction_distribution(df_passenger))

    # Load Model & Data for Confusion Matrix
    X_test, y_test, best_model, model_name = load_model_and_data()
    
    if X_test is not None and y_test is not None and best_model is not None:
        y_pred = best_model.predict(X_test)

        spacer,col3, spacer, col4 = st.columns([0.4,1, 0.2, 1])  # Adjusting widths (0.2 is the space)
        with col3:
            st.write("### 🔥 Feature Correlations with Satisfaction")
            st.pyplot(plot_feature_correlations(df_passenger))

        with col4:
            st.write(f"### 🤖 Confusion Matrix ({model_name})")
            st.pyplot(plot_confusion_matrix(y_test, y_pred, model_name))

# ✅ Success message AFTER all tabs (inside the Streamlit body)
#st.success("✅ EDA Analysis Completed!")
