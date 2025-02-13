import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from category_encoders import TargetEncoder

def feature_engineering(input_file, output_file):
    # Load cleaned data
    df = pd.read_csv(input_file)

    # Identify categorical and numerical columns
    categorical_cols = ['Airline', 'Source', 'Destination', 'Total_Stops']
    numerical_cols = ['Journey_Day', 'Journey_Month', 'Dep_Hour', 'Dep_Minute', 
                      'Arr_Hour', 'Arr_Minute', 'Duration_Minutes']

    # Target Encoding for categorical features (based on average price)
    target_encoder = TargetEncoder()
    df[categorical_cols] = target_encoder.fit_transform(df[categorical_cols], df['Price'])

    # Robust Scaling for numerical features (handles outliers better than StandardScaler)
    scaler = RobustScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Save processed data
    df.to_csv(output_file, index=False)
    print(f"✅ Feature engineering completed! Processed data saved to: {output_file}")


def process_passenger_satisfaction(input_file, output_file):
    df = pd.read_csv(input_file)
    categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    numerical_cols = ['Age', 'Flight Distance', 'Inflight wifi service', 'Seat comfort', 'Inflight entertainment']
    
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])
    
    scaler = RobustScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    df.to_csv(output_file, index=False)
    print(f"✅ Passenger Satisfaction data processed and saved to: {output_file}")   

# Run the feature engineering script
if __name__ == "__main__":
    feature_engineering("data/Cleaned_Flight_Price.csv", "data/Cleaned_Processed_Flight_Price.csv")
    process_passenger_satisfaction("data/Cleaned_Passenger_Satisfaction.csv", "data/Cleaned_Processed_Passenger_Satisfaction.csv")

