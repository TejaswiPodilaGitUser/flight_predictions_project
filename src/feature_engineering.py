import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder

def force_numeric_to_float(df):
    """Ensure all numeric columns are float to avoid MLflow warnings."""
    numeric_cols = df.select_dtypes(include=['int', 'int32', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df

def feature_engineering(input_file, output_file):
    # Load cleaned data
    df = pd.read_csv(input_file)

    # Identify categorical columns
    categorical_cols = ['Airline', 'Source', 'Destination', 'Total_Stops']

    # Target Encoding for categorical features (based on average price)
    target_encoder = TargetEncoder()
    df[categorical_cols] = target_encoder.fit_transform(df[categorical_cols], df['Price'])

    # Convert all numeric columns to float
    df = force_numeric_to_float(df)
    
    # Save processed data
    df.to_csv(output_file, index=False)
    print(f"✅ Feature engineering completed! Processed data saved to: {output_file}")

def process_passenger_satisfaction(input_file, output_file):
    df = pd.read_csv(input_file)

    categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

    # Label Encoding for categorical features
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Label Encoding for 'satisfaction' (Target column)
    df['satisfaction'] = LabelEncoder().fit_transform(df['satisfaction'])
    
    # Convert all numeric columns to float
    df = force_numeric_to_float(df)
    
    df.to_csv(output_file, index=False)
    print(f"✅ Passenger Satisfaction data processed and saved to: {output_file}")

# Run the feature engineering script
if __name__ == "__main__":
    feature_engineering("data/cleaned/Cleaned_Flight_Price.csv", "data/processed/Cleaned_Processed_Flight_Price.csv")
    process_passenger_satisfaction("data/cleaned/Cleaned_Passenger_Satisfaction.csv", "data/processed/Cleaned_Processed_Passenger_Satisfaction.csv")