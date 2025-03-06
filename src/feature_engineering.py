import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder

def force_numeric_to_float(df):
    """Ensure all numeric columns are float to avoid MLflow warnings."""
    numeric_cols = df.select_dtypes(include=['int', 'int32', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df

def feature_engineering(input_file, output_file):
    # Load cleaned data
    df = pd.read_csv(input_file)

    # Drop unnecessary columns if they exist (since some might have been handled in preprocessing)
    drop_cols = ['Route', 'Duration', 'Additional_Info']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Convert 'Total_Stops' to numerical format
    stop_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
    
    # Map the 'Total_Stops' values and fill NaN values with a default value (0 for "non-stop")
    df['Total_Stops'] = df['Total_Stops'].map(stop_mapping)
    df['Total_Stops'] = df['Total_Stops'].fillna(0)  # Fill missing values with 0 (non-stop)
    df['Total_Stops'] = df['Total_Stops'].astype(int)  # Now safe to convert to int

    # Label Encoding for 'Source' and 'Destination' to avoid multiple columns
    df['Source'] = df['Source'].astype('category').cat.codes
    df['Destination'] = df['Destination'].astype('category').cat.codes

    # Apply Target Encoding for 'Airline' column based on 'Price' (regression)
    target_encoder = TargetEncoder()
    df['Airline'] = target_encoder.fit_transform(df['Airline'], df['Price'])

    # Extract additional features (for example, journey duration or any additional calculations)
    # Create a new feature 'Journey_Hour' that represents the total travel time:
    if 'Dep_Hour' in df.columns and 'Arr_Hour' in df.columns:
        df['Journey_Hour'] = df['Arr_Hour'] - df['Dep_Hour']
        df['Journey_Hour'] = df['Journey_Hour'].clip(lower=0)  # Ensure no negative values

    # Convert all numeric columns to float for consistency
    df = force_numeric_to_float(df)

    # Save processed data
    df.to_csv(output_file, index=False)
    print(f"✅ Feature engineering completed! Processed data saved to: {output_file}")

# Run the feature engineering script
if __name__ == "__main__":
    feature_engineering("data/cleaned/Cleaned_Flight_Price.csv", "data/processed/Cleaned_Processed_Flight_Price.csv")
