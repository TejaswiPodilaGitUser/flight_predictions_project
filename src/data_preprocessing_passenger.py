import pandas as pd

ARRIVAL_DELAY_COL = "Arrival Delay in Minutes"
GENDER_COL = "Gender"
SATISFACTION_COL = "satisfaction"

def preprocess_passenger_data(input_file, output_file):
    passenger_df = pd.read_csv(input_file)

    # Convert Arrival Delay column to numeric & fill missing values
    if ARRIVAL_DELAY_COL in passenger_df.columns:
        passenger_df[ARRIVAL_DELAY_COL] = pd.to_numeric(passenger_df[ARRIVAL_DELAY_COL], errors='coerce')
        # Median imputation for skewed numerical columns like 'Arrival Delay'
        passenger_df[ARRIVAL_DELAY_COL] = passenger_df[ARRIVAL_DELAY_COL].fillna(passenger_df[ARRIVAL_DELAY_COL].median()).astype(int)

    # Standardizing categorical columns
    categorical_cols = ["Gender", "Customer Type", "Type of Travel", "Class", SATISFACTION_COL]
    for col in categorical_cols:
        if col in passenger_df.columns:
            passenger_df[col] = passenger_df[col].astype(str).str.lower().str.strip().fillna("unknown")

    # Handle Gender column specifically (Use Mode or Predictive Model)
    if GENDER_COL in passenger_df.columns:
        # If 'Gender' has a skewed distribution (e.g., male > female > other), fill with a more balanced approach.
        # If missing, use most frequent value within similar rows or perform a random sample based on known distributions
        mode_gender = passenger_df[GENDER_COL].mode()[0]  # This is just for simplicity; could be more sophisticated
        passenger_df[GENDER_COL] = passenger_df[GENDER_COL].fillna(mode_gender)

    # Fill missing values for numeric columns
    numeric_cols = passenger_df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if col != ARRIVAL_DELAY_COL:  # We have already handled 'Arrival Delay'
            passenger_df[col] = passenger_df[col].fillna(passenger_df[col].median())

    # Convert all numeric columns to float64 to avoid MLflow integer schema warnings
    passenger_df[numeric_cols] = passenger_df[numeric_cols].astype(float)

    # Drop unnecessary columns
    drop_cols = ["Unnamed: 0", "id"]
    passenger_df = passenger_df.drop(columns=[col for col in drop_cols if col in passenger_df.columns])

    # Save cleaned data
    passenger_df.to_csv(output_file, index=False)
    print(f"✅ Passenger Satisfaction Data Preprocessing Completed! Saved to: {output_file}")

if __name__ == "__main__":
    preprocess_passenger_data("data/raw/Passenger_Satisfaction.csv", "data/cleaned/Cleaned_Passenger_Satisfaction.csv")
