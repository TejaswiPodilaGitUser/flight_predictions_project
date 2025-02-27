import pandas as pd

ARRIVAL_DELAY_COL = "Arrival Delay in Minutes"

def preprocess_passenger_data(input_file, output_file):
    passenger_df = pd.read_csv(input_file)

    # Convert Arrival Delay column to numeric & fill missing values
    if ARRIVAL_DELAY_COL in passenger_df.columns:
        passenger_df[ARRIVAL_DELAY_COL] = pd.to_numeric(passenger_df[ARRIVAL_DELAY_COL], errors='coerce')
        passenger_df[ARRIVAL_DELAY_COL] = passenger_df[ARRIVAL_DELAY_COL].fillna(passenger_df[ARRIVAL_DELAY_COL].median()).astype(int)

    # Standardizing categorical columns
    categorical_cols = ["Gender", "Customer Type", "Type of Travel", "Class", "satisfaction"]
    for col in categorical_cols:
        if col in passenger_df.columns:
            passenger_df[col] = passenger_df[col].astype(str).str.lower().str.strip().fillna("unknown")

    # Fill missing values for other columns
    passenger_df = passenger_df.ffill()  # Fixed FutureWarning

    # Save cleaned data
    passenger_df.to_csv(output_file, index=False)
    print(f"✅ Passenger Satisfaction Data Preprocessing Completed! Saved to: {output_file}")

if __name__ == "__main__":
    preprocess_passenger_data("data/raw/Passenger_Satisfaction.csv", "data/cleaned/Cleaned_Passenger_Satisfaction.csv")
