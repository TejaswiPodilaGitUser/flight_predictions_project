import pandas as pd

ARRIVAL_DELAY_COL = "Arrival Delay in Minutes"

def preprocess_passenger_data(input_file, output_file):
    passenger_df = pd.read_csv(input_file, index_col=0)
    
    passenger_df[ARRIVAL_DELAY_COL] = passenger_df[ARRIVAL_DELAY_COL].fillna(
        passenger_df[ARRIVAL_DELAY_COL].median()
    )

    categorical_cols = ["Gender", "Customer Type", "Type of Travel", "Class", "satisfaction"]
    for col in categorical_cols:
        passenger_df[col] = passenger_df[col].str.lower().str.strip()

    passenger_df.to_csv(output_file, index=False)
    print(f"✅ Preprocessing complete! Cleaned data saved to: {output_file}")

if __name__ == "__main__":
    preprocess_passenger_data("data/Passenger_Satisfaction.csv", "data/Cleaned_Passenger_Satisfaction.csv")
