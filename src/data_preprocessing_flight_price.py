import pandas as pd
import numpy as np

def preprocess_flight_data(input_file, output_file):
    flight_df = pd.read_csv(input_file)

    # Convert Date_of_Journey
    flight_df["Date_of_Journey"] = pd.to_datetime(
        flight_df["Date_of_Journey"], format="%d/%m/%Y", errors="coerce"
    )
    if flight_df["Date_of_Journey"].isna().sum() > 0:
        print("\u26A0\uFE0F Warning: Some dates could not be parsed. Check your dataset!")

    flight_df = flight_df.assign(
        Journey_Day=flight_df["Date_of_Journey"].dt.day,
        Journey_Month=flight_df["Date_of_Journey"].dt.month
    ).drop(columns=["Date_of_Journey"])

    # Convert Dep_Time
    flight_df["Dep_Time"] = pd.to_datetime(flight_df["Dep_Time"], format="%H:%M", errors="coerce").fillna("00:00")
    flight_df = flight_df.assign(
        Dep_Hour=flight_df["Dep_Time"].dt.hour,
        Dep_Minute=flight_df["Dep_Time"].dt.minute
    ).drop(columns=["Dep_Time"])

    # Convert Arrival_Time (handling cases with extra date info)
    flight_df["Arrival_Time"] = flight_df["Arrival_Time"].str.split(" ").str[0]  # Keep only time part
    flight_df["Arrival_Time"] = pd.to_datetime(flight_df["Arrival_Time"], format="%H:%M", errors="coerce").fillna("00:00")
    flight_df = flight_df.assign(
        Arr_Hour=flight_df["Arrival_Time"].dt.hour,
        Arr_Minute=flight_df["Arrival_Time"].dt.minute
    ).drop(columns=["Arrival_Time"])

    # Convert Duration to Minutes
    def convert_duration(duration):
        try:
            parts = duration.split()
            hours = int(parts[0].replace("h", "")) if "h" in parts[0] else 0
            minutes = int(parts[1].replace("m", "")) if len(parts) > 1 and "m" in parts[1] else 0
            return hours * 60 + minutes
        except Exception:
            return np.nan

    flight_df["Duration_Minutes"] = flight_df["Duration"].apply(convert_duration)
    flight_df = flight_df.drop(columns=["Duration"])

    # Convert Total_Stops to Numeric and Handle NaN
    flight_df["Total_Stops"] = (
        flight_df["Total_Stops"]
        .replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4})
        .infer_objects(copy=False)  # Explicitly set the old behavior to remove warning
        .fillna(0)
        .astype(int)
    )

    # Fill Missing Values
    flight_df = flight_df.ffill()  # Fix warning by using ffill() instead of fillna(method="ffill")

    # Save to CSV
    flight_df.to_csv(output_file, index=False)
    print("\u2705 Flight Price Data Preprocessing Completed! Saved to:", output_file)

if __name__ == "__main__":
    preprocess_flight_data("data/raw/Flight_Price.csv", "data/cleaned/Cleaned_Flight_Price.csv")
