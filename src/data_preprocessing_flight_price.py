import pandas as pd
import numpy as np

def preprocess_flight_data(input_file, output_file):
    flight_df = pd.read_csv(input_file)
    
    flight_df["Date_of_Journey"] = pd.to_datetime(
        flight_df["Date_of_Journey"], format="%d/%m/%Y", errors="coerce"
    )
    
    if flight_df["Date_of_Journey"].isna().sum() > 0:
        print("\u26A0\uFE0F Warning: Some dates could not be parsed. Check your dataset!")
    
    flight_df = flight_df.assign(
        Journey_Day=flight_df["Date_of_Journey"].dt.day,
        Journey_Month=flight_df["Date_of_Journey"].dt.month
    ).drop(columns=["Date_of_Journey"])
    
    flight_df["Dep_Time"] = pd.to_datetime(flight_df["Dep_Time"], format="%H:%M", errors="coerce")
    flight_df = flight_df.assign(
        Dep_Hour=flight_df["Dep_Time"].dt.hour,
        Dep_Minute=flight_df["Dep_Time"].dt.minute
    ).drop(columns=["Dep_Time"])
    
    flight_df["Arrival_Time"] = pd.to_datetime(flight_df["Arrival_Time"], format="%H:%M", errors="coerce")
    flight_df = flight_df.assign(
        Arr_Hour=flight_df["Arrival_Time"].dt.hour,
        Arr_Minute=flight_df["Arrival_Time"].dt.minute
    ).drop(columns=["Arrival_Time"])
    
    def convert_duration(duration):
        try:
            parts = duration.split(" ")
            hours = int(parts[0].replace("h", "")) if "h" in parts[0] else 0
            minutes = int(parts[1].replace("m", "")) if len(parts) > 1 else 0
            return hours * 60 + minutes
        except Exception:
            return np.nan
    
    flight_df["Duration_Minutes"] = flight_df["Duration"].apply(convert_duration)
    flight_df = flight_df.drop(columns=["Duration"])
    
    flight_df = flight_df.ffill()
    
    flight_df.to_csv(output_file, index=False)
    print("\u2705 Flight Price Data Preprocessing Completed! Saved to:", output_file)

if __name__ == "__main__":
    preprocess_flight_data("data/Flight_Price.csv", "data/Cleaned_Flight_Price.csv")
