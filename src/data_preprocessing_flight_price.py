import pandas as pd
import numpy as np
import logging
import re

# Setup logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def parse_date(date_str):
    """Parse inconsistent date formats using regex."""
    try:
        if re.match(r"^\d{2}/\d{2}/\d{2}$", date_str):  # Format: DD/MM/YY
            return pd.to_datetime(date_str, format="%d/%m/%y")
        elif re.match(r"^\d{2}-\d{2}-\d{4}$", date_str):  # Format: DD-MM-YYYY
            return pd.to_datetime(date_str, format="%d-%m-%Y")
        elif re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):  # Format: YYYY-MM-DD
            return pd.to_datetime(date_str, format="%Y-%m-%d")
        else:
            return pd.NaT  # Return NaT for unrecognized formats
    except Exception:
        return pd.NaT

def preprocess_flight_data(input_file, output_file):
    logging.info("✅ Loading flight dataset...")
    flight_df = pd.read_csv(input_file)

    # ✅ Convert Date_of_Journey (handling mixed formats)
    flight_df["Date_of_Journey"] = flight_df["Date_of_Journey"].astype(str).apply(parse_date)

    # Log issues if any
    if flight_df["Date_of_Journey"].isna().sum() > 0:
        logging.warning("⚠️ Some dates could not be parsed. Rows affected:\n%s", 
                        flight_df["Date_of_Journey"][flight_df["Date_of_Journey"].isna()])

    # ✅ Extract Journey Day & Month
    flight_df = flight_df.assign(
        Journey_Day=flight_df["Date_of_Journey"].dt.day,
        Journey_Month=flight_df["Date_of_Journey"].dt.month
    ).drop(columns=["Date_of_Journey"])

    # ✅ Convert Dep_Time
    flight_df["Dep_Time"] = pd.to_datetime(flight_df["Dep_Time"], format="%H:%M", errors="coerce")
    flight_df = flight_df.assign(
        Dep_Hour=flight_df["Dep_Time"].dt.hour,
        Dep_Minute=flight_df["Dep_Time"].dt.minute
    ).drop(columns=["Dep_Time"])

    # ✅ Convert Arrival_Time (Handles mixed formats)
    flight_df["Arrival_Time"] = flight_df["Arrival_Time"].astype(str).str.split().str[-1]  # Extract time part
    flight_df["Arrival_Time"] = pd.to_datetime(flight_df["Arrival_Time"], format="%H:%M", errors="coerce")

    # ✅ Extract Arrival Hour & Minute
    flight_df = flight_df.assign(
        Arr_Hour=flight_df["Arrival_Time"].dt.hour,
        Arr_Minute=flight_df["Arrival_Time"].dt.minute
    ).drop(columns=["Arrival_Time"])

    # ✅ Convert Duration to Minutes
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

    # ✅ Convert Total_Stops to Numeric
    stop_mapping = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}
    flight_df["Total_Stops"] = flight_df["Total_Stops"].map(stop_mapping).astype(pd.Int64Dtype())


    # ✅ Fill Missing Values Safely
    flight_df.ffill(inplace=True)

    # ✅ Save Processed Data
    flight_df.to_csv(output_file, index=False)
    logging.info("✅ Flight Price Data Preprocessing Completed! Saved to: %s", output_file)

if __name__ == "__main__":
    preprocess_flight_data("data/raw/Flight_Price.csv", "data/cleaned/Cleaned_Flight_Price.csv")
