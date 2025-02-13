import joblib

# Define feature order (must match the model's training data)
feature_order = [
    "Airline", "Source", "Destination", "Dep_Hour", "Dep_Minute",
    "Arr_Hour", "Arr_Minute", "Duration_Minutes", "Total_Stops",
    "Additional_Info", "Journey_Day", "Journey_Month", "Route"
]

# Save feature order
joblib.dump(feature_order, "models/feature_order.pkl")
print("✅ feature_order.pkl saved successfully.")
