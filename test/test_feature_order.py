import joblib

feature_order_path = "models/feature_order.pkl"
expected_features = joblib.load(feature_order_path)

print("✅ Expected Feature Order:", expected_features)
