import pandas as pd
from joblib import load

# Load the trained model and scaler
model_path = 'rf_model.joblib'
scaler_path = 'scaler.joblib'

model = load(model_path)
scaler = load(scaler_path)

# Input data for prediction
data = {
    "pressure_board_temp": 30.88,
    "pressure_env_port1": 25192.57,
    "pressure_env_port2": 57076.92,
    "pressure_env_port3": 32862.09,
    "pressure_env_port4": 40913.93,
    "latex_board_temp": 30.88,
    "latex_env_port1": 25072.32,
    "latex_env_port2": 56195.62,
    "latex_env_port3": 40984.01,
    "latex_env_port4": 31373.82 
  }

# Create a DataFrame from the input data
input_df = pd.DataFrame([data])

# Scale the input data using the loaded scaler
columns_to_scale = ['pressure_board_temp', 'pressure_env_port1', 'pressure_env_port2',
                    'pressure_env_port3', 'pressure_env_port4', 'latex_board_temp',
                    'latex_env_port1', 'latex_env_port2', 'latex_env_port3', 'latex_env_port4']
input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])

# Predict using the loaded model
predicted_value = model.predict(input_df)

# Output the prediction result
print(f"Predicted DRC Percent Manual: {predicted_value[0]}")
