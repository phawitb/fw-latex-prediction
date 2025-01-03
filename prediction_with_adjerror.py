import pandas as pd
from joblib import load

def predict_drc_with_adjustment(data, adj_error, model, scaler):

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([data])

    # Apply adjustment error to the input data
    for column, adjustment in adj_error.items():
        if column in input_df.columns:
            input_df[column] += adjustment

    # Define the columns to scale
    columns_to_scale = ['pressure_board_temp', 'pressure_env_port1', 'pressure_env_port2',
                        'pressure_env_port3', 'pressure_env_port4', 'latex_board_temp',
                        'latex_env_port1', 'latex_env_port2', 'latex_env_port3', 'latex_env_port4']

    # Scale the input data using the loaded scaler
    input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])

    # Predict using the loaded model
    predicted_value = model.predict(input_df)

    # Return the prediction result
    return predicted_value[0]

model_path='rf_model.joblib'
scaler_path='scaler.joblib'
model = load(model_path)
scaler = load(scaler_path)

# Input data for prediction
input_data = {
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

# Adjustment errors for features
adjustment_error = {
    'pressure_board_temp': 0,
    'pressure_env_port1': -0.06,
    'pressure_env_port2': 0,
    'pressure_env_port3': 0.02,
    'pressure_env_port4': 0,
    'latex_board_temp': 0,
    'latex_env_port1': 0,
    'latex_env_port2': -0.1,
    'latex_env_port3': 0,
    'latex_env_port4': 0.3
}

# Predict and print the result
predicted_value = predict_drc_with_adjustment(input_data, adjustment_error, model, scaler)
print(f"Predicted DRC Percent Manual: {predicted_value}")