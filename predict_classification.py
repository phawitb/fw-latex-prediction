import numpy as np
import joblib

# Load the best model
model_path = 'best_cls_Random_Forest.joblib'  # Specify the exact model file
loaded_model = joblib.load(model_path)

# Load the scaler
scaler_path = 'scaler.joblib'  # Specify the scaler file
scaler = joblib.load(scaler_path)

# Example test data
# Replace these values with actual raw feature values
example_data = {
    'pressure_board_temp': 30,  # Example raw value
    'pressure_env_port1': 40,
    'pressure_env_port2': 35,
    'pressure_env_port3': 33,
    'pressure_env_port4': 28,
    'latex_board_temp': 29,
    'latex_env_port1': 32,
    'latex_env_port2': 31,
    'latex_env_port3': 30,
    'latex_env_port4': 33
}

# Convert the dictionary to a NumPy array
example_data_array = np.array([list(example_data.values())])

# Scale the input data using the saved scaler
example_data_scaled = scaler.transform(example_data_array)

# Make a prediction
prediction = loaded_model.predict(example_data_scaled)
print(f"The predicted class for the input data is: {prediction[0]}")
