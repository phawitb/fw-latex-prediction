"""
RUN : uvicorn api_prediction_with_adjerror:app --reload --port 8000
POST : http://127.0.0.1:8000/predict
{
  "data":{
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
  },
  "adj_error": {
    "pressure_board_temp": 0,
    "pressure_env_port1": 0,
    "pressure_env_port2": 0,
    "pressure_env_port3": 0,
    "pressure_env_port4": 0,
    "latex_board_temp": 0,
    "latex_env_port1": 0,
    "latex_env_port2": 0,
    "latex_env_port3": 0,
    "latex_env_port4": 0
  }
}


RETURN :: 
{
    "predicted_drc_percent_manual": 33.55199999999991
}
"""

from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the model and scaler
model_path = 'rf_model.joblib'
scaler_path = 'scaler.joblib'
model = load(model_path)
scaler = load(scaler_path)

# Define input schema using Pydantic
class PredictionInput(BaseModel):
    pressure_board_temp: float
    pressure_env_port1: float
    pressure_env_port2: float
    pressure_env_port3: float
    pressure_env_port4: float
    latex_board_temp: float
    latex_env_port1: float
    latex_env_port2: float
    latex_env_port3: float
    latex_env_port4: float

class AdjustmentError(BaseModel):
    pressure_board_temp: float = 0
    pressure_env_port1: float = 0
    pressure_env_port2: float = 0
    pressure_env_port3: float = 0
    pressure_env_port4: float = 0
    latex_board_temp: float = 0
    latex_env_port1: float = 0
    latex_env_port2: float = 0
    latex_env_port3: float = 0
    latex_env_port4: float = 0

class PredictionRequest(BaseModel):
    data: PredictionInput
    adj_error: AdjustmentError

# Prediction function
@app.post("/predict")
def predict(request: PredictionRequest):
    # Convert request data into dictionaries
    data = request.data.dict()
    adj_error = request.adj_error.dict()

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
    return {"predicted_drc_percent_manual": predicted_value[0]}
