"""
RUN : uvicorn api_prediction_with_adjerror:app --reload --port 8000
POST : http://127.0.0.1:8000/predict
{
  "data": {
    "pressure_board_temp": 0.3647457627118642,
    "pressure_env_port1": 0.0699428216426412,
    "pressure_env_port2": 0.8623649553705768,
    "pressure_env_port3": 0.09191874664534061,
    "pressure_env_port4": 0.6000553779349063,
    "latex_board_temp": 0.35020519835841335,
    "latex_env_port1": 0.04297192330376909,
    "latex_env_port2": 0.707629518391446,
    "latex_env_port3": 0.4327944098408234,
    "latex_env_port4": 0.32820913650448624
  },
  "adj_error": {
    "pressure_board_temp": 0,
    "pressure_env_port1": -0.06,
    "pressure_env_port2": 0,
    "pressure_env_port3": 0.02,
    "pressure_env_port4": 0,
    "latex_board_temp": 0,
    "latex_env_port1": 0,
    "latex_env_port2": -0.1,
    "latex_env_port3": 0,
    "latex_env_port4": 0.3
  }
}

RETURN :: 
{
    "predicted_drc_percent_manual": 28.871199999999888
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
