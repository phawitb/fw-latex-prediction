from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
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
    pressure_board_temp: Optional[float] = 0
    pressure_env_port1: Optional[float] = 0
    pressure_env_port2: Optional[float] = 0
    pressure_env_port3: Optional[float] = 0
    pressure_env_port4: Optional[float] = 0
    latex_board_temp: Optional[float] = 0
    latex_env_port1: Optional[float] = 0
    latex_env_port2: Optional[float] = 0
    latex_env_port3: Optional[float] = 0
    latex_env_port4: Optional[float] = 0

class PredictionRequest(BaseModel):
    data: PredictionInput
    adj_error: Optional[AdjustmentError] = None

# Prediction function
@app.post("/predict")
def predict(request: PredictionRequest):
    # Convert request data into dictionaries
    data = request.data.dict()

    # Use provided adj_error or default to zeros
    if request.adj_error is None:
        adj_error = {key: 0 for key in data.keys()}
    else:
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
