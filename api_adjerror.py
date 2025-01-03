"""
RUN :: uvicorn api_adjerror:app --reload --port 8081
POST :: http://127.0.0.1:8081/adjerror

{
    "WATER": {
        "pressure_board_temp": 31.350350877192984,
        "pressure_env_port1": 25729.862807017547,
        "pressure_env_port2": 57352.156754385964,
        "pressure_env_port3": 33642.73263157895,
        "pressure_env_port4": 40109.32131578948,
        "latex_board_temp": 31.37456140350877,
        "latex_env_port1": 25674.897456140352,
        "latex_env_port2": 53939.83640350878,
        "latex_env_port3": 52826.26859649124,
        "latex_env_port4": 21394.58605263158
    },
    "FINGER": {
        "pressure_board_temp": 30.934347826086956,
        "pressure_env_port1": 25727.433391304352,
        "pressure_env_port2": 57346.38652173913,
        "pressure_env_port3": 33611.20939130434,
        "pressure_env_port4": 40123.82217391304,
        "latex_board_temp": 30.99495652173913,
        "latex_env_port1": 25729.402782608697,
        "latex_env_port2": 57349.95660869568,
        "latex_env_port3": 33618.69208695652,
        "latex_env_port4": 40121.650347826086
    },
    "AIR": {
        "pressure_board_temp": 31.20275862068965,
        "pressure_env_port1": 25727.433275862066,
        "pressure_env_port2": 57350.09034482758,
        "pressure_env_port3": 33621.22094827586,
        "pressure_env_port4": 40130.29172413793,
        "latex_board_temp": 31.22198275862069,
        "latex_env_port1": 25728.993017241384,
        "latex_env_port2": 55334.77456896553,
        "latex_env_port3": 44296.64336206897,
        "latex_env_port4": 29169.05172413793
    }
}


"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import json
import numpy as np

# FastAPI instance
app = FastAPI()

# Define Pydantic models for request/response validation
class Values(BaseModel):
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

class CurrentValues(BaseModel):
    WATER: Values
    FINGER: Values
    AIR: Values

# Function to calculate differences
def find_different(current_values: Dict[str, Dict[str, float]], standard_values: Dict[str, Dict[str, float]]):
    differences = {}
    for key, current in current_values.items():
        standard = standard_values.get(key, {})
        differences[key] = {
            param: current[param] - standard.get(param, 0)
            for param in current.keys()
        }
    return differences

# Endpoint to receive current values and return differences
@app.post("/adjerror")
async def calculate_differences(current_values: CurrentValues):
    # Load standard values (you can change this to load from a file or DB)
    with open("standard_values.json", "r") as json_file:
        standard_values = json.load(json_file)

    # Calculate and return differences
    differences = find_different(current_values.dict(), standard_values)

    # Calculate the average of each feature
    averages = {}
    for feature in differences["WATER"]:
        feature_values = [differences[category][feature] for category in differences]
        averages[feature] = np.mean(feature_values)

    return {"adj_error": averages,"differences": differences}






# import json

# def find_different(current_values,standard_values):
#   # Compute the differences
#   differences = {}
#   for key, current in current_values.items():
#       standard = standard_values.get(key, {})
#       differences[key] = {
#           param: current[param] - standard.get(param, 0)
#           for param in current.keys()
#       }

#   return differences

# file_path = "standard_values.json"
# with open(file_path, "r") as json_file:
#   standard_values = json.load(json_file)

# find_different(current_values,standard_values)



# standard_values = {
#         {'AIR': {'pressure_board_temp': 30.934347826086956,
#             'pressure_env_port1': 25727.433391304352,
#             'pressure_env_port2': 57346.38652173913,
#             'pressure_env_port3': 33611.20939130434,
#             'pressure_env_port4': 40123.82217391304,
#             'latex_board_temp': 30.99495652173913,
#             'latex_env_port1': 25729.402782608697,
#             'latex_env_port2': 57349.95660869568,
#             'latex_env_port3': 33618.69208695652,
#             'latex_env_port4': 40121.650347826086},
#         'WATER': {'pressure_board_temp': 31.350350877192984,
#             'pressure_env_port1': 25729.862807017547,
#             'pressure_env_port2': 57352.156754385964,
#             'pressure_env_port3': 33642.73263157895,
#             'pressure_env_port4': 40109.32131578948,
#             'latex_board_temp': 31.37456140350877,
#             'latex_env_port1': 25674.897456140352,
#             'latex_env_port2': 53939.83640350878,
#             'latex_env_port3': 52826.26859649124,
#             'latex_env_port4': 21394.58605263158},
#         'FINGER': {'pressure_board_temp': 31.20275862068965,
#             'pressure_env_port1': 25727.433275862066,
#             'pressure_env_port2': 57350.09034482758,
#             'pressure_env_port3': 33621.22094827586,
#             'pressure_env_port4': 40130.29172413793,
#             'latex_board_temp': 31.22198275862069,
#             'latex_env_port1': 25728.993017241384,
#             'latex_env_port2': 55334.77456896553,
#             'latex_env_port3': 44296.64336206897,
#             'latex_env_port4': 29169.05172413793}}
#     },