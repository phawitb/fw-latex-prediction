import pandas as pd
import json

def try_convert_to_float(cell):
    try:
        return float(cell)
    except ValueError:
        return cell  # Keep the original value if conversion fails

def find_average(df_filtered,probe,pressure_type):
  features = ['pressure_board_temp', 'pressure_env_port1','pressure_env_port2', 'pressure_env_port3', 'pressure_env_port4','latex_board_temp', 'latex_env_port1', 'latex_env_port2','latex_env_port3', 'latex_env_port4']
  
  df_filtered = df_filtered[df_filtered['probe']==probe]
  df_filtered = df_filtered[df_filtered['drc_percent_manual'].str.contains(pressure_type, na=False)]
  means = df_filtered[features].mean()

  return dict(means)

file_path = 'RawDataAllModelG_All2024G.csv'
df = pd.read_csv(file_path)

df = df.applymap(try_convert_to_float)
df_filtered = df[df['drc_percent_manual'].apply(lambda x: isinstance(x, str))]
df_filtered.to_csv('filtered_data.csv', index=False)

k = ['AIR','WATER','FINGER']
standard_values = {}
for i in k:
  standard_values[i] = find_average(df_filtered,'P1',i)

file_path = "standard_values.json"
with open(file_path, "w") as json_file:
    json.dump(standard_values, json_file, indent=4)