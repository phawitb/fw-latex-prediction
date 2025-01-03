import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

# Load the dataset
file_path = 'RawDataAllModelG_All2024G.csv'
df = pd.read_csv(file_path)

# Drop unnecessary columns
df = df.drop(['mac_addr', 'drc_id', 'thai_date_created', 'date_created', 'probe', 'qty', 'optional_field3'], axis=1)

# Initialize the scaler
scaler = MinMaxScaler()
columns_to_scale = ['pressure_board_temp', 'pressure_env_port1', 'pressure_env_port2',
                    'pressure_env_port3', 'pressure_env_port4', 'latex_board_temp',
                    'latex_env_port1', 'latex_env_port2', 'latex_env_port3',
                    'latex_env_port4']

# Scale the specified columns
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Save the scaler to a file
scaler_path = 'scaler.joblib'
dump(scaler, scaler_path)

# Convert 'drc_percent_manual' to numeric and drop NaN rows
df['drc_percent_manual'] = pd.to_numeric(df['drc_percent_manual'], errors='coerce')
df = df.dropna(subset=['drc_percent_manual'])

# Convert the entire DataFrame to float
df = df.astype(float)

# Split the data into features and target
X = df.drop(columns=['drc_percent_manual'])
y = df['drc_percent_manual']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
model_path = 'rf_model.joblib'
dump(model, model_path)

print("Training complete. Model and scaler saved.")
