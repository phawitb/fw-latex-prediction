import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from joblib import dump

def classify_drc(drc_value):
    """
    Classifies drc_percent_manual into categories based on value ranges.
    Args:
    - drc_value (float): The DRC percentage to classify.
    
    Returns:
    - str: The classification label (e.g., '20-25', '25-30', etc.).
    """
    if pd.isnull(drc_value):  # Handle NaN values
        return None
    
    # Define the range step and starting point
    start = 20
    step = 5
    
    # Calculate the class index and range
    class_index = (drc_value - start) // step
    if class_index < 0 or drc_value >= 50:  # Out-of-range values
        return None
    
    # Compute the range
    lower_bound = start + class_index * step
    upper_bound = lower_bound + step
    return f"{int(lower_bound)}-{int(upper_bound)}"

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

# Convert Y to range-based class labels
df['drc_class'] = df['drc_percent_manual'].apply(classify_drc)

X = df.drop(columns=['drc_percent_manual', 'drc_class'])
y = df['drc_class']

print(f'X::{X.shape}\n', X)
print(f'y::{y.shape}\n', y)

# Train Model
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Support Vector Machine": SVC(kernel='linear', random_state=42)
}

# Train and evaluate each model
best_model = None
best_model_name = ""
best_accuracy = 0.0

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"=== {name} ===")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))
    print("\n")
    
    # Update the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# Save the best model to a file
if best_model:
    print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy:.2f}")
    best_model_file = f"best_cls_{best_model_name.replace(' ', '_')}.joblib"
    dump(best_model, best_model_file)
    print(f"Model saved as '{best_model_file}'")
