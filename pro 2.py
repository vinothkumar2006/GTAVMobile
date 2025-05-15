import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Simulated real-time data from sensors
def get_real_time_data():
    return {
        'temperature': np.random.uniform(18, 30),       # in °C
        'humidity': np.random.uniform(30, 70),          # in %
        'occupancy': np.random.randint(0, 2),           # 0 or 1
        'time_of_day': np.random.randint(0, 24)         # hour
    }

# Sample historical data for training
data = pd.DataFrame({
    'temperature': np.random.uniform(18, 30, 1000),
    'humidity': np.random.uniform(30, 70, 1000),
    'occupancy': np.random.randint(0, 2, 1000),
    'time_of_day': np.random.randint(0, 24, 1000),
    'energy_consumption': np.random.uniform(2, 10, 1000)  # kWh
})

# Train the model
X = data.drop('energy_consumption', axis=1)
y = data['energy_consumption']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'energy_model.pkl')

# Load and predict in real-time
model = joblib.load('energy_model.pkl')
real_time_data = pd.DataFrame([get_real_time_data()])
predicted_energy = model.predict(real_time_data)[0]

# Control logic (simplified)
if predicted_energy > 7.0:
    action = "Reduce HVAC load"
else:
    action = "Maintain current settings"

print(f"Predicted Energy Consumption: {predicted_energy:.2f} kWh")
print(f"Recommended Action: {action}")