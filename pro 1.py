import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Simulated dataset: features are temperature (°C), occupancy (people), and hour of day
np.random.seed(42)
data = pd.DataFrame({
    'temperature': np.random.uniform(18, 30, 100),
    'occupancy': np.random.randint(0, 50, 100),
    'hour': np.random.randint(0, 24, 100)
})

# Simulate energy consumption (target) with some noise
data['energy_consumption'] = (
    0.5 * data['temperature'] +
    2.0 * data['occupancy'] +
    0.1 * data['hour'] +
    np.random.normal(0, 2, 100)
)

# Features and target
X = data[['temperature', 'occupancy', 'hour']]
y = data['energy_consumption']

# Fit regression model
model = LinearRegression()
model.fit(X, y)

# Optimization: minimize energy consumption
def energy_function(x):
    return model.predict([x])[0]

# Constraints: temperature (20-26), occupancy (0-50), hour (0-23)
bounds = [(20, 26), (0, 50), (0, 23)]

# Initial guess
x0 = [22, 10, 12]

result = minimize(energy_function, x0, bounds=bounds)

print("Optimal settings:")
print(f"Temperature: {result.x[0]:.2f} °C")
print(f"Occupancy: {int(result.x[1])} people")
print(f"Hour: {int(result.x[2])}:00")
print(f"Predicted Energy Consumption: {energy_function(result.x):.2f} kWh")
