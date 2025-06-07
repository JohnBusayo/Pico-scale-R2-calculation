import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Data from Tables 3.1–3.6
data = [
    # Distance 1.4 m
    [8, 10, 1.4, 1.4, 5.2], [8, 10, 1.53, 1.4, 2.94], [8, 10, 1.7, 1.4, 0.325],
    [8, 10, 2.0, 1.4, 0.001275], [8, 10, 2.3, 1.4, 0], [8, 10, 2.5, 1.4, 0],
    # Distance 1.8 m
    [8, 10, 1.4, 1.8, 6.47], [8, 10, 1.53, 1.8, 3.85574], [8, 10, 1.7, 1.8, 1.24537],
    [8, 10, 2.0, 1.8, 0.28144], [8, 10, 2.3, 1.8, 0], [8, 10, 2.5, 1.8, -0.0345],
    # Distance 2.0 m
    [8, 10, 1.4, 2.0, 6.68], [8, 10, 1.53, 2.0, 3.916], [8, 10, 1.7, 2.0, 1.28],
    [8, 10, 2.0, 2.0, 0.00225], [8, 10, 2.3, 2.0, 0], [8, 10, 2.5, 2.0, 0],
    # Distance 2.2 m
    [8, 10, 1.4, 2.2, 6.9], [8, 10, 1.53, 2.2, 4.69872], [8, 10, 1.7, 2.2, 1.571955],
    [8, 10, 2.0, 2.2, 0.070947], [8, 10, 2.3, 2.2, 0.042248], [8, 10, 2.5, 2.2, 0.026561],
    # Distance 2.4 m
    [8, 10, 1.4, 2.4, 7.35], [8, 10, 1.53, 2.4, 5.035], [8, 10, 1.7, 2.4, 1.8],
    [8, 10, 2.0, 2.4, 0.084], [8, 10, 2.3, 2.4, 0.04875], [8, 10, 2.5, 2.4, 0.0303],
    # Distance 2.6 m
    [8, 10, 1.4, 2.6, 7.81], [8, 10, 1.53, 2.6, 5.35], [8, 10, 1.7, 2.6, 2.0475],
    [8, 10, 2.0, 2.6, 0.098], [8, 10, 2.3, 2.6, 0.06], [8, 10, 2.5, 2.6, 0.0357]
]

# Prepare data
X = np.array([[row[0], row[1], row[2], row[3]] for row in data])  # Wind speed, AoA, height, distance
y = np.array([row[4] for row in data])  # Power output

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train linear regression model
model = LinearRegression()
model.fit(X_scaled, y)

# Predict and evaluate
y_pred = model.predict(X_scaled)
r2 = r2_score(y, y_pred)

# Print model details
print("Multiple Linear Regression Model for VAWT Power Output")
print("Coefficients (wind speed, AoA, height, distance):", model.coef_)
print("Intercept:", model.intercept_)
print("R² Score:", r2)


# Function to predict power output
def predict_power(wind_speed, aoa, height, distance):
    try:
        wind_speed = float(wind_speed)
        aoa = float(aoa)
        height = float(height)
        distance = float(distance)

        # Input validation with warnings for extrapolation
        if not (2 <= wind_speed <= 10):
            print("Warning: Wind speed outside simulation range (2–10 m/s). Prediction may be less accurate.")
        if not (0 <= aoa <= 25):
            print("Warning: AoA outside simulation range (0–25°). Prediction may be less accurate.")
        if not (1.4 <= height <= 2.5):
            print("Warning: Obstacle height outside simulation range (1.4–2.5 m). Prediction may be less accurate.")
        if not (1.4 <= distance <= 2.6):
            print("Warning: Distance outside simulation range (1.4–2.6 m). Prediction may be less accurate.")

        input_data = np.array([[wind_speed, aoa, height, distance]])
        input_scaled = scaler.transform(input_data)
        power = model.predict(input_scaled)[0]
        return power
    except ValueError:
        return "Error: Please enter valid numeric values."


# Interactive input loop
while True:
    print("\nEnter parameters to predict VAWT power output (or type 'exit' to quit):")
    wind_speed = input("Wind speed (m/s, e.g., 8): ")
    if wind_speed.lower() == 'exit':
        break
    aoa = input("Angle of attack (degrees, e.g., 10): ")
    height = input("Obstacle height (m, e.g., 1.7): ")
    distance = input("Horizontal distance (m, e.g., 4.0): ")

    power = predict_power(wind_speed, aoa, height, distance)
    if isinstance(power, str):
        print(power)
    else:
        print(f"Predicted power output: {power:.3f} W")
