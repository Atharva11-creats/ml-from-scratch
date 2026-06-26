import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Prepare the data
force = np.array([[10], [20], [30], [40], [50]])
stress = np.array([25, 52, 74, 101, 123])

# 2. Initialize the AI model
model = LinearRegression()

# 3. Train the model (The Learning Phase)
model.fit(force, stress)

# 4. Use the model to predict
new_force = np.array([[2000]])
predicted_stress = model.predict(new_force)

model.score(force, stress)
model.score


print(f"Predicted Stress for 2000 N: {predicted_stress[0]}")
print(f"Model Equation: Stress = {model.coef_[0]} * Force + {model.intercept_}")


from sklearn.metrics import mean_absolute_error

# 1. Ask the model to predict values for our original data
predictions = model.predict(force)

# 2. Calculate the average error
mae = mean_absolute_error(stress, predictions)

# 3. Print the results
print(f"Actual Values:    {stress}")
print(f"Predicted Values: {predictions}")
print(f"Mean Absolute Error (MAE): {mae:.2f} MPa")

# Calculate the model's score
score = model.score(force, stress)

print(f"Model R^2 Score: {score:.4f}")
print(f"Accuracy in Percentage: {score * 100:.2f}%")
