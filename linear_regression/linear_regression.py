import numpy as np
from sklearn.linear_model import LinearRegression

# Prepare the data
force = np.array([[10], [20], [30], [40], [50]])
stress = np.array([25, 52, 74, 101, 123])

# Initialize and train the AI model
model = LinearRegression()
model.fit(force, stress)

# Check the Accuracy Score
score = model.score(force, stress)

print(f"Model R^2 Score: {score * 100:.2f}%")
print(f"Predicted Stress for 35 N: {model.predict([[35]])[0]}")