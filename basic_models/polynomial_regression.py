import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# =====================================================================
# 1. THE DATA
# =====================================================================
# Input Feature: Fluid Velocity (m/s)
X = np.array([[1], [2], [3], [4], [5]])
# Target: Aerodynamic Drag Force (N)
y = np.array([2.1, 7.9, 18.2, 32.4, 49.8])

# =====================================================================
# 2. THEORETICAL DATA DATA (Terminal Prints)
# =====================================================================
print("--- THEORETICAL DATA INPUTS ---", flush=True)
print("Original X (Single column of inputs):")
print(X, flush=True)

# Transform input features to include x^2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print("\nTransformed X_poly Matrix [x, x^2] (The AI's two dials):")
print(X_poly)
print("-" * 35)

# =====================================================================
# 3. TRAINING THE MODEL
# =====================================================================
model = LinearRegression()
model.fit(X_poly, y)

# Retrieve calculated weights (m1, m2) and intercept (c)
m1 = model.coef_[0]
m2 = model.coef_[1]
c = model.intercept_

print("\n--- CALCULATED AI MODEL EQUATION ---")
print(f"Equation: Drag = ({m2:.2f} * v^2) + ({m1:.2f} * v) + {c:.2f}")
print(f"Model R^2 Accuracy Score: {model.score(X_poly, y) * 100:.2f}%")
print("-" * 35)

# Predict drag for a new velocity (e.g., 3.5 m/s)
new_velocity = np.array([[3.5]])
new_velocity_poly = poly.transform(new_velocity)
prediction = model.predict(new_velocity_poly)

print(f"\nPrediction for 3.5 m/s: {prediction[0]:.2f} N\n")

# =====================================================================
# 4. VISUAL GRAPH GENERATION
# =====================================================================
# Generate 100 smooth points between 1 and 5 to draw the parabolic curve
X_smooth = np.linspace(1, 5, 100).reshape(-1, 1)
X_smooth_poly = poly.transform(X_smooth)
y_smooth_pred = model.predict(X_smooth_poly)

# Plot experimental points
plt.scatter(X, y, color='red', s=100, label='Experimental Data Points', zorder=5)

# Plot the smooth polynomial regression curve
plt.plot(X_smooth, y_smooth_pred, color='blue', linewidth=2.5, label='AI Polynomial Fit ($y = m_1x + m_2x^2 + c$)')

# Labeling the graph window
plt.title('Fluid Velocity vs Aerodynamic Drag (Polynomial Regression)', fontsize=12, fontweight='bold')
plt.xlabel('Velocity (m/s)', fontsize=10)
plt.ylabel('Drag Force (N)', fontsize=10)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)

# Display the plot window
plt.show()