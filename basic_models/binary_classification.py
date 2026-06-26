import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Features: [Surface Roughness (µm), Dimensional Deviation (mm)]
X = np.array([
    [0.8, 0.01],  # component 1 (Good)
    [1.2, 0.02],  # component 2 (Good)
    [3.5, 0.15],  # component 3 (Defective)
    [4.2, 0.20],  # component 4 (Defective)
    [0.9, 0.03],  # component 5 (Good)
    [3.8, 0.18]   # component 6 (Defective)
])

# 2. Target Labels: 0 = Pass, 1 = Fail
y = np.array([0, 0, 1, 1, 0, 1])

# 3. Initialize and train the Classifier
classifier = LogisticRegression()
classifier.fit(X, y)

# 4. Test a completely new manufacturing component
# Roughness = 1.0 µm, Deviation = 0.02 mm
new_component = np.array([[1.0, 0.02]])
prediction = classifier.predict(new_component)
probability = classifier.predict_proba(new_component)
model_score = classifier.score(X, y)

print(f"Prediction (0=Pass, 1=Fail): {prediction[0]}")
print(f"Confidence Levels [Prob of Pass, Prob of Fail]: {probability[0]}")
print(f"Model Accuracy: {model_score}")