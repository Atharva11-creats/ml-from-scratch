import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================================
# SECTION 1: LOAD LOCAL INDUSTRIAL DATASET
# =====================================================================
# Define the absolute local directory path to your downloaded file
local_path = r"C:\Users\Atharva\Desktop\maching learning\ai4i2020.csv"
print("--- SECTION 1: LOADING DATA ---", flush=True)
print(f"Reading local dataset from: {local_path}...", flush=True)

df = pd.read_csv(local_path)
print(f"Dataset successfully loaded into memory.", flush=True)
print(f"Total Rows (Observations): {df.shape[0]}")
print(f"Total Columns (Features/Labels): {df.shape[1]}\n")

# =====================================================================
# SECTION 2: DATA CLEANING & RENAME MAPPING
# =====================================================================
print("--- SECTION 2: CLEANING TELEMETRY HEADER MATRICES ---", flush=True)
# The raw dataset contains brackets and units which make parsing difficult.
# We map them to clean, pythonic attribute names.
df = df.rename(columns={
    'Air temperature [K]': 'Air_Temperature_K',
    'Process temperature [K]': 'Process_Temperature_K',
    'Rotational speed [rpm]': 'Rotational_Speed_RPM',
    'Torque [Nm]': 'Torque_Nm',
    'Tool wear [min]': 'Tool_Wear_Min',
    'Machine failure': 'Machine_Failure'
})
print("Headers mapped successfully.\n", flush=True)

# =====================================================================
# SECTION 3: FEATURE MATRIX (X) AND TARGET LABEL (y) SEPARATION
# =====================================================================
print("--- SECTION 3: ISOLATING FEATURES AND LABELS ---", flush=True)
# We isolate the columns representing physical sensor readings (Features).
# We exclude row metadata identifiers like 'UDI', 'Product ID', and 'Type'.
feature_cols = ['Air_Temperature_K', 'Process_Temperature_K', 'Rotational_Speed_RPM', 'Torque_Nm', 'Tool_Wear_Min']
X = df[feature_cols]

# 'Machine_Failure' is our target label (0 = Operational Normal, 1 = Failure)
y = df['Machine_Failure']
print(f"Features dimension (X): {X.shape}")
print(f"Target dimension (y): {y.shape}\n")

# =====================================================================
# SECTION 4: TRAIN-TEST SPLIT CONFIGURATION
# =====================================================================
print("--- SECTION 4: PARTITIONING DATASETS ---", flush=True)
# We reserve 20% of the dataset to test how smart the AI is on completely unseen data.
# 'stratify=y' ensures the split maintains identical failure ratios in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training matrices isolated: {X_train.shape[0]} rows.")
print(f"Testing matrices isolated: {X_test.shape[0]} rows.\n")

# =====================================================================
# SECTION 5: MODEL INITIALIZATION & TRAINING
# =====================================================================
print("--- SECTION 5: TRAINING RANDOM FOREST FORESTRY ---", flush=True)
# n_estimators=50 builds a committee of 50 independent decision trees.
# max_depth=10 limits the tree depth to prevent overfitting.
model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
model.fit(X_train, y_train)
print("Random Forest fit operations complete.\n", flush=True)

# =====================================================================
# SECTION 6: MODEL PERFORMANCE EVALUATION
# =====================================================================
print("--- SECTION 6: UNSEEN PERFORMANCE EVALUATION ---", flush=True)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)

print(f"Overall Testing Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Performance Class Matrix Breakdown:")
print(classification_report(y_test, y_pred))
print("-" * 50 + "\n")

# =====================================================================
# SECTION 7: FACTORY FLOOR SENSOR IMPORTANCE RANKING
# =====================================================================
print("--- SECTION 7: CRITICAL SENSOR INFLUENCE RANKING ---", flush=True)
# Extract the statistical influence of each sensor on predicting a failure
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X.shape[1]):
    print(f"{f + 1}. Sensor Array '{feature_cols[indices[f]]}' -> Weight: {importances[indices[f]]:.4f}")
print("-" * 50 + "\n")

# =====================================================================
# SECTION 8: LIVE PREDICTIVE MAINTENANCE TELEMETRY QUERY
# =====================================================================
print("--- SECTION 8: AUTOMATED PRODUCTION CALL PIPELINE ---", flush=True)
# Simulating a live incoming telemetry scan from a physical CNC machine floor node:
# Air Temp=302.1K, Process Temp=311.5K, Speed=1350RPM, Torque=55.2Nm, Tool Wear=210min
live_telemetry = pd.DataFrame([[302.1, 311.5, 1350, 55.2, 210]], columns=feature_cols)

prediction = model.predict(live_telemetry)
probabilities = model.predict_proba(live_telemetry)[0]

print(f"Live Input Readings: {live_telemetry.values[0]}")
if prediction[0] == 1:
    print(f"STATUS ALERT: CRITICAL FAULT PREDICTED (Confidence: {probabilities[1]*100:.2f}%)")
    print("ACTION SENT: Interrupting cell power sequence. Dispatching mechanical maintenance.")
else:
    print(f"STATUS REPORT: OPERATIONAL SYSTEM STABLE (Confidence: {probabilities[0]*100:.2f}%)")
print("----------------------------------------------------------------------")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================================
# SECTION 1: LOAD LOCAL INDUSTRIAL DATASET
# =====================================================================
local_path = r"C:\Users\Atharva\Desktop\maching learning\ai4i2020.csv"
print("--- SECTION 1: LOADING DATA ---", flush=True)
df = pd.read_csv(local_path)
print(f"Dataset successfully loaded. Shape: {df.shape[0]} rows\n", flush=True)

# =====================================================================
# SECTION 2: DATA CLEANING & RENAME MAPPING
# =====================================================================
df = df.rename(columns={
    'Air temperature [K]': 'Air_Temperature_K',
    'Process temperature [K]': 'Process_Temperature_K',
    'Rotational speed [rpm]': 'Rotational_Speed_RPM',
    'Torque [Nm]': 'Torque_Nm',
    'Tool wear [min]': 'Tool_Wear_Min',
    'Machine failure': 'Machine_Failure'
})

# =====================================================================
# SECTION 3: FEATURE MATRIX AND TARGET LABEL SEPARATION
# =====================================================================
feature_cols = ['Air_Temperature_K', 'Process_Temperature_K', 'Rotational_Speed_RPM', 'Torque_Nm', 'Tool_Wear_Min']
X = df[feature_cols]
y = df['Machine_Failure']

# =====================================================================
# SECTION 4: TRAIN-TEST SPLIT CONFIGURATION
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =====================================================================
# SECTION 5: MODEL INITIALIZATION & TRAINING
# =====================================================================
print("--- SECTION 5: TRAINING RANDOM FOREST FORESTRY ---", flush=True)
model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
model.fit(X_train, y_train)
print("Model training complete.\n", flush=True)

# =====================================================================
# SECTION 6: UNSEEN PERFORMANCE EVALUATION
# =====================================================================
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"Overall Testing Accuracy: {accuracy * 100:.2f}%\n")

# =====================================================================
# SECTION 7: GRAPHICAL VISUALIZATION (MATPLOTLIB)
# =====================================================================
print("--- SECTION 7: GENERATING GRAPH WINDOWS ---", flush=True)

# Create a window dashboard with 1 row and 2 columns for side-by-side plots
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Confusion Matrix Heatmap (Left Side)
cm = confusion_matrix(y_test, y_pred)

# Display the core heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax[0], cbar=False,
            xticklabels=['Healthy', 'Faulty'], yticklabels=['Healthy', 'Faulty'],
            annot_kws={"size": 13, "weight": "bold", "color": "black"})

# --- DRAWING THE DIAGONAL RECTANGLES / LINE ---
# This draws a dashed green box outline around the "True Healthy" and "True Faults" diagonal
# [x_start, x_end], [y_start, y_end]
ax[0].plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='green', linewidth=3, linestyle='--', label='Correct Path')
ax[0].plot([1, 2, 2, 1, 1], [1, 1, 2, 2, 1], color='green', linewidth=3, linestyle='--')

ax[0].set_title('Confusion Matrix: Predictions vs Reality', fontsize=12, fontweight='bold')
ax[0].set_xlabel('Predicted Label (AI Verdict)', fontweight='bold')
ax[0].set_ylabel('True Label (Actual Status)', fontweight='bold')

# Plot 2: Feature Importance Bar Chart (Right Side)
importances = model.feature_importances_
indices = np.argsort(importances)

ax[1].barh(range(len(indices)), importances[indices], color='darkred', align='center')
ax[1].set_yticks(range(len(indices)))
ax[1].set_yticklabels([feature_cols[i] for i in indices], fontweight='bold')
ax[1].set_title('Sensor Influence Ranking', fontsize=12, fontweight='bold')
ax[1].set_xlabel('Statistical Contribution Weight', fontweight='bold')

# Adjust layout structure and display window
plt.tight_layout()
print("Displaying dashboard window... (Close the window to return to terminal)", flush=True)
plt.show()