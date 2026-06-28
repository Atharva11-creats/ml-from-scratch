import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =====================================================================
# SECTION 1: DATA PREPARATION & NORMALIZATION
# =====================================================================
print("--- SECTION 1: PREPARING MATRIX FOR NEURAL NETWORK ---", flush=True)
local_path = r"C:\Users\Atharva\Desktop\maching learning\ai4i2020.csv"
df = pd.read_csv(local_path)

# FIXED: Added the rename mapping back so Pandas can find the columns in the next step!
df = df.rename(columns={
    'Air temperature [K]': 'Air_Temperature_K',
    'Process temperature [K]': 'Process_Temperature_K',
    'Rotational speed [rpm]': 'Rotational_Speed_RPM',
    'Torque [Nm]': 'Torque_Nm',
    'Tool wear [min]': 'Tool_Wear_Min',
    'Machine failure': 'Machine_Failure'
})

# Isolate features and target label
feature_cols = ['Air_Temperature_K', 'Process_Temperature_K', 'Rotational_Speed_RPM', 'Torque_Nm', 'Tool_Wear_Min']
X = df[feature_cols].values
y = df['Machine_Failure'].values

# Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features to equalize variable magnitudes for the neural network
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert NumPy arrays into PyTorch Tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1) # Reshape to column vector
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
print("Data matrices cleaned, scaled, and converted to PyTorch Tensors successfully.\n", flush=True)

# =====================================================================
# SECTION 2: DEFINE THE BRAIN (NEURAL NETWORK ARCHITECTURE)
# =====================================================================
class PredictiveMaintenanceANN(nn.Module):
    def __init__(self, input_dim):
        super(PredictiveMaintenanceANN, self).__init__()
        
        # Layer 1: Input Layer (5 sensors) connected to Hidden Layer (16 neurons)
        self.hidden = nn.Linear(input_dim, 16)
        # Activation Function: Introduces non-linearity
        self.relu = nn.ReLU()
        # Layer 2: Hidden Layer (16 neurons) connected to 1 Output Node
        self.output = nn.Linear(16, 1)
        # Sigmoid forces output between 0.0 and 1.0 (Probability)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

# Instantiate the model
model = PredictiveMaintenanceANN(input_dim=5)
print("Neural Network structural framework successfully initialized.\n", flush=True)

# =====================================================================
# SECTION 3: DEFINE THE LOSS FUNCTION & OPTIMIZER
# =====================================================================
criterion = nn.BCELoss() # Binary Cross Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam optimization engine

# =====================================================================
# SECTION 4: THE DEEP LEARNING TRAINING LOOP (EPOCHS)
# =====================================================================
print("--- SECTION 4: TRAINING THE NEURAL NETWORK ---", flush=True)
epochs = 100 

for epoch in range(epochs):
    model.train()
    
    # 1. Forward Pass
    predictions = model(X_train_t)
    
    # 2. Compute Loss
    loss = criterion(predictions, y_train_t)
    
    # 3. Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print status every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] -> Loss (Error Metric): {loss.item():.4f}", flush=True)

# =====================================================================
# SECTION 5: FINAL EVALUATION
# =====================================================================
print("\n--- SECTION 5: TESTING THE BRAIN ON UNSEEN DATA ---", flush=True)
model.eval()
with torch.no_grad(): 
    test_preds = model(X_test_t)
    predicted_classes = (test_preds > 0.5).float()
    
    # Calculate accuracy metrics
    correct = (predicted_classes == y_test_t).sum().item()
    total = y_test_t.size(0)
    accuracy = (correct / total) * 100
    print(f"Neural Network Final Test Accuracy: {accuracy:.2f}%", flush=True)