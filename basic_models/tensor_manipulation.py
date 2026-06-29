import torch

# =====================================================================
# CONCEPT 1: SEQUENTIAL TENSOR GENERATION (torch.arange)
# =====================================================================
print("--- CONCEPT 1: GENERATING DATA ---")
raw_readings = torch.arange(12, dtype=torch.float32)
print(f"Original 1D Vector:\n{raw_readings}")
print(f"Shape: {raw_readings.shape}\n")

# =====================================================================
# CONCEPT 2: RESHAPING MATRIX DIMENSIONS (.view)
# =====================================================================
print("--- CONCEPT 2: RESHAPING TENSORS ---")
structured_matrix = raw_readings.view(4, 3)
print(f"Reshaped Matrix (4x3):\n{structured_matrix}")
print(f"New Shape: {structured_matrix.shape}\n")

# =====================================================================
# CONCEPT 3: TENSOR SLICING (Extracting Specific Sub-Data)
# =====================================================================
print("--- CONCEPT 3: SLICING AND DICING ---")
# 1. Grab only the very first row (Row Index 0)
first_row = structured_matrix[0, :]
print(f"First Row (All Columns): {first_row}")

# 2. Grab only the second column (Column Index 1)
second_column = structured_matrix[:, 1]
print(f"Second Column (All Rows): {second_column}")

# 3. Grab a specific sub-grid (Rows 1 & 2, Columns 0 & 1)
sub_grid = structured_matrix[1:3, 0:2]
print(f"Extracted Sub-Grid Matrix:\n{sub_grid}")