import pandas as pd
import numpy as np
import os

# Ensure directory exists
os.makedirs('dataset', exist_ok=True)

# Generate Complex Synthetic Data
# Scenario: 10 Input Features, 3 Classes
# Complexity: Non-linear combination of features
np.random.seed(99)
n_samples = 1000

# 10 random features
X = np.random.randn(n_samples, 10)

# Target logic:
# Class 0: sum(squares of first 3 features) < 1
# Class 1: sum(squares of first 3 features) between 1 and 3
# Class 2: sum(squares of first 3 features) > 3
# (Just a concentric circle pattern in high dimension)
r_squared = np.sum(X[:, :3]**2, axis=1)

y = np.zeros(n_samples, dtype=int)
y[r_squared >= 1] = 1
y[r_squared >= 3] = 2

cols = [f'feature_{i}' for i in range(10)]
df = pd.DataFrame(X, columns=cols)
df['y'] = y

# Save to CSV
file_path = os.path.join('dataset', 'complex_data.csv')
df.to_csv(file_path, index=False)

print(f"Dataset generated at: {os.path.abspath(file_path)}")
print(f"Shape: {df.shape}")
print(df['y'].value_counts())
