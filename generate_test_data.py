import pandas as pd
import numpy as np
import os

# Ensure directory exists
os.makedirs('dataset', exist_ok=True)

# Generate synthetic data
# Scenario: 2 Input Features, 1 Target
# Logic: If (f1 + f2) > 10 => Class 1, else Class 0
np.random.seed(42)
n_samples = 100

f1 = np.random.uniform(0, 10, n_samples)
f2 = np.random.uniform(0, 10, n_samples)
y = (f1 + f2 > 10).astype(int)

df = pd.DataFrame({
    'feature_1': f1,
    'feature_2': f2,
    'y': y
})

# Save to CSV
file_path = os.path.join('dataset', 'test_data.csv')
df.to_csv(file_path, index=False)

print(f"Dataset generated at: {os.path.abspath(file_path)}")
print(df.head())
