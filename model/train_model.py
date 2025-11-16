import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# ---------------------------
# 1. Simulate / Generate Health Dataset
# ---------------------------
data = {
    'timestamp': pd.date_range(start='2025-11-01', periods=200, freq='H'),
    'heart_rate': np.random.randint(60, 100, 200),
    'blood_oxygen': np.random.randint(90, 100, 200),
    'activity_level': np.random.choice(['low','moderate','high'], 200),
    'sleep_hours': np.random.uniform(5, 8, 200)
}

df = pd.DataFrame(data)
df.to_csv('health_data.csv', index=False)
print("Simulated dataset created:")
print(df.head())

# ---------------------------
# 2. Isolation Forest for Anomaly Detection
# ---------------------------
# Load dataset (optional, here we already have df)
# df = pd.read_csv('health_data.csv')

# Select features
X = df[['heart_rate','blood_oxygen','sleep_hours']]

# Train model
model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(X)
df['anomaly'] = df['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

# Save results
df.to_csv('health_anomalies.csv', index=False)
print("Anomaly detection done. Results saved:")
print(df.head())
