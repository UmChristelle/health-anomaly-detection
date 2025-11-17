import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# --------------------------
# Step 1: Generate new simulated health data
# --------------------------
data = {
    'timestamp': pd.date_range(start='2025-11-17', periods=200, freq='H'),
    'heart_rate': np.random.randint(60, 100, 200),
    'blood_oxygen': np.random.randint(90, 100, 200),
    'activity_level': np.random.choice(['low','moderate','high'], 200),
    'sleep_hours': np.random.uniform(5, 8, 200)
}

df = pd.DataFrame(data)
df.to_csv('new_health_data.csv', index=False)
print("New health data saved as new_health_data.csv")

# --------------------------
# Step 2: Run Isolation Forest for anomaly detection
# --------------------------
X = df[['heart_rate','blood_oxygen','sleep_hours']]
model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(X)
df['anomaly'] = df['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

# --------------------------
# Step 3: Save the results for the dashboard
# --------------------------
df.to_csv('new_health_anomalies.csv', index=False)
print("Anomaly detection results saved as new_health_anomalies.csv")
