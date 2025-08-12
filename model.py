import pandas as pd
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib

# Generate a synthetic concentric circles dataset
X, y = make_circles(n_samples=400, noise=0.04, factor=0.5, random_state=42)

# Scale features for DBSCAN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train DBSCAN model
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan.fit(X_scaled)

# Save model, scaler, and training data for nearest neighbor assignment
joblib.dump((dbscan, scaler, X_scaled, dbscan.labels_), 'concentric_dbscan_model.pkl')

# Save feature names
features = ["X1", "X2"]
joblib.dump(features, 'concentric_dbscan_features.pkl')
