import pandas as pd
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib

X, y = make_circles(n_samples=400, noise=0.04, factor=0.5, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan.fit(X_scaled)


joblib.dump((dbscan, scaler, X_scaled, dbscan.labels_), 'concentric_dbscan_model.pkl')


features = ["X1", "X2"]
joblib.dump(features, 'concentric_dbscan_features.pkl')
