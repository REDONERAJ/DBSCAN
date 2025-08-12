from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
dbscan, scaler, X_train_scaled, labels = joblib.load('concentric_dbscan_model.pkl')
features = joblib.load('concentric_dbscan_features.pkl')

@app.route("/", methods=["GET", "POST"])
def index():
    cluster = None
    if request.method == "POST":
        try:
            values = [float(request.form[f]) for f in features]
            scaled_values = scaler.transform([values])

            distances = np.linalg.norm(X_train_scaled - scaled_values, axis=1)
            nearest_idx = distances.argmin()
            cluster = int(labels[nearest_idx])
        except:
            cluster = "Invalid input"
    return render_template("index.html", features=features, cluster=cluster)

if __name__ == "__main__":
    app.run(debug=True)
