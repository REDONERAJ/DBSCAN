# 🔵 DBSCAN Clustering Web App – Concentric Circles Dataset

This project is a **Flask-based web application** that demonstrates **Density-Based Spatial Clustering** (DBSCAN) on a **synthetic concentric circles dataset**.  
It uses `make_circles` from scikit-learn to generate two rings of data points and applies DBSCAN to separate clusters of arbitrary shape — an area where traditional algorithms like K-Means fail.

---

## 📊 Features
- Generates **synthetic concentric circles dataset** locally (no downloads needed).
- Uses **DBSCAN** for density-based clustering, capable of detecting non-linear cluster shapes.
- Identifies noise points in sparse areas.
- Web form lets you input `X1` and `X2` coordinates and see **cluster** (`0/1`) or **noise** (`-1`).

---

## 📂 Project Structure
```
├── model.py # Generates dataset, trains DBSCAN, saves model and data
├── app.py # Flask server for input & prediction
├── templates/
│ └── index.html # Web form UI
├── concentric_dbscan_model.pkl # Saved DBSCAN model with training data
├── concentric_dbscan_features.pkl # Saved feature names
├── requirements.txt # Dependencies
└── README.md # Documentation

```

---

## ⚙️ Installation

1. **Clone this repository**
git clone <your_repo_url>
cd <project_folder>



2. **Install dependencies**
pip install -r requirements.txt



3. **Train the model**
python model.py



4. **Run the Flask app**
python app.py



5. **Open in your browser**
http://127.0.0.1:5000/


---

## 📝 Example Inputs for Testing
| Example Type  | X1    | X2    | Expected Output          |
|---------------|-------|-------|--------------------------|
| Cluster 0     | 0.8   | 0.2   | Cluster 0 (outer circle) |
| Cluster 1     | 0.0   | -0.5  | Cluster 1 (inner circle) |
| Noise         | 2.8   | 2.9   | Noise (-1)               |

---

## 📦 Requirements
Flask
scikit-learn
pandas
numpy
joblib


---

## 📚 How It Works
1. `model.py` generates a **2D concentric circles dataset** using `make_circles`.
2. Features are standardized with **StandardScaler**.
3. **DBSCAN** is trained with `eps=0.3` and `min_samples=5`.
4. Since DBSCAN has **no predict method**, new inputs are assigned to the nearest point’s cluster from the training set.
5. Flask app provides a user-friendly interface to input coordinates and get cluster or noise output.

---

## 🛠 Notes
- Adjust `eps` and `min_samples` in `model.py` to change DBSCAN cluster sensitivity.
- Increasing noise in `make_circles` will produce more scattered points and may increase noise detection.

---

## 📷 Screenshot
