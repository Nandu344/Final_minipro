from sklearn.cluster import KMeans
import joblib
import os

def apply_kmeans(X, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    os.makedirs("models", exist_ok=True)
    joblib.dump(kmeans, "models/kmeans_model.pkl")

    return kmeans.labels_