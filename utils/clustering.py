from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def perform_clustering(X, k=4):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels

def train_classifier(X, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = SVC(kernel="linear", class_weight="balanced")
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, accuracy