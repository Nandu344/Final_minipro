from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def create_tfidf_features(texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)

    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    return X, vectorizer