import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

from utils.text_extraction import extract_all_resumes


# Path to clustered resumes folder
DATASET_PATH = "dataset/clustered_resumes"

texts = []
labels = []

# Read resumes from cluster folders
for category in os.listdir(DATASET_PATH):
    category_path = os.path.join(DATASET_PATH, category)

    if os.path.isdir(category_path):
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)

            if file.endswith(".pdf") or file.endswith(".docx"):
                extracted_texts, _ = extract_all_resumes(category_path)
                texts.extend(extracted_texts)
                labels.extend([category] * len(extracted_texts))
                break


print("Total samples:", len(texts))

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
X = vectorizer.fit_transform(texts)
y = labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM classifier
model = LinearSVC()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/resume_classifier.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel saved successfully.")