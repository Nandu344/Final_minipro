import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pdfplumber
from docx import Document

DATASET_PATH = "dataset/clustered_resumes"

def extract_text(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text
    return text


texts = []
labels = []

for cluster in os.listdir(DATASET_PATH):
    cluster_path = os.path.join(DATASET_PATH, cluster)

    if os.path.isdir(cluster_path):
        for file in os.listdir(cluster_path):
            file_path = os.path.join(cluster_path, file)
            text = extract_text(file_path)
            if text.strip():
                texts.append(text)
                labels.append(cluster)
            else:
                print(f"Skipped unreadable file: {file}")


vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(texts)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X, labels)

pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
pickle.dump(classifier, open("models/classifier.pkl", "wb"))

print("Classifier trained successfully.")