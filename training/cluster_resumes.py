import os
import shutil
import pickle
import pdfplumber
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(PROJECT_ROOT, "dataset", "raw_resumes")
output_dir = os.path.join(PROJECT_ROOT, "dataset", "clustered_resumes")
models_path = os.path.join(PROJECT_ROOT, "models")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

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


def load_resumes(folder_path):
    texts = []
    filenames = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf") or file.endswith(".docx"):
            path = os.path.join(folder_path, file)
            text = extract_text(path)
            if text.strip():   # only keep non-empty text
                texts.append(text)
                filenames.append(file)
            else:
                print(f"Skipped unreadable file: {file}")

    return texts, filenames


# Load data
texts, filenames = load_resumes(dataset_path)

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# Create cluster folders
for i in range(3):
    os.makedirs(os.path.join(output_dir, f"cluster_{i}"), exist_ok=True)

# Copy files into clusters
for file, label in zip(filenames, cluster_labels):
    src = os.path.join(dataset_path, file)
    dst = os.path.join(output_dir, f"cluster_{label}", file)
    shutil.copy(src, dst)

# Train classifier (pseudo-labeling)

classifier = SVC(kernel='linear')
classifier.fit(X, cluster_labels)

# Save models
pickle.dump(vectorizer, open(os.path.join(models_path, "vectorizer.pkl"), "wb"))
pickle.dump(kmeans, open(os.path.join(models_path, "kmeans.pkl"), "wb"))
pickle.dump(classifier, open(os.path.join(models_path, "classifier.pkl"), "wb"))

print("Clustering + model training completed successfully.")