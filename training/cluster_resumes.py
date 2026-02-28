import os
import sys
import pickle
from pathlib import Path

import pdfplumber
import docx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


# -------------------------------
# FIX PROJECT ROOT
# -------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))


# -------------------------------
# LOAD RESUMES
# -------------------------------
def load_resumes(folder_path):
    texts = []
    filenames = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        try:
            if file.endswith(".pdf"):
                text = ""
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        if page.extract_text():
                            text += page.extract_text()

            elif file.endswith(".docx"):
                doc = docx.Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])

            else:
                continue

            if text.strip():
                texts.append(text)
                filenames.append(file)

        except Exception as e:
            print(f"Error reading {file}: {e}")

    return texts, filenames


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():

    dataset_path = os.path.join(PROJECT_ROOT, "dataset", "raw_resumes")

    print("\nLoading resumes...")
    resumes, filenames = load_resumes(dataset_path)

    if len(resumes) == 0:
        print("No resumes found.")
        return

    print(f"Total resumes loaded: {len(resumes)}")

    # -------------------------------
    # TF-IDF
    # -------------------------------
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=2000,
        ngram_range=(1,2)
    )

    X = vectorizer.fit_transform(resumes)

    # -------------------------------
    # KMEANS CLUSTERING
    # -------------------------------
    print("\nClustering resumes...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    cluster_labels = kmeans.labels_

    # Print cluster assignment
    print("\nCluster Results:")
    for file, label in zip(filenames, cluster_labels):
        print(f"{file} → Cluster {label}")

    # -------------------------------
    # SHOW TOP WORDS PER CLUSTER
    # -------------------------------
    print("\nTop terms per cluster:")

    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    for i in range(3):
        print(f"\nCluster {i} top words:")
        for ind in order_centroids[i, :10]:
            print(terms[ind])

    # -------------------------------
    # TRAIN CLASSIFIER USING PSEUDO LABELS
    # -------------------------------
    print("\nTraining classifier using pseudo labels...")

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X, cluster_labels)

    # -------------------------------
    # SAVE EVERYTHING
    # -------------------------------
    models_path = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_path, exist_ok=True)

    pickle.dump(kmeans, open(os.path.join(models_path, "kmeans.pkl"), "wb"))
    pickle.dump(vectorizer, open(os.path.join(models_path, "vectorizer.pkl"), "wb"))
    pickle.dump(classifier, open(os.path.join(models_path, "classifier.pkl"), "wb"))

    print("\n✅ Clustering + Classifier training complete.")
    print("Models saved in /models")


if __name__ == "__main__":
    main()