import os
import shutil
from utils.text_extraction import extract_all_resumes
from utils.preprocessing import clean_text
from utils.feature_engineering import create_tfidf_features
from utils.clustering import apply_kmeans

RAW_FOLDER = "dataset/raw_resumes"
CLUSTER_FOLDER = "dataset/clustered_resumes"

# Step 1: Extract
texts, filenames = extract_all_resumes(RAW_FOLDER)

# Step 2: Preprocess
cleaned_texts = [clean_text(text) for text in texts]

# Step 3: TF-IDF
X, vectorizer = create_tfidf_features(cleaned_texts)

# Step 4: KMeans
labels = apply_kmeans(X, n_clusters=4)

# Step 5: Create cluster folders
os.makedirs(CLUSTER_FOLDER, exist_ok=True)

for i in range(4):
    os.makedirs(os.path.join(CLUSTER_FOLDER, f"cluster_{i}"), exist_ok=True)

# Step 6: Move files
for file, label in zip(filenames, labels):
    source = os.path.join(RAW_FOLDER, file)
    destination = os.path.join(CLUSTER_FOLDER, f"cluster_{label}", file)
    shutil.copy(source, destination)

print("Clustering completed successfully.")