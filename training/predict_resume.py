import os
import sys
import pickle
from pathlib import Path

import pdfplumber
import docx

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))


# ---------------------------------
# LOAD MODELS (only once)
# ---------------------------------
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
classifier = pickle.load(open("models/classifier.pkl", "rb"))

# Manual cluster → job role mapping
cluster_role_map = {
    0: "Data Science",
    1: "Web Development",
    2: "Finance"
}


# ---------------------------------
# TEXT EXTRACTION
# ---------------------------------
def extract_text(file_path):

    if file_path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text()
        return text

    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    else:
        return ""


# ---------------------------------
# PREDICTION FUNCTION
# ---------------------------------
def predict_text(text):
    X = vectorizer.transform([text])
    prediction = classifier.predict(X)[0]
    return cluster_role_map.get(prediction, "Unknown Role")


# ---------------------------------
# HANDLE SINGLE FILE
# ---------------------------------
def predict_single(file_path):

    text = extract_text(file_path)

    if not text.strip():
        print(f"{file_path} → Could not extract text.")
        return

    role = predict_text(text)

    print(f"{os.path.basename(file_path)} → {role}")


# ---------------------------------
# HANDLE FOLDER (BATCH)
# ---------------------------------
def predict_folder(folder_path):

    for file in os.listdir(folder_path):

        if file.endswith((".pdf", ".docx")):

            file_path = os.path.join(folder_path, file)
            predict_single(file_path)


# ---------------------------------
# MAIN
# ---------------------------------
if __name__ == "__main__":

    path = input("Enter file path OR folder path: ").strip()

    if os.path.isfile(path):
        print("\nSingle Resume Prediction:\n")
        predict_single(path)

    elif os.path.isdir(path):
        print("\nBatch Resume Prediction:\n")
        predict_folder(path)

    else:
        print("Invalid path.")