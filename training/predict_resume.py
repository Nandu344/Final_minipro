import os
import sys
import pickle
from pathlib import Path

import pdfplumber
import docx

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))


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


def predict_resume(file_path):

    vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
    classifier = pickle.load(open("models/classifier.pkl", "rb"))

    text = extract_text(file_path)

    X = vectorizer.transform([text])

    prediction = classifier.predict(X)

    return prediction[0]


if __name__ == "__main__":
    file_path = input("Enter resume path: ")
    result = predict_resume(file_path)

# Manual mapping based on cluster interpretation
    cluster_role_map = {
        0: "Data Science",
        1: "Web Development",
        2: "Finance"
    }

    job_role = cluster_role_map.get(result, "Unknown Role")

    print(f"\nPredicted Job Role: {job_role}")