import os
import pickle
import pdfplumber
import docx


# -----------------------------
# LOAD MODELS
# -----------------------------
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
classifier = pickle.load(open("models/classifier.pkl", "rb"))

# Map cluster → job role
cluster_role_map = {
    0: "Data Science",
    1: "Web Development",
    2: "Finance"
}


# -----------------------------
# TEXT EXTRACTION
# -----------------------------
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

    return ""


# -----------------------------
# PREDICTION
# -----------------------------
def predict_resume(file_path):

    text = extract_text(file_path)

    if not text.strip():
        return "Could not extract text"

    X = vectorizer.transform([text])
    cluster = classifier.predict(X)[0]

    return cluster_role_map.get(cluster, "Unknown Role")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    path = input("Enter file path OR folder path: ").strip()

    if os.path.isfile(path):

        print("\nSingle Resume Prediction:\n")
        result = predict_resume(path)
        print(f"{os.path.basename(path)} → {result}")

    elif os.path.isdir(path):

        print("\nBatch Resume Prediction:\n")

        for file in os.listdir(path):
            if file.endswith((".pdf", ".docx")):
                file_path = os.path.join(path, file)
                result = predict_resume(file_path)
                print(f"{file} → {result}")

    else:
        print("Invalid path.")