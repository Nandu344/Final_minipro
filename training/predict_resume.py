import pickle
import pdfplumber
from docx import Document

vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
classifier = pickle.load(open("models/classifier.pkl", "rb"))

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


def predict_file(file_path):
    text = extract_text(file_path)
    if not text.strip():
        print("Error: Could not extract readable text from file.")
        exit()
    X = vectorizer.transform([text])
    prediction = classifier.predict(X)[0]
    return prediction


if __name__ == "__main__":
    file_path = input("Enter file path: ")
    result = predict_file(file_path)
    print("Predicted Cluster:", result)