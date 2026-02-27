import joblib
from utils.text_extraction import extract_text_from_pdf
from utils.preprocessing import clean_text

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
kmeans = joblib.load("models/kmeans_model.pkl")

def predict_resume_category(file_path):
    text = extract_text_from_pdf(file_path)
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    cluster = kmeans.predict(features)
    return f"Predicted Category: cluster_{cluster[0]}"


if __name__ == "__main__":
    test_file = "sample_resume.pdf"
    print(predict_resume_category(test_file))