from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(texts):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    return X, vectorizer