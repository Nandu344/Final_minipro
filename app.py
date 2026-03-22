import streamlit as st
import pickle
import pdfplumber
from docx import Document
import pandas as pd
import os
from datetime import datetime

UPLOAD_DIR = "dataset/raw_resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)

vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
classifier = pickle.load(open("models/classifier.pkl", "rb"))

def extract_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    elif file.name.endswith(".docx"):
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text
    return text


cluster_role_map = {
    0: "Data Science",
    1: "Web Development",
    2: "Finance"
}

st.title("AI Resume Screening System")

uploaded_files = st.file_uploader("Upload resumes", accept_multiple_files=True)

# Predict button
if st.button("Predict"):

    if uploaded_files:
        results = []

        for file in uploaded_files:
            # Create unique filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            saved_filename = f"{timestamp}_{file.name}"
            save_path = os.path.join(UPLOAD_DIR, saved_filename)
            
            # Save file
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
                
            # Extract text from saved file
            text = extract_text(open(save_path, "rb"))
                
            # Skip unreadable files
            if not text.strip():
                st.warning(f"{file.name} is unreadable.")
                continue
            
            X = vectorizer.transform([text])
            cluster = classifier.predict(X)[0]
            role = cluster_role_map.get(cluster, "Unknown")
            
            results.append({
                "Name": file.name,
                "Job Category": role
            })

        # Convert to DataFrame
        df = pd.DataFrame(results)
        df.index = df.index + 1

        # Display table
        st.subheader("Prediction Results")
        st.dataframe(df)

    else:
        st.warning("Please upload at least one file.")

st.markdown("""
<style>
.stButton > button {
    background-color: #2196F3 !important;  /* Blue */
    color: white !important;
    border-radius: 8px;
    border: none;
    padding: 0.5em 1em;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #2196F3 !important;  /* Darker blue */
    color: white !important;
}
</style>
""", unsafe_allow_html=True)