import os
import re
import pdfplumber
import docx2txt


MIN_CHAR_THRESHOLD = 100       # minimum total characters
MIN_WORD_THRESHOLD = 20        # minimum words


def extract_text_from_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except:
        return ""
    return text.strip()


def extract_text_from_docx(file_path):
    try:
        text = docx2txt.process(file_path)
        return text.strip()
    except:
        return ""


def is_valid_resume(text):
    if not text:
        return False

    if len(text) < MIN_CHAR_THRESHOLD:
        return False

    words = text.split()
    if len(words) < MIN_WORD_THRESHOLD:
        return False

    # Check if text has enough alphabets
    alpha_ratio = len(re.findall(r"[A-Za-z]", text)) / len(text)
    if alpha_ratio < 0.5:
        return False

    return True


def extract_all_resumes(folder_path):
    texts = []
    filenames = []
    discarded = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if file.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        else:
            continue

        if is_valid_resume(text):
            texts.append(text)
            filenames.append(file)
        else:
            discarded.append(file)

    print("\nValid resumes:", len(texts))
    print("Discarded resumes:", len(discarded))
    print("Discarded files:", discarded)

    return texts, filenames