from docx import Document

doc = Document("dataset/raw_resumes/kartik.docx")

print("---- FULL TEXT ----")
for p in doc.paragraphs:
    print(repr(p.text))