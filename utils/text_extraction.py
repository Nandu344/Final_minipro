import os

def load_resumes(folder_path):
    texts = []
    filenames = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            path = os.path.join(folder_path, file)

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()

                if len(content) > 0:
                    texts.append(content)
                    filenames.append(file)

    return texts, filenames