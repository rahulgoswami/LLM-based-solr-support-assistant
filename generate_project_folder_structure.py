# setup_project_structure.py

import os

FOLDERS = [
    "data/mailing_list/solr-user",
    "data/mailing_list/lucene-user",
    "data/github_issues",          # for later
    "data/docs",                   # for later
    "vector_store",                # ChromaDB file-backed index
    "scripts",
    "core",
    "app",                         # UI / API layer
    "notebooks",                   # optional for experiments
    "logs"
]

FILES = {
    ".gitignore": """
venv/
__pycache__/
*.pyc
*.log
vector_store/
""",
    "README.md": "# Solr & Lucene Technical Support Assistant\n",
    "requirements.txt": "",  # will be overwritten by your real one
}

def create_structure():
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")
    for filename, content in FILES.items():
        with open(filename, "w") as f:
            f.write(content.strip())
        print(f"Created file: {filename}")

if __name__ == "__main__":
    create_structure()

