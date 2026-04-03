import os, hashlib, json
from langchain_core.documents import Document

HASH_FILE = "file_hashes.json"  # inline to remove config dependency

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".cpp", ".c", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php", ".cs",
    ".html", ".css", ".md", ".txt", ".yaml", ".yml", ".json", ".toml"
}

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def load_hashes():
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            return json.load(f)
    return {}

def save_hashes(hashes):
    with open(HASH_FILE, "w") as f:
        json.dump(hashes, f)

def load_document(folder_path):
    documents = []
    old_hashes = load_hashes()
    new_hashes = {}

    for root, dirs, files in os.walk(folder_path):
        # Skip hidden dirs and common noise folders
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {
            "__pycache__", "node_modules", ".git", ".venv", "venv", "dist", "build"
        }]

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            full_path = os.path.join(root, file)

            try:
                h = file_hash(full_path)
                new_hashes[full_path] = h

                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()

                if not content:  # skip empty files
                    continue

                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": full_path,
                        "language": ext.lstrip("."),
                        "filename": file,
                    }
                ))
            except Exception as e:
                print(f"[loader] Skipping {full_path}: {e}")
                continue

    save_hashes(new_hashes)
    return documents