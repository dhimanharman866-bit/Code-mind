import os
import shutil
import time
from langchain_community.vectorstores import FAISS
from config import INDEX_PATH

def build_or_load_vectorstore(chunks, embeddings, force_new=False):

    print("INDEX_PATH =", INDEX_PATH)

    if os.path.exists(INDEX_PATH):
        print("Trying to delete old index...")
        # Retry on Windows file locking issues
        for attempt in range(5):
            try:
                shutil.rmtree(INDEX_PATH)
                print("Deleted old index")
                break
            except PermissionError as e:
                if attempt == 4:
                    raise
                print(f"Permission denied, retrying in 0.5s... ({attempt+1}/5)")
                time.sleep(0.5)

    print("Building FAISS...")
    vs = FAISS.from_documents(chunks, embeddings)

    print("Saving FAISS...")
    vs.save_local(INDEX_PATH)

    print("Saved successfully")

    return vs