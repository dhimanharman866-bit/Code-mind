import os
import shutil
from langchain_community.vectorstores import FAISS
from config import INDEX_PATH

def build_or_load_vectorstore(chunks, embeddings, force_new=False):
    # Always build fresh — never reuse stale index
    if os.path.exists(INDEX_PATH):
        shutil.rmtree(INDEX_PATH)

    if not chunks:
        raise ValueError("No chunks to index. Check your loader — 0 documents were found.")

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(INDEX_PATH)
    return vs