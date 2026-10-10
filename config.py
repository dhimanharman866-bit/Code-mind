import os
FOLDER_PATH = os.getenv("FOLDER_PATH", "")
INDEX_PATH="faiss_index_new"
HASH_FILE="file_hashes.json"

CHUNK_SIZE=1500
CHUNK_OVERLAP=200

EMBED_MODEL="BAAI/bge-large-en-v1.5"