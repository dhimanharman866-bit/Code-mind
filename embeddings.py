from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBED_MODEL

def get_embedding():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"batch_size":16,"normalize_embeddings":True}
    )