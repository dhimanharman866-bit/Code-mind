from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from config import CHUNK_OVERLAP,CHUNK_SIZE

def get_language(file):
    if file.endswith(".py"):
        return Language.PYTHON
    elif file.endswith(".js"):
        return Language.JS
    elif(file.endswith(".java")):
        return Language.JAVA
    elif file.endswith(".cpp"):
        return Language.CPP
    else:
        return None

def split_document(documents):
    all_chunk=[]

    for doc in documents:
        lang=get_language(doc.metadata["source"])

        if lang:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
        else:
            splitter=RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
        chunk=splitter.split_documents([doc])
        all_chunk.extend(chunk)
    return all_chunk