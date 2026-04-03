from config import FOLDER_PATH
from loader import load_document
from chunker import split_document
from embeddings import get_embedding
from vectorstore import build_or_load_vectorstore
from retriever import get_retriever
from qa_chain import build_qa_chain

def main():
    docs=load_document(FOLDER_PATH)
    
    chunks=split_document(docs)

    embeddings=get_embedding()
    vector_store=build_or_load_vectorstore(chunks,embeddings)

    retriever=get_retriever(vector_store)
    qa=build_qa_chain(retriever)

    print("Ready! type 'exit' to quit")

    while True:
        query=input("ask:").strip()
        if query.lower()=="exit":
            break
        result=qa.invoke(query)
        print("answer: ",result)

        print("\n📂 Sources:")
        for doc in docs:
            print(doc.metadata["source"])

if __name__ == "__main__":
    main()