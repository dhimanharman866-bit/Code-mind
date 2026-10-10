import os
import logging
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq

# Suppress MultiQueryRetriever's verbose generated-queries log
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.WARNING)


def get_retriever(vector_store):
    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 25}
    )

    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",   # fast, cheap model just for query generation
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )

    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )


def get_all_sources(vector_store):
    """Get all unique source file paths from the vectorstore."""
    try:
        # Access docstore directly
        docstore = vector_store.docstore
        if hasattr(docstore, '_dict'):
            sources = set()
            for doc in docstore._dict.values():
                src = doc.metadata.get('source', 'unknown')
                sources.add(src)
            return sorted(sources)
    except Exception:
        pass
    return []
