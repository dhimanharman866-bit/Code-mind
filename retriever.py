def get_retriever(vector_store):
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 30}
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