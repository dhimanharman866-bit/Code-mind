from sentence_transformers import CrossEncoder

_reranker = None

def get_reranker():
    """Lazily load the cross-encoder reranker (cached after first call)."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank_docs(query: str, docs: list, top_n: int = 5) -> list:
    """
    Re-score retrieved docs by reading the query and each chunk together.
    Returns the top_n highest-scoring docs, ordered by relevance.
    """
    if not docs:
        return docs

    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    # Sort by score descending, keep top_n
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]
