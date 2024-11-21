from transformers import pipeline

def rerank_results(query, retrieved_chunks, model_name):
    """
    Rerank retrieved chunks using a cross-encoder model.

    Args:
        query (str): The input query string.
        retrieved_chunks (list): List of retrieved chunks (dictionaries with "text" and "score").
        model_name (str): Name of the cross-encoder model to use.

    Returns:
        List[dict]: Retrieved chunks with updated scores, sorted by relevance.
    """
    #print(f"Loading reranking model: {model_name}...")
    reranker = pipeline("text-classification", model=model_name)

    #print("Reranking results...")
    reranked_results = []
    for chunk in retrieved_chunks:
        # Combine query and chunk text for reranking
        input_text = f"Query: {query} Document: {chunk['text']}"
        score = reranker(input_text, top_k=1)[0]['score']  # Get relevance score
        reranked_results.append({
            "text": chunk["text"],
            "initial_score": chunk["score"],
            "rerank_score": score
        })

    # Sort by reranked scores (descending)
    reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked_results

