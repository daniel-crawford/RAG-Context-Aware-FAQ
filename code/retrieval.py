import faiss
from sentence_transformers import SentenceTransformer
import json
import numpy as np

def load_faiss_index(index_file: str):
    """
    Load a FAISS index from a file.

    Args:
        index_file (str): Path to the FAISS index file.

    Returns:
        faiss.IndexFlatL2: Loaded FAISS index.
    """
    #print(f"Loading FAISS index from {index_file}...")
    return faiss.read_index(index_file)


def load_text_chunks(input_file: str):
    """
    Load preprocessed text chunks from a JSON file.

    Args:
        input_file (str): Path to the JSON file containing text chunks.

    Returns:
        List[str]: List of text chunks.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        chunks = json.load(file)
    return chunks


def encode_query(query: str, model_name: str):
    """
    Encode a query string into a vector embedding using a pre-trained model.

    Args:
        query (str): The input query string.
        model_name (str): Name of the SentenceTransformers model to use.

    Returns:
        np.ndarray: The encoded query vector.
    """
    #print(f"Encoding query: \"{query}\"...")
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query], show_progress_bar=False)
    return np.array(query_embedding, dtype='float32')


def retrieve_similar_chunks(index, query_vector, text_chunks, top_k=5):
    """
    Retrieve the most similar text chunks for a given query vector.

    Args:
        index: FAISS index.
        query_vector: Encoded query vector.
        text_chunks (list): List of text chunks corresponding to the index.
        top_k (int): Number of top results to retrieve.

    Returns:
        List[dict]: List of retrieved chunks with their scores.
    """
    #print(f"Searching for the top {top_k} similar chunks...")
    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:  # Ensure valid index
            results.append({
                "text": text_chunks[idx],
                "score": distances[0][i]
            })
    return results


