# Context-Aware FAQ Generator with RAG

import json
import os
import sys

sys.path.append('./code')
from answer_generation import *
from create_indexing import *
from process_data import *
from retrieval import *
from reranking import *


from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Access variables from the config
dataset_name = config.get("dataset_name", "wikipedia")
chunks_file = config.get("output_file", "./data/wikipedia_chunks.json")
chunk_size = config.get("chunk_size", 300)
max_articles = config.get("max_articles", 1000)
faiss_file = config.get("faiss_file", "./data/document_index.faiss")
top_k_results = config.get("top_k_results", 5)

encoding_model = config.get("encoding_model", "all-MiniLM-L6-v2")
reranking_model = config.get("reranking_model", "cross-encoder/ms-marco-MiniLM-L-12-v2")
answer_generation_model = config.get("answer_generation_model", "EleutherAI/gpt-neo-1.3B")

# Create data directory if it doesn't exist
if not os.path.exists('./data'):
    os.makedirs('./data')

def process_data():
    # Preprocess and encode text data

    preprocess_hf_data(
            dataset_name=dataset_name,
            output_file=chunks_file,
            chunk_size=chunk_size,
            max_articles=max_articles  # Process the first 1000 articles for quick experimentation
        )

    # Load and encode text chunks
    text_chunks = load_chunks(chunks_file)
    embeddings = encode_chunks(text_chunks, encoding_model)

    # Create and save FAISS index
    faiss_index = create_faiss_index(embeddings)
    save_faiss_index(faiss_index, faiss_file)

    print("Indexing complete!\n\n")

def process_query(user_query):
    # Load the FAISS index and text chunks
    index = load_faiss_index(faiss_file)

    # Encode the query
    query_vector = encode_query(user_query, encoding_model)

    # Load and encode text chunks
    text_chunks = load_chunks(chunks_file)

    # Retrieve similar chunks
    similar_chunks = retrieve_similar_chunks(index, query_vector, text_chunks, top_k=top_k_results)
    # Perform reranking
    reranked_results = rerank_results(user_query, similar_chunks, reranking_model)


    # Generate answer using the LLM
    final_answer = generate_answer(user_query, reranked_results, answer_generation_model, open_ai_key)

    return final_answer


if __name__ == "__main__":
    if os.path.exists('./data') and os.listdir('./data'):
        answer = input("It seems that your data directory is not empty. Would you like to clear it? (Y/n):      ")
        if answer.lower() == 'y':
            os.system('rm -r ./data/*')
            print("Data directory cleared.")
    
            # Preprocess data
            print("Processing Data...")
            process_data()

    # Process query
    while True:
        user_query = input("Enter your Topic (Q to Quit): ")
        if user_query.lower() == 'q':
            break
        final_answer = process_query(user_query)

        # Print the answer
        print("\n\n\n\nAnswer:", final_answer, "\n\n\n\n")

    