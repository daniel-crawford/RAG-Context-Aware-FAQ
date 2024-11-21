# RAG-Context-Aware-FAQ
Context-Aware FAQ Generator with RAG

# Project Overview

This project demonstrates the use of Retrieval-Augmented Generation (RAG) to enhance question-answering systems by leveraging external knowledge from a retrieval step before generating responses. Using the **Hugging Face** library, this project integrates a document retrieval process followed by the generation of natural language responses from the retrieved documents.

**Key Features:**
- Retrieval-augmented generation using models like **RAG-Token** and **T5**.
- Integration of a custom retriever for document retrieval (using Hugging Face's **RagRetriever**).
- Clean and structured code that can be reused for various retrieval-augmented tasks.

# Options and Defaults
These options refer to various parameters used in a Retrieval-Augmented Generation (RAG) setup where you're integrating retrieval-based systems with a language model for generating answers based on a knowledge base. Here's an explanation of each option and its default value:

1. "dataset_name": "wikipedia"
Explanation: This specifies the name of the dataset you're using for document retrieval. In this case, it’s set to "wikipedia", which typically refers to the Wikipedia dataset. You could use another dataset name if you're working with a custom dataset.
Default: "wikipedia" (you can change it to any dataset you'd like).
2. "chunks_file": "./data/wikipedia_chunks.json"
Explanation: This is the file path where the document chunks are stored. The data is divided into smaller chunks (usually paragraphs or sections), which are indexed for efficient retrieval. This file contains the preprocessed chunks, typically in JSON format.
Default: "./data/wikipedia_chunks.json". You can adjust the path based on where you store your chunked data.
3. "faiss_file": "./data/document_index.faiss"
Explanation: This is the path to the FAISS index file, which stores the embeddings of your document chunks. FAISS (Facebook AI Similarity Search) is a library used for efficient similarity search and clustering. It helps with fast retrieval of the most relevant document chunks when you query the model.
Default: "./data/document_index.faiss". This is where the FAISS index is saved and should be loaded from when performing retrieval.
4. "chunk_size": 300
Explanation: This is the maximum size (in characters or tokens) for each document chunk. Larger chunks may contain more context but will also increase memory usage and decrease retrieval speed. Smaller chunks improve retrieval precision but may require more segments to represent the full knowledge base.
Default: 300 tokens or characters. You can adjust the chunk size depending on the trade-off between retrieval efficiency and context richness.
5. "max_articles": 10000
Explanation: This defines the maximum number of articles or documents you want to include in your dataset for retrieval. If you have a large knowledge base like Wikipedia, you may want to limit the number of documents to a manageable number for faster retrieval and processing.
Default: 10000. You can adjust this if your dataset is smaller or if you want to include more articles.
6. "top_k_results": 10
Explanation: This determines how many of the top results (relevant document chunks) you want to retrieve from the FAISS index when given a query. It’s a trade-off between response quality and performance.
Default: 10 results. Adjusting this parameter can affect the relevance and speed of the retrieved results.
7. "encoding_model": "all-MiniLM-L6-v2"
Explanation: This specifies the model used to encode documents into vector representations (embeddings). all-MiniLM-L6-v2 is a lightweight model from sentence-transformers that is designed for efficient and accurate sentence embeddings.
Default: "all-MiniLM-L6-v2". You could replace this with another sentence transformer model if you'd like a different encoding model (e.g., paraphrase-MiniLM-L6-v2 for better quality but slightly slower performance).
8. "reranking_model": "cross-encoder/ms-marco-MiniLM-L-12-v2"
Explanation: This is the model used for reranking the top K retrieved documents based on relevance to the query. A cross-encoder model like "cross-encoder/ms-marco-MiniLM-L-12-v2" takes both the query and document as input and scores their relevance together. This improves the quality of the retrieval by ensuring that only the most relevant documents are used for the answer generation phase.
Default: "cross-encoder/ms-marco-MiniLM-L-12-v2". You could choose a different reranking model if needed (e.g., roberta-large-ms-marco).
9. "answer_generation_model": "gpt-3.5-turbo"
Explanation: This specifies the language model used to generate the final answer based on the retrieved document chunks. gpt-3.5-turbo is a powerful language model provided by OpenAI (a variant of GPT-3) optimized for performance and cost. You might use GPT-4 for more advanced reasoning capabilities or choose a model like T5 or BART for generating answers in a more controlled way.
Default: "gpt-3.5-turbo". You can replace this with any compatible model from OpenAI (e.g., gpt-4, text-davinci-003, etc.).

# Open AI API
Add your personal openai api key to a .env sibling file to main.py

### Summary of Parameter Defaults and Use Cases:
"dataset_name": "wikipedia": Refers to the knowledge base being used for document retrieval.
"chunks_file": "./data/wikipedia_chunks.json": Path to preprocessed document chunks.
"faiss_file": "./data/document_index.faiss": Path to the FAISS index of the chunk embeddings.
"chunk_size": 300: Defines the maximum chunk size for document retrieval.
"max_articles": 10000: Limits the number of documents considered during retrieval.
"top_k_results": 10: Specifies how many top documents to retrieve and rerank.
"encoding_model": "all-MiniLM-L6-v2": The embedding model for encoding document chunks.
"reranking_model": "cross-encoder/ms-marco-MiniLM-L-12-v2": Model for reranking retrieved documents based on relevance.
"answer_generation_model": "gpt-3.5-turbo": The model used for generating the final answer based on retrieved documents.
These settings help configure the RAG pipeline for optimal document retrieval and generation, balancing between efficiency and the quality of the results.

# Installation

To run this project, you'll need Python and the following libraries:

- `transformers` (Hugging Face)
- `torch`
- `faiss-cpu` (for document retrieval)
- `sentence-transformers` (for embedding-based retrieval)
- `openai` (for generation)

### Prerequisites

Install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```