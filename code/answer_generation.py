from transformers import pipeline
import openai


def generate_answer(query, reranked_chunks, model_name, open_ai_key):
    """
    Generate an answer to a query using reranked chunks and a language model.

    Args:
        query (str): The input query string.
        reranked_chunks (list): List of reranked chunks (sorted by relevance).
        model_name (str): Name of the language model to use for generation.

    Returns:
        str: Generated answer.
    """
    #print(f"Loading language model: {model_name}...")
    # Use Hugging Face's text-generation pipeline or similar interface
    #answer_generator = pipeline("text-generation", model=model_name)

    # Combine top-k context chunks into a single context string
    top_k = 10  # Number of top chunks to use
    context = "\n".join([chunk["text"] for chunk in reranked_chunks[:top_k]])
    
    #print(f"Context: \n\n {context}\n\n")

    # Format the prompt for the LLM
    prompt = (
        f"The following is relevant information for answering the query:\n\n"
        f"{context}\n\n"
        f"Based on the above information, ask a common question about {query}."
    )

    # Generate the answer
    #print("Generating answer...")

    openai.api_key = open_ai_key
    answer = openai.Completion.create(
        model=model_name,  # or gpt-4
        prompt=prompt,
        max_tokens=50
        ).choices[0].text.strip()

    return answer

