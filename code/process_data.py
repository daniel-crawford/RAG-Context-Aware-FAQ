from datasets import load_dataset
import json

def preprocess_hf_data(dataset_name: str, output_file: str, chunk_size: int = 300, max_articles: int = 1000):
    """
    Load and preprocess a Wikipedia dataset from Hugging Face.

    Args:
        dataset_name (str): The name of the Hugging Face dataset 
        output_file (str): Path to the output JSON file to save chunks.
        chunk_size (int): Maximum length of each chunk (in words).
        max_articles (int): Maximum number of articles to process.

    Returns:
        None
    """
    def chunk_text(text: str, chunk_size: int):
        """
        Split a text into smaller chunks of a specified size.
        """
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    # Load the dataset
    #print("Loading the dataset...")
    dataset = load_dataset(dataset_name, "20220301.en", split="train")  # English Wikipedia snapshot

    # Preprocess articles
    chunks = []
    for i, article in enumerate(dataset):
        if i >= max_articles:  # Limit the number of articles processed
            break
        text = article.get("text", "")
        if text:
            text_chunks = chunk_text(text, chunk_size)
            chunks.extend(text_chunks)

    # Save preprocessed chunks to a JSON file
    #print(f"Saving {len(chunks)} chunks to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    #print("Preprocessing complete!")

