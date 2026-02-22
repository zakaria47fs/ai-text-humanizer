import os
import glob

import chromadb
from chromadb.utils import embedding_functions


def build_corpus(samples_dir: str, db_path: str = "./chroma_db"):
    """Build ChromaDB vector store from human text samples."""

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="human_writing",
        embedding_function=ef
    )

    files = glob.glob(os.path.join(samples_dir, "*.txt"))
    if not files:
        print(f"No .txt files found in {samples_dir}. Corpus not built.")
        return collection

    documents = []
    ids = []
    metadatas = []

    for i, filepath in enumerate(files):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        documents.append(text)
        ids.append(f"sample_{i}")
        metadatas.append({"source": os.path.basename(filepath)})

    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    print(f"Added {len(documents)} samples to corpus.")
    return collection


def get_style_references(input_text: str, n_results: int = 3, db_path: str = "./chroma_db") -> list[str]:
    """Retrieve the most stylistically similar human writing samples."""

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=db_path)

    try:
        collection = client.get_collection(
            name="human_writing",
            embedding_function=ef
        )
    except Exception:
        # Corpus not built yet — return empty list so pipeline continues
        return []

    count = collection.count()
    if count == 0:
        return []

    actual_n = min(n_results, count)
    results = collection.query(
        query_texts=[input_text],
        n_results=actual_n
    )

    return results["documents"][0]  # List of matching human text samples
