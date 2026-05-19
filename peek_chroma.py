import chromadb
from pathlib import Path

CHROMA_PATH = Path(__file__).parent / "camelbert_rag" / "chroma_db"
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection("algerian_law_camelbert")

# Get 5 items
results = collection.get(limit=5)

with open("chroma_sample.txt", "w", encoding="utf-8") as f:
    for i in range(len(results["ids"])):
        f.write(f"ID: {results['ids'][i]}\n")
        f.write(f"Metadata: {results['metadatas'][i]}\n")
        f.write(f"Raw Text Chunk:\n{results['documents'][i]}\n")
        f.write("-" * 80 + "\n")
