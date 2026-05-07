import chromadb
from pathlib import Path

print("Importing chroma...")
VECTORSTORE_DIR = Path("backend") / "vectorstore_local"
print(f"Creating client at {VECTORSTORE_DIR}...")
chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
print("Client created.")
print("Getting collection...")
collection = chroma_client.get_or_create_collection(name="algerian_law_local", metadata={"hnsw:space": "cosine"})
print("Collection created.")
