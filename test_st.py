import sys
print("Importing SentenceTransformer...")
from sentence_transformers import SentenceTransformer

print("Loading model...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("Model loaded. Encoding...")
embeddings = model.encode(["Hello World"])
print(f"Encoded shape: {embeddings.shape}")
print("Done.")
