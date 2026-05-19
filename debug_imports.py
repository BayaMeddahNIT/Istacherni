
print("DEBUG: Starting imports...")
import sys
import os
print(f"DEBUG: sys.path: {sys.path}")

print("DEBUG: Importing graph_retriever...")
try:
    from graph_rag_local.graph_retriever import graph_retrieve
    print("DEBUG: graph_retriever imported successfully.")
except Exception as e:
    print(f"DEBUG: Failed to import graph_retriever: {e}")

print("DEBUG: Importing graph_generator...")
try:
    from graph_rag_local.graph_generator import graph_generate
    print("DEBUG: graph_generator imported successfully.")
except Exception as e:
    print(f"DEBUG: Failed to import graph_generator: {e}")

print("DEBUG: All imports done.")
