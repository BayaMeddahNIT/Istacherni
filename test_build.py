import sys
from graph_rag_local.graph_builder import build_graph

if __name__ == "__main__":
    print("Starting graph build...")
    build_graph(force=True)
    print("Graph build complete!")
