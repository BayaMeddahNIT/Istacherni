import os
os.environ["PYTHONIOENCODING"] = "utf-8"
from agentic_rag.agentic_agent import agentic_answer_stream
import sys

def test_query(q):
    print(f"\n--- Testing query: {q} ---")
    for event in agentic_answer_stream(q, []):
        if event["type"] == "status":
            print(f"[STATUS]: {event.get('message')}")
        elif event["type"] == "token":
            print(event["content"], end="", flush=True)
        elif event["type"] == "sources":
            print(f"\n[SOURCES]: {len(event.get('sources', []))} sources")
    print("\n--- Done ---\n")

if __name__ == "__main__":
    test_query("ما هو الفرق بين SARL و SPA ؟")
