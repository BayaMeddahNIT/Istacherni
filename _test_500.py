import sys
import traceback
from backend.app import chat, ChatRequest

try:
    print("Testing standard")
    chat(ChatRequest(question="Test", top_k=1, rag_type="standard", history=[]))
except Exception as e:
    traceback.print_exc()

try:
    print("Testing agentic")
    chat(ChatRequest(question="Test", top_k=1, rag_type="agentic", history=[]))
except Exception as e:
    traceback.print_exc()
