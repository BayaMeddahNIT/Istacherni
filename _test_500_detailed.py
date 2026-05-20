import os
from dotenv import load_dotenv
load_dotenv('.env')

from backend.app import chat, ChatRequest
import traceback

try:
    print("Testing Agentic RAG with Arabic query...")
    req = ChatRequest(question="ما هي عقوبة السرقة؟", top_k=5, rag_type="agentic", history=[])
    res = chat(req)
    print("Success:", res)
except Exception as e:
    print("\n🚨 ACTUAL ERROR CAUGHT:")
    traceback.print_exc()
