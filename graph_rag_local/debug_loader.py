"""
debug_loader.py
---------------
Diagnostic script to verify that ALL local components are working:
  1. BGE-M3 embedding model
  2. Ollama local LLM server
  3. Available Ollama models
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Section 1: Embedding model ────────────────────────────────────────────────
print("=" * 55)
print("  Part 1: BGE-M3 Embedding Model")
print("=" * 55)

try:
    from graph_rag_local.embeddings import embed_text, MODEL_NAME, DEVICE

    print(f"Model : {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    test_text = "ما هي عقوبة السرقة في القانون الجزائري؟"
    emb = embed_text(test_text, show_progress=False)
    print(f"Embedding shape : {emb.shape}")
    print("✅ BGE-M3 is working.\n")

except ImportError:
    print("❌ Missing: pip install sentence-transformers torch\n")
except Exception as e:
    print(f"❌ Error: {e}\n")


# ── Section 2: Ollama local LLM ───────────────────────────────────────────────
print("=" * 55)
print("  Part 2: Ollama Local LLM")
print("=" * 55)

try:
    from graph_rag_local.local_llm import (
        is_ollama_running, list_available_models, local_generate, LOCAL_LLM_MODEL, OLLAMA_BASE_URL
    )

    print(f"Ollama URL   : {OLLAMA_BASE_URL}")
    print(f"Target model : {LOCAL_LLM_MODEL}")

    if not is_ollama_running():
        print("❌ Ollama is NOT running.\n")
        print("Fix:")
        print("  1. Download Ollama: https://ollama.com/download")
        print("  2. Open a terminal and run: ollama serve")
        print("  3. Pull a model:  ollama pull qwen2:7b")
    else:
        print("✅ Ollama server is running.")
        models = list_available_models()
        if models:
            print(f"Installed models: {models}")
        else:
            print("⚠️  No models installed. Run: ollama pull qwen2:7b")

        # Quick generation test
        print("\nRunning a quick generation test...")
        result = local_generate(
            "Say exactly: 'Local LLM is working.' in Arabic.",
            max_tokens=50
        )
        print(f"Response: {result}")
        print("✅ Local LLM is working.\n")

except Exception as e:
    print(f"❌ Error: {e}\n")

print("=" * 55)
print("  Diagnostic complete.")
print("=" * 55)
