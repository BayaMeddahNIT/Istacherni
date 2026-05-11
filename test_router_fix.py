import json
from agentic_rag.agentic_agent import _run_router

def test_router_hardening():
    queries = [
        "كيف أستخرج سجل تجاري؟",
        "كيف أتحصل على رخصة تجارية؟",
        "ما الفرق بين SARL و SPA؟"
    ]
    
    print("\n--- Testing Router Hardening (No Chinese Leakage) ---")
    for q in queries:
        print(f"\nQuery: {q}")
        result = _run_router(q, [])
        print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        rewritten = result.get("rewritten_query", "")
        if rewritten and any('\u4e00' <= char <= '\u9fff' for char in rewritten):
            print("❌ FAILED: Chinese characters detected!")
        else:
            print("✅ PASSED: No Chinese characters.")

if __name__ == "__main__":
    test_router_hardening()
