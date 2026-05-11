"""
test_graph_rag_local.py
-----------------------
End-to-end test for the 100% local Graph RAG pipeline.

  Embedding  → BAAI/bge-m3  (local)
  Retrieval  → NetworkX Knowledge Graph  (local)
  Generation → Ollama LLM  (local, offline)

No Gemini. No OpenAI. No external APIs whatsoever.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from graph_rag_local.graph_retriever import graph_retrieve, expand_acronyms
    from graph_rag_local.graph_generator  import graph_generate
    from graph_rag_local.local_llm        import LOCAL_LLM_MODEL, is_ollama_running
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

TEST_QUERIES = [
    # Penal Law (قانون العقوبات)
    "ما هي عقوبة الغش في بيع السلع؟",
    #"ما هي عقوبة إصدار شيك بدون رصيد؟",
    #"هل الاحتيال الإلكتروني (online scam) جريمة؟",

    # Civil Law (القانون المدني)
    #"هل العقد الشفهي ملزم قانونياً؟",
    #"ما هي شروط صحة العقد؟",
    #"ما هي القوة القاهرة في العقود؟",

    # Administrative Law (القانون الإداري)
    #"كيف أتحصل على رخصة تجارية؟",
    #"هل يمكن مقاضاة إدارة عمومية؟",
    #"كيف أستخرج سجل تجاري؟",

    # Labor Law (قانون العمل)
    #"ما هي حقوق العامل في الجزائر؟",
    #"هل يمكن طردي بدون سبب؟",
    #"ما هي ساعات العمل القانونية؟",

    # Commercial Law (القانون التجاري)
    #"كيف أفتح شركة في الجزائر؟",
    "ما الفرق بين SARL و SPA؟",
    #"ما هي إجراءات الإفلاس؟",
    #"كيف أتحصل على رخصة تجارية؟",

    #"شخص باعني منتج مزور، هل هذا يعتبر جريمة؟",
    #"هل إخفاء الأرباح للتهرب الضريبي جريمة؟",
    #"هل يمكن تعديل العقد بعد توقيعه؟",
    #"هل العقد الشفهي ملزم قانونياً؟",
    #"كيف يمكنني مقاضاة شركة أجنبية؟",
    #"ما هي الضرائب على الشركات؟",
    #"هل يمكن معاقبة شخص على المنافسة غير الشريفة؟",
    #"أعطني تفاصيل حول قوانين حماية البيانات الشخصية في الجزائر.",
    "ما هي أركان جريمة خيانة الأمانة في السياق التجاري؟",
    #"ما هي المسؤولية الجزائية لصاحب العمل في حالة حدوث وفاة بسبب انعدام شروط السلامة؟",
    #"ما هي الآثار القانونية المترتبة عن التسوية القضائية للشركات؟"
    #"كيف أستخرج شهادة الميلاد رقم 12؟",
    

    
]


def run_test():
    print("\n" + "=" * 70)
    print("   LOCAL GRAPH RAG — 100% Offline End-to-End Test")
    print(f"   LLM: {LOCAL_LLM_MODEL}  |  Embeddings: BAAI/bge-m3")
    print("=" * 70)

    if not is_ollama_running():
        print("\n❌ Ollama is not running. Please start it:")
        print("   1. Install: https://ollama.com/download")
        print("   2. Run in terminal: ollama serve")
        print(f"  3. Pull model: ollama pull {LOCAL_LLM_MODEL}")
        sys.exit(1)

    for q in TEST_QUERIES:
        print(f"\nQUERY: {q}")
        print("-" * 60)

        # ── Step 1: Semantic Graph Retrieval ──────────────────────────
        print("🔎 Step 1 — Semantic Graph Retrieval...")
        # Sanitize query: replace French acronyms before hitting retriever AND generator
        clean_q = expand_acronyms(q)
        try:
            results = graph_retrieve(clean_q, top_k=7)
        except Exception as e:
            print(f"   ❌ Retrieval Error: {e}")
            continue

        if not results:
            print("   ⚠️  No relevant articles found.")
            continue

        for r in results:
            print(f"   ├─ [{r['law_name']}] المادة {r['article_number']}  "
                  f"score={r['graph_score']:.3f} | {r.get('title','')[:40]}")

        # ── Step 2: Local LLM Generation ──────────────────────────────
        print(f"\n🤖 Step 2 — Local Generation ({LOCAL_LLM_MODEL})...")
        print("   ⏳ Generating answer (streaming)...")
        try:
            print("\n✅ Answer:")
            # stream=True: prints tokens as they arrive to prevent timeouts/user interrupts
            answer = graph_generate(clean_q, results, stream=True)
            print() # Ensure newline after streaming
        except ConnectionError as e:
            print(f"\n   ❌ Connection Error: {e}")
            break
        except RuntimeError as e:
            print(f"\n   ❌ Model Error: {e}")
            break
        except Exception as e:
            print(f"\n   ❌ Generation Error: {e}")
            continue

        print("\n" + "=" * 70)

    print("\n🏁 Test complete.\n")


if __name__ == "__main__":
    run_test()
