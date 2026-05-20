import sys
import io

# Force stdout to utf-8 so Arabic console output doesn't crash
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from graph_rag_local.graph_retriever import graph_retrieve
from graph_rag_local.graph_generator import graph_generate

test_questions = [
    "ما عقوبة النصب والاحتيال في القانون الجزائري؟",
    "ما هي شروط صحة العقد؟",
    "كيف أسجل شركة ذات مسؤولية محدودة؟",
    "ما حقوق العامل عند الفصل التعسفي؟",
    "ما هي عقوبة السرقة بالإكراه؟",
]

with open("spot_check_output.txt", "w", encoding="utf-8") as f:
    for i, q in enumerate(test_questions, 1):
        print(f"\n[SpotCheck] Query {i}/{len(test_questions)}: {q}")
        f.write(f"\n{'='*60}\n")
        f.write(f"Q{i}: {q}\n")
        try:
            # Step 1: Retrieve
            print(f"[SpotCheck] Retrieving...")
            articles = graph_retrieve(q, top_k=5)
            retrieved_refs = [(a.get('article_number'), a.get('law_name')) for a in articles]
            print(f"[SpotCheck] Retrieved: {retrieved_refs}")
            f.write(f"Retrieved: {retrieved_refs}\n")
            f.flush()
            
            # Step 2: Generate
            print(f"[SpotCheck] Generating...")
            answer = graph_generate(q, articles)
            print(f"[SpotCheck] Answer received ({len(answer)} chars).")
            f.write(f"Answer:\n{answer}\n")
            f.flush()
            print(f"[SpotCheck] Done with Q{i}.")
        except Exception as e:
            import traceback
            err_msg = f"ERROR on query {i} '{q}': {e}\n{traceback.format_exc()}"
            print(err_msg)
            f.write(err_msg + "\n")
            f.flush()
