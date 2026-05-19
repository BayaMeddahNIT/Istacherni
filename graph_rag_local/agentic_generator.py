"""
agentic_generator.py
--------------------
Multi-agent generation logic for Istacherni.
Uses a "Draft-and-Review" architecture to reduce latency while maintaining high legal precision.

Architecture:
1. Junior Clerk (qwen2:0.5b): Rapidly synthesizes a draft from retrieved context (~20-30s).
2. Senior Partner (qwen2:7b): Reviews, corrects, and verifies citations (~2-3 mins).
"""

import time
from typing import List, Dict
from graph_rag_local.local_llm import local_generate

DRAFTER_MODEL = "qwen2:0.5b"
REVIEWER_MODEL = "qwen2:7b"

# ── Prompts ────────────────────────────────────────────────────────────────────

DRAFT_PROMPT_TEMPLATE = """أنت مساعد قانوني (Junior Clerk). مهمتك هي كتابة مسودة أولية سريعة ومنظمة للإجابة على سؤال المستخدم بناءً على نصوص القانون المرفقة.
ركز على الهيكل والوضوح. لا تقلق كثيراً بشأن الصياغة النهائية، بل ركز على استخراج النقاط الأساسية.

السؤال:
{question}

السياق القانوني المستخرج (أهم 3 مواد):
{context}

اكتب الإجابة الآن بشكل منظم (نقاط أو فقرات):
"""

REVIEW_PROMPT_TEMPLATE = """أنت قاضٍ جزائري خبير (Senior Partner). أمامك مسودة إجابة قانونية أعدها مساعدك، ومعك النصوص القانونية الأصلية.
مهمتك هي مراجعة المسودة وتصحيحها لضمان الدقة القانونية التامة والاحترافية.

يجب عليك:
1. التأكد من أن كل استشهاد برقم المادة (مثلاً المادة 59) صحيح وموجود في السياق المرفق.
2. تصحيح أي أخطاء في التفسير القانوني.
3. إضافة أي تفاصيل جوهرية سقطت من المسودة.
4. التأكد من أن اللغة العربية سليمة وقانونية رصينة.

السياق القانوني الأصلي:
{context}

المسودة المطلوب مراجعتها:
{draft}

الإجابة النهائية المصححة (يرجى البدء بالإجابة مباشرة):
"""

# ── Core Agentic Logic ─────────────────────────────────────────────────────────

def agentic_generate(question: str, articles: List[Dict], stream: bool = True) -> Dict:
    """
    Step 1: Draft with 0.5b
    Step 2: Review with 7b
    Returns: A dictionary with draft, final_answer, and timing metadata.
    """
    
    # 1. Build Context String (Tiered)
    context_str = ""
    for i, art in enumerate(articles[:10]):
        if i < 3:
            # High-priority context for top 3
            context_str += f"--- {art['law_name']} - المادة {art['article_number']} ---\n"
            context_str += f"النص: {art['text_original']}\n"
            if art.get('text_explanation'):
                context_str += f"شرح: {art['text_explanation']}\n"
        else:
            # Low-priority summaries for the rest
            context_str += f"- {art['law_name']} (المادة {art['article_number']}): {art['summary']}\n"

    # --- PHASE 1: DRAFTING ---
    print(f"\n[AgenticGen] PHASE 1: Drafting with {DRAFTER_MODEL}...")
    draft_start = time.time()
    draft_prompt = DRAFT_PROMPT_TEMPLATE.format(question=question, context=context_str)
    
    # We use a lower temperature for the draft to keep it grounded
    draft = local_generate(draft_prompt, model=DRAFTER_MODEL, temperature=0.1, stream=stream)
    draft_time = time.time() - draft_start
    print(f"\n[AgenticGen] Draft complete in {draft_time:.2f}s")

    # --- PHASE 2: REVIEWING ---
    print(f"\n[AgenticGen] PHASE 2: Reviewing with {REVIEWER_MODEL}...")
    review_start = time.time()
    review_prompt = REVIEW_PROMPT_TEMPLATE.format(context=context_str, draft=draft)
    
    # The review phase is the "Golden" output
    final_answer = local_generate(review_prompt, model=REVIEWER_MODEL, temperature=0.1, stream=stream)
    review_time = time.time() - review_start
    print(f"\n[AgenticGen] Review complete in {review_time:.2f}s")

    total_time = draft_time + review_time
    
    return {
        "draft": draft,
        "final_answer": final_answer,
        "metadata": {
            "draft_time": draft_time,
            "review_time": review_time,
            "total_time": total_time,
            "models": {"drafter": DRAFTER_MODEL, "reviewer": REVIEWER_MODEL}
        }
    }

if __name__ == "__main__":
    # Test on a single query
    from graph_rag_local.graph_retriever import graph_retrieve
    
    test_q = "ما هي شروط صحة العقد؟"
    print(f"Testing Agentic Generation for: {test_q}")
    
    arts = graph_retrieve(test_q)
    result = agentic_generate(test_q, arts, stream=True)
    
    print("\n" + "="*50)
    print("FINAL AGENTIC RESULT")
    print("="*50)
    print(f"Total Time: {result['metadata']['total_time']:.2f}s")
    print(f"Drafter ({DRAFTER_MODEL}): {result['metadata']['draft_time']:.2f}s")
    print(f"Reviewer ({REVIEWER_MODEL}): {result['metadata']['review_time']:.2f}s")
    # print(f"\nFinal Answer:\n{result['final_answer']}")
