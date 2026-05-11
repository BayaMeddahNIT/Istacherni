import os
import json
from graph_rag_local.local_llm import local_generate

_CLASSIFIER_PROMPT = """أنت خبير قانوني جزائري. مهمتك هي تحليل سؤال المستخدم بدقة.
تذكر دائماً أن "شروط صحة العقد" (الرضا، الأهلية، المحل، السبب) تنتمي بالأساس إلى القانون المدني (Civil Law).

قم بتحليل السؤال التالي وارجع النتيجة بصيغة JSON فقط، وبنفس البنية التالية (بدون أي نص خارج الـ JSON):
{
  "domain": "الفرع القانوني الدقيق (اختر من: Civil Law, Penal Code, Labor Law, Commercial Law, Code of Civil and Administrative Procedures)",
  "law_type": "استخدم إما 'substantive' للأحكام الموضوعية (مثل شروط وحقوق العقد) أو 'procedural' للإجراءات والدليل، أو 'both'",
  "keywords": ["المصطلحات الأساسية"],
  "synonyms": ["مرادفات دقيقة موجودة في نصوص القانون المدني مثل: الرضا، الأهلية، المحل، السبب، التراضي، البطلان"],
  "amplified_query": "صياغة قانونية مفصلة للسؤال",
  "hyde_document": "فقرة نظرية قصيرة تجيب عن السؤال بأسلوب أكاديمي يشمل الأركان والشروط للبحث الدلالي",
  "negative_concepts": ["مفاهيم مشابهة مستبعدة"]
}

يجب أن يكون المخرج JSON صالحاً فقط. لا تكتب ```json ولا أي كلمة أخرى.

السؤال:
{question}
"""

# ── Fast Rule-Based Synonym Expansion ──────────────────────────────────────────
_SYNONYM_MAP = {
    "شروط صحة العقد": ["الرضا", "الأهلية", "المحل", "السبب", "تطابق الإرادتين", "أركان العقد"],
    "أركان العقد": ["الرضا", "الأهلية", "المحل", "السبب"],
    "بطلان العقد": ["إبطال", "البطلان المطلق", "البطلان النسبي", "دعوى البطلان", "الغلط", "التدليس", "الإكراه"],
    "الطلاق": ["فك الرابطة الزوجية", "التطليق", "الخلع", "النشوز"],
}

def _fast_synonym_expansion(query: str) -> list[str]:
    synonyms = []
    for key, terms in _SYNONYM_MAP.items():
        if key in query:
            synonyms.extend(terms)
    return list(set(synonyms))

def classify_query(question: str) -> dict:
    mode = os.getenv("QUERY_CLASSIFIER", "auto").lower() # "auto", "camelbert", "qwen2"
    
    # Fast path: Pure CamelBERT routing
    if mode == "camelbert":
        from graph_rag_local.camelbert_classifier import get_camelbert_domain
        threshold = float(os.getenv("CAMELBERT_MIN_CONFIDENCE", "0.60"))
        domain = get_camelbert_domain(question, threshold)
        synonyms = _fast_synonym_expansion(question)
        return {
            "domain": domain,
            "law_type": "both",
            "keywords": synonyms,  # Inject fast synonyms here
            "synonyms": synonyms,
            "amplified_query": question,
            "hyde_document": "",
            "negative_concepts": []
        }
        
    # Hybrid path: CamelBERT for domain, LLM for the rest
    if mode == "auto":
        from graph_rag_local.camelbert_classifier import get_camelbert_domain
        threshold = float(os.getenv("CAMELBERT_MIN_CONFIDENCE", "0.60"))
        fast_domain = get_camelbert_domain(question, threshold)

    prompt = _CLASSIFIER_PROMPT.replace("{question}", question)
    raw_response = local_generate(prompt, temperature=0.1)

    try:
        # Clean potential markdown wrapping
        text = raw_response.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        parsed = json.loads(text.strip())
        
        # Override domain if auto mode is on and CamelBERT is confident
        domain = fast_domain if (mode == "auto" and fast_domain) else parsed.get("domain", "")
        
        return {
            "domain": domain,
            "law_type": parsed.get("law_type", "both"),
            "keywords": parsed.get("keywords", []),
            "synonyms": parsed.get("synonyms", []),
            "amplified_query": parsed.get("amplified_query", question),
            "hyde_document": parsed.get("hyde_document", ""),
            "negative_concepts": parsed.get("negative_concepts", [])
        }
    except json.JSONDecodeError:
        print(f"[QueryClassifier] Failed to parse JSON. Raw output: {raw_response}")
        domain = fast_domain if (mode == "auto" and fast_domain) else ""
        return {
            "domain": domain,
            "law_type": "both",
            "keywords": [],
            "synonyms": [],
            "amplified_query": question,
            "hyde_document": "",
            "negative_concepts": []
        }

if __name__ == "__main__":
    test_q = "ما هي شروط صحة العقد؟"
    res = classify_query(test_q)
    print("Classification Result:")
    print(json.dumps(res, indent=2, ensure_ascii=False))
