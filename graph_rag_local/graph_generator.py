"""
graph_generator.py
------------------
Generation module for the fully-local Graph RAG pipeline.
Takes a question + graph-retrieved articles → Local LLM (via Ollama) → Arabic answer.

Zero external API calls. Works completely offline after Ollama setup.
"""

from graph_rag_local.local_llm import local_generate, LOCAL_LLM_MODEL
import re

# ── Context pollution blacklist ─────────────────────────────────────────────
# Fix A: Generic scope/definitions articles that are semantically adjacent to
# every query but contain zero specific legal conclusions. They occupy top-3
# context slots without contributing any legal value.
# Keyed as (law_name_fragment, article_number) for resilient matching.
_CONTEXT_BLACKLIST = {
    ("قانون العقوبات",                                        "3"),   # scope article
    ("قانون الوقاية من الجرائم المتصلة بتكنولوجيات",         "2"),   # cybercrime definitions
    ("قانون الوقاية من الجرائم المتصلة بتكنولوجيات",         "3"),   # cybercrime scope
    ("القانون المدني",                                         "1"),   # general provisions
    ("القانون التجاري",                                        "1"),   # general provisions
}

def _is_blacklisted(art: dict) -> bool:
    """Return True if the article is a known context-polluting generic article."""
    law = art.get("law_name", "")
    num = str(art.get("article_number", ""))
    return any(
        frag in law and num == num_key
        for frag, num_key in _CONTEXT_BLACKLIST
    )

# ── Arabic output sanitizer ─────────────────────────────────────────────────
# Fix D: The 7B model occasionally bleeds into CJK (Chinese/Japanese/Korean)
# character ranges under high context pressure. Replace any such bleed with a
# clean Arabic placeholder so the output remains professionally usable.
_CJK_RANGE = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+')

def _sanitize_arabic(text: str) -> str:
    """Strip any CJK character sequences from model output."""
    return _CJK_RANGE.sub('[...]‏', text)

# ── System instruction ─────────────────────────────────────────────────────────
# Phase 3.1: Structured Chain-of-Thought Citation Prompt.
# Forces the LLM through a 4-step reasoning chain before generating an answer,
# directly attacking F2 (Legal Conclusion Errors) and hallucination by separating
# issue identification from article matching from conclusion drawing.
_SYSTEM = """أنت مستشار قانوني جزائري متخصص. أجب حصراً بناءً على المواد المرفقة.
**قاعدة الأولوية:** ابدأ تحليلك دائماً من المادة [1] باعتبارها الأعلى صلة. لا تتجاهل المادة الأولى لصالح مادة لاحقة بسبب تشابه المصطلحات وحده — الترتيب يعكس الصلة الفعلية بالسؤال.
اكتفِ بهذا الهيكل المختصر:
**1. التحليل والمواد:** حدد المسألة والمواد المنطبقة [المادة X].
**2. الحكم:** النتيجة القانونية المباشرة.
**3. تنبيه:** اذكر أي نقص في التغطية القانونية (إن وجد).
لا تستخدم مواد غير مباشرة. الإجابة باللغة العربية حصراً. يمنع استخدام أي لغة أخرى (مثل الصينية أو الإنجليزية)."""

# ── Context builder ────────────────────────────────────────────────────────────
def _build_context(retrieved: list[dict], max_articles_full: int = 3) -> str:
    """
    Builds the prompt context from retrieved articles.
    Strictly tiers the context: full text for top-3 articles, only summaries for the rest,
    to prevent 'lost in the middle' hallucinations and context saturation.

    Phase 3.2: Injects a multi-domain disambiguation warning when retrieved articles
    span more than one law domain (e.g., Civil + Commercial mix), preventing the LLM
    from conflating partnership rules from different legal codes.
    """
    if not retrieved:
        return "لا توجد مواد قانونية ذات صلة."

    # Disambiguation warning: if articles come from multiple law domains,
    # alert the LLM to use only the most directly relevant ones.
    domains_found = set(a.get("law_domain", "") for a in retrieved if a.get("law_domain"))
    parts = []
    if len(domains_found) > 1:
        parts.append(
            "⚠️ تحذير — مواد من قوانين مختلفة: المواد أدناه مستمدة من مجالات قانونية متعددة "
            f"({' / '.join(domains_found)}). "
            "استخدم فقط المواد المنطبقة مباشرة على السؤال المطروح، وتجاهل المواد الأخرى."
        )
    # Fix A: Filter blacklisted generic scope articles BEFORE building context.
    # Replace each removed slot with the next highest-ranked non-blacklisted article.
    filtered = [a for a in retrieved if not _is_blacklisted(a)]
    if len(filtered) < len(retrieved):
        removed = len(retrieved) - len(filtered)
        print(f"[Generator] Blacklist filtered {removed} generic article(s) from context.")
    # Use filtered list — maintains original rank order, just without noise articles.
    retrieved = filtered if filtered else retrieved  # safety: never return empty

    for i, art in enumerate(retrieved, 1):
        is_top_priority = (i <= max_articles_full)
        
        title = f" ({art['title']})" if art.get("title") else ""
        header = f"━━━ المادة {i}: 【{art['law_name']} — المادة {art['article_number']}】{title} ━━━"

        sections = [header]

        # Summary — always include as it anchors the article concept
        summary = art.get("summary", "").strip()
        if summary:
            sections.append(f"📌 الملخص:\n{summary}")

        if is_top_priority:
            # Explanation
            explanation = art.get("text_explanation", "").strip()
            if explanation:
                sections.append(f"📖 الشرح والتعليل:\n{explanation}")

            # Original text — ONLY for top priority
            text = art.get("text_original", "نص غير متوفر").strip()
            sections.append(f"⚖️ النص الأصلي للمادة:\n{text}")

            # Metadata
            if art.get("legal_conditions_summary"):
                sections.append(f"📋 الشروط القانونية: {art['legal_conditions_summary']}")
            if art.get("penalties_summary"):
                sections.append(f"⚠️ العقوبات: {art['penalties_summary']}")

        parts.append("\n".join(sections))
    return "\n\n---\n\n".join(parts)

def _estimate_max_tokens(question: str, retrieved: list[dict]) -> int:
    """Dynamically set max_tokens to reduce latency."""
    unique_laws = len(set(a.get("law_name", "") for a in retrieved if a.get("law_name")))
    if unique_laws >= 2:
        return 768  # Multi-code synthesis
    if len(retrieved) <= 2 and len(question) < 60:
        return 256   # Simple factual lookup
    return 400       # Optimized default



# ── Main generation function ───────────────────────────────────────────────────
def graph_generate(
    question: str,
    retrieved: list[dict],
    chat_history: list[dict] = None,
    stream: bool = True,
    model: str = None,
) -> str:
    """
    Generate a legal answer using a fully local LLM (Ollama).

    Args:
        question:     The user's legal question (Arabic).
        retrieved:    List of article dicts from graph_retrieve().
        chat_history: Optional list of prior messages [{role, content}].
        stream:       If True, stream tokens to stdout in real time.

    Returns:
        The generated Arabic answer as a string.
    """
    if not retrieved:
        return "عذراً، لم أجد مواد قانونية كافية للإجابة على سؤالك."

    # ── CPU-Optimized Corrective RAG Gate (Score-Based) ───────────────────────
    # If the top retrieved article has a very low RRF score, the retrieval likely failed.
    # We abort early with an honest fallback to prevent the LLM from hallucinating.
    MIN_RELEVANCE_SCORE = 0.08  # Adjusted for new RRF+PPR scaling
    top_score = max((r.get("graph_score", 0) for r in retrieved), default=0)
    
    if top_score < MIN_RELEVANCE_SCORE:
        return (
            "لم أتمكن من العثور على نصوص قانونية ذات صلة كافية بسؤالك في قاعدة بياناتي. "
            "يُرجى إعادة صياغة السؤال أو التواصل مع محامٍ متخصص للحصول على استشارة دقيقة."
        )

    # ── Python-side Procedural Query Guard ────────────────────────────────────
    # This check runs BEFORE the LLM, making it 100% deterministic.
    _PROCEDURAL_TRIGGERS = {
        "كيف", "كيفية", "طريقة", "خطوات", "إجراءات",
        "استخرج", "أستخرج", "أسجل", "أتحصل", "أحصل"
    }

    query_lower = question
    is_procedural_query = any(t in query_lower for t in _PROCEDURAL_TRIGGERS)

    if is_procedural_query:
        # Pass to LLM with specialized procedural prompt.
        # Old code returned hardcoded response violating the Arabic-only system prompt.
        procedural_context = _build_context(retrieved)
        procedural_prompt = (
            f"{_SYSTEM}\n\n"
            "تنبيه خاص: السؤال يتعلق بإجراءات أو خطوات. "
            "اعرض فقط ما تنص عليه المواد القانونية المسترجعة من التزامات وشروط قانونية. "
            "لا تخترع خطوات إدارية غير مذكورة صراحةً في النصوص. "
            "إذا لم تتضمن النصوص إجراءات، فاذكر الإطار القانوني فقط.\n\n"
            f"=== المواد القانونية المسترجعة ===\n\n{procedural_context}\n\n"
            f"=== سؤال المستخدم ===\n\n{question}\n\n"
            "=== الإجابة ==="
        )
        return _sanitize_arabic(
            local_generate(procedural_prompt, temperature=0.0, stream=stream, model=model)
        )
    # ── End of Procedural Guard ────────────────────────────────────────────────
    context = _build_context(retrieved, max_articles_full=3)
    max_tok = _estimate_max_tokens(question, retrieved)

    chat_text = ""
    if chat_history:
        lines = []
        for msg in chat_history:
            role = "المستخدم" if msg.get("role") == "user" else "المساعد"
            lines.append(f"{role}: {msg.get('content', '')}")
        chat_text = "\n\n=== تاريخ المحادثة ===\n" + "\n".join(lines) + "\n"

    prompt = (
        f"{_SYSTEM}\n\n"
        f"=== المواد القانونية المسترجعة ===\n\n"
        f"{context}\n\n"
        f"{chat_text}"
        f"=== سؤال المستخدم ===\n\n"
        f"{question}\n\n"
        f"=== الإجابة ==="
    )

    return _sanitize_arabic(
        local_generate(prompt, temperature=0.0, max_tokens=max_tok, stream=stream, model=model)
    )


if __name__ == "__main__":
    from graph_rag_local.graph_retriever import graph_retrieve

    q = "ما هي عقوبة السرقة في القانون الجزائري؟"
    print(f"Model: {LOCAL_LLM_MODEL}\n")
    articles = graph_retrieve(q, top_k=3)
    print("Generating answer (streaming)...\n")
    answer = graph_generate(q, articles, stream=True)
    print(f"\nDone. ({len(answer)} chars)")