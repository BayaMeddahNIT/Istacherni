"""
graph_generator.py
------------------
Generation module for the fully-local Graph RAG pipeline.
Takes a question + graph-retrieved articles → Local LLM (via Ollama) → Arabic answer.

Zero external API calls. Works completely offline after Ollama setup.
"""

from graph_rag_local.local_llm import local_generate, LOCAL_LLM_MODEL

# ── System instruction ─────────────────────────────────────────────────────────
_SYSTEM = """أنت مستشار قانوني جزائري خبير ودقيق. مهمتك الوحيدة هي الإجابة على سؤال المستخدم بناءً حصراً على النصوص القانونية ذات الصلة المرفقة.

تعليمات إلزامية وصارمة:
1. اقرأ جميع المواد المُقدَّمة أولاً، ثم قيّم مدى صلتها بسؤال المستخدم.
2. 🚫 إذا كانت مادةٌ ما لا علاقة لها المباشرة بسؤال المستخدم، تجاهلها تماماً ولا تذكرها في إجابتك ولو بكلمة واحدة.
3. ✅ استخدم فقط المواد التي تجيب بشكل مباشر على السؤال المطروح.
4. اكتب إجابة منظمة بالنقاط تذكر فيها كل حكم قانوني مع رقم المادة التي جاء منها.
5. الإجابة باللغة العربية الفصحى فقط — يُمنع منعاً باتاً استخدام أي كلمات أو رموز أجنبية.
6. كن مختصراً ودقيقاً — لا تكرر المعلومات ولا تطوّل دون فائدة.
7. استخدم مصطلح "صاحب العمل" أو "المُستخدِم" للإشارة إلى رب العمل (لا تستخدم رموزاً أجنبية مثل 雇主).
8. بعد كل جملة تستند إلى نص قانوني، اذكر رقم المادة بين قوسين مربعين: [المادة X]. إذا لم تستطع نسب الجملة لمادة محددة مُقدَّمة في السياق، لا تكتبها.
9. يُمنع منعاً باتاً استخدام أي لغة أخرى غير اللغة العربية. لا تستخدم أي رموز أو كلمات أجنبية أو صينية.
10. إذا كانت النصوص المسترجعة لا تذكر صراحة تفاصيل معينة (مثل مسؤولية الشركاء)، يُمنع منعاً باتاً استنتاجها أو تأليفها. اكتفِ بما هو مكتوب فقط."""

# ── Context builder ────────────────────────────────────────────────────────────
def _build_context(retrieved: list[dict]) -> str:
    if not retrieved:
        return "لا توجد مواد قانونية ذات صلة."
    parts = []
    for i, art in enumerate(retrieved, 1):
        title = f" ({art['title']})" if art.get("title") else ""
        header = f"━━━ المادة {i}: 【{art['law_name']} — المادة {art['article_number']}】{title} ━━━"

        sections = [header]

        # Summary — first priority for conceptual understanding
        summary = art.get("summary", "").strip()
        if summary:
            sections.append(f"📌 الملخص:\n{summary}")

        # Explanation — second priority for legal reasoning
        explanation = art.get("text_explanation", "").strip()
        if explanation:
            sections.append(f"📖 الشرح والتعليل:\n{explanation}")

        # Original text — third priority for legal citation
        text = art.get("text_original", "نص غير متوفر").strip()
        sections.append(f"⚖️ النص الأصلي للمادة:\n{text}")

        # Keywords — contextual anchors
        keywords = art.get("keywords", [])
        if keywords:
            sections.append(f"🔑 الكلمات المفتاحية: {' | '.join(keywords)}")

        # Optional penalty/conditions metadata
        if art.get("legal_conditions_summary"):
            sections.append(f"📋 الشروط القانونية: {art['legal_conditions_summary']}")
        if art.get("penalties_summary"):
            sections.append(f"⚠️ العقوبات: {art['penalties_summary']}")

        parts.append("\n".join(sections))
    return "\n\n---\n\n".join(parts)



# ── Main generation function ───────────────────────────────────────────────────
def graph_generate(
    question: str,
    retrieved: list[dict],
    chat_history: list[dict] = None,
    stream: bool = False,
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
        # Build an honest summary from the retrieved articles
        legal_facts = []
        for a in retrieved[:4]:
            art_num = a.get("article_number", "")
            law = a.get("law_name", "")
            summary = (a.get("summary") or a.get("text_original") or "")[:200]
            if art_num and summary:
                legal_facts.append(f"• المادة {art_num} ({law}): {summary.strip()}")

        facts_text = "\n".join(legal_facts)
        return (
            "النصوص القانونية المسترجعة تحدد الإطار القانوني لهذا الموضوع، "
            "ولكنها لا تحتوي على الإجراءات الإدارية التفصيلية (الأوراق، الرسوم، المكاتب).\n\n"
            "**ما ينص عليه القانون:**\n"
            f"{facts_text}\n\n"
            "للحصول على الإجراءات الإدارية المحددة، يُرجى التواصل مباشرةً مع الجهة المختصة "
            "(مثل المركز الوطني للسجل التجاري CNRC أو الجهة الإدارية ذات الصلة)."
        )
    # ── End of Procedural Guard ────────────────────────────────────────────────
    context = _build_context(retrieved)

    chat_text = ""
    if chat_history:
        lines = []
        for msg in chat_history:
            role = "المستخدم" if msg.get("role") == "user" else "المساعد"
            lines.append(f"{role}: {msg.get('content', '')}")
        chat_text = "\n\n=== تاريخ المحادثة ===\n" + "\n".join(lines) + "\n"

    # ── Last-Word Anti-Hallucination Gate + Chain-of-Thought ──────────────────
    # This block is placed at the VERY BOTTOM of the prompt, immediately before
    # the answer token. Recency bias ensures the model reads this last.
    # The CoT line forces it to audit the context before writing a single word.
    anti_hallucination_gate = (
        "[تنبيه صارم وحاسم قبل الإجابة]:\n"
        "راجع المواد المسترجعة أعلاه الآن. هل تحتوي على خطوات إدارية "
        "(أوراق مطلوبة، رسوم، مكاتب، إجراءات تطبيقية)؟\n\n"
        "ابدأ إجابتك بجملة تحليل بين قوسين:\n"
        "- إذا كانت المواد تحتوي على خطوات: اكتب (التحليل: السياق يحتوي على [...]) ثم اعرض المعلومات القانونية.\n"
        "- إذا كانت المواد لا تحتوي على خطوات إدارية: يجب أن تكتب الجملة التالية حرفياً ثم تتوقف:\n"
        '  "(التحليل: النصوص المسترجعة لا تحتوي على إجراءات إدارية.)\n'
        "  النصوص القانونية المسترجعة تنص على الالتزام القانوني فقط:\n"
        "  - [اذكر ما تنص عليه المادة 19 ومواد أخرى صراحةً بدون إضافة خطوات من عندك]\"\n"
    )

    # ONLY attach the anti-hallucination gate for procedural queries.
    # For pure legal queries (e.g. "what are worker rights?"), omit it entirely
    # so the model produces a clean bulleted answer without the (التحليل...) header.
    if is_procedural_query:
        gate_block = f"{anti_hallucination_gate}\n\n"
    else:
        gate_block = ""

    prompt = (
        f"{_SYSTEM}\n\n"
        f"=== المواد القانونية المسترجعة ===\n\n"
        f"{context}\n\n"
        f"{chat_text}"
        f"=== سؤال المستخدم ===\n\n"
        f"{question}\n\n"
        f"{gate_block}"
        f"=== الإجابة ==="
    )

    return local_generate(prompt, temperature=0.0, stream=stream)


if __name__ == "__main__":
    from graph_rag_local.graph_retriever import graph_retrieve

    q = "ما هي عقوبة السرقة في القانون الجزائري؟"
    print(f"Model: {LOCAL_LLM_MODEL}\n")
    articles = graph_retrieve(q, top_k=3)
    print("Generating answer (streaming)...\n")
    answer = graph_generate(q, articles, stream=True)
    print(f"\nDone. ({len(answer)} chars)")
