"""
agentic_agent.py
----------------
Agentic RAG engine — 100% local brain, local Graph RAG (Offline Only).

Architecture:
  ┌──────────────────────────────────────────────────────────────────┐
  │  User query                                                      │
  │      ↓                                                           │
  │  [1] Acronym Expander  (expand_acronyms)                         │
  │      ↓                                                           │
  │  [2] Intent Classifier (Python rule-based, instant)              │
  │      └─ SUBSTANTIVE/PROCEDURAL → Tool: graph_retrieve            │
  │      ↓                                                           │
  │  [3] Local Agentic Loop  (qwen2:7b via Ollama, JSON-mode)        │
  │      ├─ Model outputs {"tool": "...", "args": {...}}              │
  │      ├─ Python executes graph_retrieve, feeds result back        │
  │      └─ Loop until model outputs {"tool": "final_answer", ...}   │
  │      ↓                                                           │
  │  [4] Final Arabic answer                                         │
  └──────────────────────────────────────────────────────────────────┘

Brain: qwen2:7b via Ollama (offline, private, zero API cost)
Local tool: graph_rag_local.graph_retriever (local vector graph)
"""

OFFLINE_MODE = True  # Toggle for 100% offline operation

import json
import re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from graph_rag_local.graph_retriever import graph_retrieve, expand_acronyms
from graph_rag_local.graph_generator import graph_generate
from graph_rag_local.local_llm import local_generate, LOCAL_LLM_MODEL
from agentic_rag.web_search_tool import web_search

MAX_TOOL_ROUNDS = 3   # Maximum agentic iterations before forcing final answer


# ══════════════════════════════════════════════════════════════════════════════
# ── Intent Classifier ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── Instant CHITCHAT Patterns (O(1) Python-first, zero LLM cost) ──────────────
_CHITCHAT_PATTERNS = {
    # Arabic greetings
    "مرحبا", "مرحباً", "السلام", "السلام عليكم", "صباح الخير", "مساء الخير",
    "هلا", "هاي", "أهلاً", "أهلا", "شكراً", "شكرا", "وداعاً", "مع السلامة",
    # English greetings
    "hello", "hi", "hey", "good morning", "good evening", "thanks", "thank you", "bye",
    # French greetings
    "bonjour", "salut", "bonsoir", "merci", "au revoir", "bonne journée",
}

_CHITCHAT_RESPONSES = {
    "AR": "مرحباً! أنا مستشارك القانوني الجزائري. كيف يمكنني مساعدتك اليوم؟",
    "EN": "Hello! I'm your Algerian legal assistant. How can I help you today?",
    "FR": "Bonjour ! Je suis votre assistant juridique algérien. Comment puis-je vous aider ?",
}

def _detect_language_fast(text: str) -> str:
    """Detect language from script. Fast O(n) check with no LLM."""
    stripped = text.strip()
    arabic_chars = sum(1 for c in stripped if '\u0600' <= c <= '\u06FF')
    if arabic_chars / max(len(stripped), 1) > 0.3:
        return "AR"
    low = stripped.lower()
    # Unambiguous French single words and markers
    french_words = {"bonjour", "salut", "bonsoir", "merci", "quelles", "quel",
                    "comment", "pourquoi", "quelle", "est-ce", "dans", "les",
                    "des", "une", "au revoir", "bonne"}
    if any(w in low.split() or low == w for w in french_words):
        return "FR"
    french_markers = ["quell", "est-ce", "au revoir"]
    if any(m in low for m in french_markers):
        return "FR"
    return "EN"

def _check_instant_chitchat(query: str) -> tuple[bool, str]:
    """Return (is_chitchat, language) in O(1). Runs before any LLM call."""
    normalized = query.strip().lower().rstrip("!؟?.,")
    lang = _detect_language_fast(query)
    if normalized in _CHITCHAT_PATTERNS:
        return True, lang
    # Also check if the entire message is very short and contains a greeting word
    if len(normalized.split()) <= 3:
        for pat in _CHITCHAT_PATTERNS:
            if pat in normalized:
                return True, lang
    return False, lang


_ROUTER_SYSTEM = """You are an intelligent multilingual classifier for Algerian legal queries.
Analyze the user's last message based on the conversation context and output a single JSON object.

Detect the user's language:
- If the message contains Arabic script → "AR"
- If the message is in French → "FR"
- If the message is in English → "EN"

Intent types:
1. CHITCHAT: Greetings, thanks, or off-topic messages.
2. FOLLOW_UP: Question depends entirely on the previous answer (e.g., "explain more", "does this apply to...").
3. SUBSTANTIVE: Legal substance or procedural questions (rights, penalties, contract conditions, steps, registration).

Rules for rewritten_query (CRITICAL — this is the retrieval query):
- ALWAYS write rewritten_query in Arabic, even if the user wrote in French or English.
- Translate the question into Arabic for the Arabic legal database.
- Expand French acronyms (SARL → شركة ذات مسؤولية محدودة, SPA → شركة مساهمة).
- If CHITCHAT, set rewritten_query to null.
- If FOLLOW_UP, resolve pronouns using context history and write the full standalone Arabic query.

Rules for direct_response:
- Only populate this field if intent = CHITCHAT.
- Write the greeting response in the SAME language as detected_language.
- Otherwise set to null.

Output ONLY valid JSON in this exact format:
{
  "detected_language": "AR",
  "intent": "SUBSTANTIVE",
  "rewritten_query": "Arabic search query",
  "direct_response": null,
  "extracted_articles": []
}

EXAMPLES:
User: "How to register a company?" -> {"detected_language": "EN", "intent": "SUBSTANTIVE", "rewritten_query": "كيفية تسجيل شركة في الجزائر؟", "direct_response": null}
User: "مرحبا" -> {"detected_language": "AR", "intent": "CHITCHAT", "rewritten_query": null, "direct_response": "مرحباً! كيف أساعدك؟"}

CRITICAL RULES:
- `detected_language`: Input language (AR, EN, or FR).
- `rewritten_query`: ALWAYS provide the search query in Arabic. 
- NO Chinese/Asian characters allowed. Strictly Arabic.
Expansion Rule: SARL -> شركة ذات مسؤولية محدودة, SPA -> شركة مساهمة.
"""

def _run_router(query: str, history: list[dict], verbose: bool = False) -> dict:
    """Combines intent classification, query rewriting, and entity extraction into a single LLM pass."""
    prompt = _ROUTER_SYSTEM + "\n\n"
    if history:
        prompt += "Conversation context (last 3 messages):\n"
        for msg in history[-3:]:
            if msg.get("role") in ["user", "assistant"]:
                prompt += f"{msg['role']}: {msg['content']}\n"
    
    lang_hint = _detect_language_fast(query)
    prompt += f"\nUser message: {query}\n"
    prompt += f"Language hint: {lang_hint}\n"
    prompt += "\nOutput ONLY valid JSON:\n"

    raw = local_generate(prompt, temperature=0.0, stream=False)
    
    # Extract JSON
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            res = json.loads(match.group())
            # [SECURITY PATCH] Force-strip Chinese characters from rewritten_query
            if res.get("rewritten_query"):
                res["rewritten_query"] = re.sub(r"[\u4e00-\u9fff]+", "", res["rewritten_query"]).strip()
            return res
        except:
            pass
            
    # Fallback if parsing fails
    return {
        "detected_language": _detect_language_fast(query),
        "intent": "SUBSTANTIVE",
        "rewritten_query": query,
        "direct_response": None,
        "extracted_articles": []
    }



# ══════════════════════════════════════════════════════════════════════════════
# ── Tool Executor ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _tool_graph_retrieve(args: dict) -> str:
    """Call local graph retriever and return a JSON string of article summaries."""
    query   = args.get("query", "")
    top_k   = min(int(args.get("top_k", 7)), 10)
    results = graph_retrieve(query, top_k=top_k)

    articles = []
    for r in results:
        articles.append({
            "law_name":       r.get("law_name", ""),
            "article_number": r.get("article_number", ""),
            "title":          r.get("title", ""),
            "text":           (r.get("text_original") or r.get("summary") or "")[:400],
            "score":          round(r.get("graph_score", 0), 3),
        })
    return json.dumps({"tool": "graph_retrieve", "count": len(articles), "articles": articles},
                      ensure_ascii=False)


def _tool_web_search(args: dict) -> str:
    """Call DuckDuckGo scoped to Algerian government domains."""
    query   = args.get("query", "")
    domains = args.get("domains", None)   # None → use default whitelist
    result  = web_search(query, domains)
    return json.dumps({"tool": "web_search", **result}, ensure_ascii=False)


_TOOL_REGISTRY = {
    "graph_retrieve": _tool_graph_retrieve,
    "web_search":     _tool_web_search,
}


def _execute_tool(name: str, args: dict) -> str:
    fn = _TOOL_REGISTRY.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool '{name}'"})
    try:
        return fn(args)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# ── JSON Tool-Call Prompt ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

_AGENT_SYSTEM = """\
أنت محامٍ رقمي متخصص في القانون الجزائري. تعمل في وضع عدم الاتصال (Offline) وتعتمد حصرياً على قاعدة بيانات القوانين المحلية.

الأدوات المتاحة:
1. graph_retrieve(query, top_k=7)  — البحث في قاعدة بيانات المواد القانونية والإجراءات المحلية.
2. final_answer(text)              — إصدار الإجابة النهائية للمستخدم باللغة العربية.

قواعد صارمة:
- أجب دائماً بكائن JSON واحد فقط بهذا الشكل بالضبط:
  {"tool": "اسم_الأداة", "args": {"المفتاح": "القيمة"}}
- لا تكتب أي نص خارج كائن JSON.
- استخدم graph_retrieve للبحث عن المعلومات القانونية أو الإجرائية في قاعدة البيانات المحلية.
- بعد الحصول على نتائج كافية، استخدم final_answer مع نص الإجابة الكاملة.
- لا تخترع معلومات. اعتمد فقط على ما استرجعته من الأدوات.
- يُمنع منعاً باتاً استخدام أي لغة أخرى غير اللغة العربية. لا تستخدم أي رموز أو كلمات أجنبية أو صينية.
"""

def _build_agent_prompt(question: str, history: list[dict], hint_procedural: bool) -> str:
    """Build the full prompt string for the local LLM."""
    hint = '\n[تلميح]: استخدم graph_retrieve للبحث في قاعدة البيانات القانونية.\n'

    turns = ""
    for entry in history:
        role   = entry["role"]   # "tool_call" | "tool_result"
        content = entry["content"]
        if role == "tool_call":
            turns += f"\n[استدعيت]: {content}\n"
        elif role == "tool_result":
            turns += f"\n[نتيجة الأداة]: {content}\n"

    return (
        f"{_AGENT_SYSTEM}\n"
        f"{hint}\n"
        f"سؤال المستخدم: {question}\n"
        f"{turns}\n"
        f"أجب الآن بكائن JSON:"
    )


_LANGUAGE_DIRECTIVE = {
    "AR": "اكتب الإجابة النهائية بالعربية الفصحى فقط. لا تستخدم أي لغة أجنبية.",
    "EN": "The user wrote in English. You retrieved Arabic law — that is correct. Now synthesize the legal findings and write your ENTIRE 'answer' field in fluent, professional English. Translate legal article citations accurately.",
    "FR": "L'utilisateur a écrit en français. Tu as récupéré les lois en arabe — c'est correct. Maintenant synthétise les conclusions juridiques et rédige TOUT le champ 'answer' en français fluide et professionnel. Traduis avec précision les citations des articles.",
}

def _build_qa_system(detected_language: str = "AR", intent: str = "SUBSTANTIVE") -> str:
    """Build the QA system prompt dynamically based on the user's detected language."""
    lang_directive = _LANGUAGE_DIRECTIVE.get(detected_language, _LANGUAGE_DIRECTIVE["AR"])
    
    validation_rule = """- CRITICAL: You MUST rely exclusively on the retrieved legal and procedural texts from the local database. 

- SYNTHESIS PERMISSION: If the user asks for a comparison or difference between two concepts (e.g., SARL vs SPA), or steps for a procedure, and the retrieved context contains the relevant definitions or rules, you are AUTHORIZED to synthesize the answer yourself. Extract the information from the provided articles and structure them clearly.

- If the provided context does NOT contain information about the query, you MUST refuse to answer."""

    return f"""You are an expert Algerian digital legal advisor.
Think step by step inside <thought> tags to analyze the question and the retrieved sources.
After thinking, output ONLY a valid JSON object — no other text:
{{
  "is_context_sufficient": "yes" | "no",
  "answer": "Your structured answer"
}}

Strict rules:
- {lang_directive}
- IDENTIFY the SPECIFIC sub-point the user is asking about. Answer ONLY that sub-point. Do NOT summarize the entire legal article.
- Structure procedural answers as numbered steps. Cite article numbers [المادة X] for substantive answers.
- Legal Acronym Dictionary: Never guess acronyms. Strictly use: SARL = شركة ذات مسؤولية محدودة, SPA = شركة مساهمة, SNC = شركة تضامن, EURL = مؤسسة ذات الشخص الوحيد وذات المسؤولية المحدودة.
{validation_rule}
- If the retrieved context does NOT contain sufficient information to answer, set is_context_sufficient to "no" and refuse to guess.
- NEVER invent legal facts not present in the retrieved context.
- Do NOT use Chinese characters or any unexpected symbols.
- CRITICAL: Output ONLY the JSON. 
- TONE: Always start your response with a brief, polite, and professional opening in the user's language before delivering the legal or procedural facts. Ensure the flow is natural and helpful.

FINAL INSTRUCTION: You MUST output your final answer ENTIRELY in the same language as the user's original query. If the query is in Arabic, your entire response must be in fluent Arabic. Do not use English unless defining an acronym."""

def _flatten_json_list(data):
    """Recursively flatten JSON lists into string format."""
    if isinstance(data, list):
        return "\n".join([str(item) for item in data])
    if isinstance(data, dict):
        return {k: _flatten_json_list(v) for k, v in data.items()}
    return data

def _parse_thought_and_json(raw: str) -> tuple[str, dict]:
    """Extract <thought> block and parse JSON securely."""
    thoughts = ""
    t_match = re.search(r"<thought>(.*?)</thought>", raw, re.DOTALL)
    if t_match:
        thoughts = t_match.group(1).strip()
        raw = raw.replace(t_match.group(0), "").strip()

    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return thoughts, json.loads(match.group())
        except:
            pass
    return thoughts, {"is_context_sufficient": "no", "answer": ""}

# ══════════════════════════════════════════════════════════════════════════════
# ── Main Agentic State Machine ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def agentic_answer(
    question: str,
    chat_history: list[dict] = None,
    verbose: bool = True,
    retriever_type: str = "bge",
    skip_gen: bool = False,
) -> dict:
    """Run the state-based hybrid agentic RAG loop."""
    # ── O(1) Instant CHITCHAT Interceptor — zero LLM cost ─────────────────────
    is_chitchat, lang = _check_instant_chitchat(question)
    if is_chitchat:
        return {"answer": _CHITCHAT_RESPONSES.get(lang, _CHITCHAT_RESPONSES["AR"]), "tools_called": [], "rounds": 0}

    state = "ROUTE"
    context_text = ""
    routing_data = {}
    tools_log = []
    final_answer_text = ""
    detected_language = _detect_language_fast(question)

    # Metadata for evaluation
    retrieved_ids = []
    context_sufficient = True # Default to true for chitchat/early returns

    while state != "END":
        if state == "ROUTE":
            routing_data = _run_router(question, chat_history, verbose)
            detected_language = routing_data.get("detected_language", detected_language)
            if verbose:
                print(f"  [Router] Lang: {detected_language} | Intent: {routing_data.get('intent')} | Rewritten: {routing_data.get('rewritten_query')}")
            if routing_data.get("intent") == "CHITCHAT":
                direct = routing_data.get("direct_response") or _CHITCHAT_RESPONSES.get(detected_language, _CHITCHAT_RESPONSES["AR"])
                return {"answer": direct, "tools_called": [], "rounds": 0}
            state = "RETRIEVE"

        elif state == "RETRIEVE":
            query_to_search = routing_data.get("rewritten_query") or question
            extracted = routing_data.get("extracted_articles", [])
            
            if retriever_type == "camelbert":
                from camelbert_rag.camelbert_retriever import camelbert_retrieve
                res = camelbert_retrieve(query_to_search, top_k=7)
                tool_name = "camelbert_retrieve"
            else:
                res = graph_retrieve(query_to_search, top_k=7, seeds=extracted)
                tool_name = "graph_retrieve"
                
            retrieved_ids = [r.get("id") for r in res if r.get("id")]
            articles = []
            for r in res:
                law = r.get('law_name', '')
                num = r.get('article_number', '')
                txt = r.get('text_original', '') or r.get('summary', '')
                articles.append(f"[{law} - المادة {num}]:\n{txt}")
            context_text = "النصوص القانونية والإجرائية المسترجعة:\n\n" + "\n\n".join(articles)
            tools_log.append({"tool": tool_name, "args": {"query": query_to_search, "seeds": extracted}})
            state = "VERIFY_AND_ANSWER"

        elif state == "VERIFY_AND_ANSWER":
            if skip_gen:
                final_answer_text = ""
                context_sufficient = True
                state = "END"
                continue

            intent = routing_data.get("intent", "SUBSTANTIVE")
            qa_system = _build_qa_system(detected_language, intent)
            prompt = qa_system + f"\n\nRetrieved context:\n{context_text}\n\nUser question: {question}\nOutput JSON only:"
            raw = local_generate(prompt, temperature=0.0, stream=False)
            thoughts, final_json = _parse_thought_and_json(raw)
            if verbose and thoughts:
                print(f"  [Thought] {thoughts}")
            
            context_sufficient = final_json.get("is_context_sufficient") == "yes"
            if context_sufficient:
                ans_raw = final_json.get("answer", "")
                
                # Sanitizer: Handle stubborn JSON arrays from 7B model
                if isinstance(ans_raw, str):
                    ans_raw = ans_raw.strip()
                    if ans_raw.startswith("[") and ans_raw.endswith("]"):
                        try:
                            maybe_list = json.loads(ans_raw)
                            if isinstance(maybe_list, list):
                                ans_raw = "\n".join(str(item) for item in maybe_list)
                        except: pass
                
                if isinstance(ans_raw, list):
                    final_answer_text = "\n".join(str(item) for item in ans_raw)
                elif isinstance(ans_raw, dict):
                    final_answer_text = json.dumps(ans_raw, ensure_ascii=False, indent=2)
                else:
                    final_answer_text = str(ans_raw)
            else:
                _no_context = {"AR": "عذراً، لم أتمكن من العثور على نص قانوني أو إجرائي دقيق يجيب على هذا السؤال في قاعدة البيانات المحلية.", "EN": "Sorry, I could not find a specific legal or procedural text to answer this question in the local database.", "FR": "Désolé, je n'ai pas trouvé de texte juridique ou procédural précis pour répondre à cette question dans la base de données locale."}
                final_answer_text = _no_context.get(detected_language, _no_context["AR"])
            state = "END"

    return {
        "answer": final_answer_text,
        "tools_called": tools_log,
        "retrieved_ids": retrieved_ids,
        "contexts": articles,
        "is_context_sufficient": context_sufficient,
        "rounds": 1
    }

def agentic_answer_stream(
    question: str,
    chat_history: list[dict] = None,
):
    """Generator version of the state machine."""
    # ── O(1) Instant CHITCHAT Interceptor — zero LLM cost ─────────────────────
    is_chitchat, lang = _check_instant_chitchat(question)
    if is_chitchat:
        resp = _CHITCHAT_RESPONSES.get(lang, _CHITCHAT_RESPONSES["AR"])
        # Yield status FIRST — gives React one render cycle to commit the
        # initial bot message to state before the content token arrives.
        yield {"type": "status", "message": "💬 ..."}
        # Yield the full greeting as ONE token (not char-by-char) to avoid
        # a flood of micro-events that can be dropped during state batching.
        yield {"type": "token", "content": resp}
        yield {"type": "done"}
        return

    yield {"type": "status", "message": "🔍 جاري تحليل السؤال ومراجعة السياق..."}
    state = "ROUTE"
    context_text = ""
    routing_data = {}
    detected_language = _detect_language_fast(question)

    while state != "END":
        if state == "ROUTE":
            routing_data = _run_router(question, chat_history)
            detected_language = routing_data.get("detected_language", detected_language)
            intent = routing_data.get("intent", "SUBSTANTIVE")

            if intent == "CHITCHAT":
                direct = routing_data.get("direct_response") or _CHITCHAT_RESPONSES.get(detected_language, _CHITCHAT_RESPONSES["AR"])
                yield {"type": "status", "message": "💬 ..."}
                yield {"type": "token", "content": direct}
                yield {"type": "done"}
                return
            
            lang_label = {"EN": "Offline query", "FR": "Requête hors ligne", "AR": "بحث محلي"}.get(detected_language, "بحث محلي")
            yield {"type": "status", "message": f"⚖️ {lang_label}: جاري البحث في قاعدة البيانات..."}
            state = "RETRIEVE"

        elif state == "RETRIEVE":
            query_to_search = routing_data.get("rewritten_query") or question
            extracted = routing_data.get("extracted_articles", [])
            
            # 100% Offline: Always use graph_retrieve
            res = graph_retrieve(query_to_search, top_k=7, seeds=extracted)
            articles, sources = [], []
            for r in res:
                law = r.get('law_name', '')
                num = r.get('article_number', '')
                txt = r.get('text_original', '') or r.get('summary', '')
                articles.append(f"[{law} - المادة {num}]:\n{txt}")
                sources.append({"law_name": law, "article_number": num, "title": r.get('title', ''), "score": r.get('graph_score', 1.0)})
            context_text = "النصوص القانونية والإجرائية المسترجعة:\n\n" + "\n\n".join(articles)
            yield {"type": "sources", "sources": sources}
            yield {"type": "status", "message": "🤖 التفكير في الإجابة وصياغتها..."}
            state = "VERIFY_AND_ANSWER"

        elif state == "VERIFY_AND_ANSWER":
            intent = routing_data.get("intent", "SUBSTANTIVE")
            qa_system = _build_qa_system(detected_language, intent)
            prompt = qa_system + f"\n\nRetrieved context:\n{context_text}\n\nUser question: {question}\nOutput JSON only:"
            raw = local_generate(prompt, temperature=0.0, stream=False)
            thoughts, final_json = _parse_thought_and_json(raw)
            if final_json.get("is_context_sufficient") == "yes":
                ans = final_json.get("answer", "")
                
                # Sanitizer: Handle stubborn JSON arrays from 7B model
                if isinstance(ans, str):
                    ans = ans.strip()
                    if ans.startswith("[") and ans.endswith("]"):
                        try:
                            maybe_list = json.loads(ans)
                            if isinstance(maybe_list, list):
                                ans = "\n".join(str(item) for item in maybe_list)
                        except: pass
                
                if isinstance(ans, list):
                    ans = "\n".join(str(item) for item in ans)
            else:
                _no_ctx = {"AR": "عذراً، لم أتمكن من العثور على نص قانوني أو إجرائي دقيق يجيب على هذا السؤال في قاعدة البيانات المحلية.", "EN": "Sorry, I could not find a specific legal or procedural text to answer this question in the local database.", "FR": "Désolé, je n'ai pas trouvé de texte juridique ou procédural précis pour répondre à cette question dans la base de données locale."}
                ans = _no_ctx.get(detected_language, _no_ctx["AR"])
            if thoughts:
                yield {"type": "status", "message": f"💡 فكرة: {thoughts[:200]}"}
            for char in ans:
                yield {"type": "token", "content": char}
            state = "END"

    yield {"type": "done"}
