"""
run_qwen_rag_baseline.py
------------------------
Generates a raw, unoptimized Qwen 7B baseline run (Column 1b).
Uses qwen_retrieve() + raw Qwen 7B via Ollama (using the unoptimized system prompt).
Saves answers to answers_qwen_rag_baseline.txt.

Usage:
    python run_qwen_rag_baseline.py
"""

import sys
import time
import os
import urllib.request
import urllib.error
import json
import re
from pathlib import Path
from dotenv import load_dotenv

# ── UTF-8 output (Windows) ───────────────────────────────────────────────────────
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# ── Path setup ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

# ── Import retriever ────────────────────────────────────────────────────────────
from qwen_rag.qwen_retriever import qwen_retrieve

# ── Config ───────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL:    str = "qwen2:1.5b"
OLLAMA_TIMEOUT:  int = 600
OLLAMA_NUM_CTX:  int = 2048
OLLAMA_TEMP:     float = 0.1

# ── System prompt (Exactly the unoptimized baseline from gemma_generator.py) ─────
_SYSTEM = """مساعد قانوني متخصص في القانون الجزائري.
أجب حصراً بناءً على المواد المرفقة.
قواعد صارمة:
1. ابدأ تحليلك دائماً من المادة [1] باعتبارها الأعلى صلة. لا تتجاهل المادة الأولى لصالح مادة لاحقة.
2. اذكر دائماً رقم المادة واسم القانون لكل معلومة.
3. إذا لم تجد في المواد المقدمة نصاً صريحاً، اذكر: 'لا تتضمن المواد المقدمة نصاً على هذه الحالة'.
4. أجب بنص عادي مباشر وابتعد عن التنسيق المعقد."""

# ── Context pollution blacklist ─────────────────────────────────────────────
_CONTEXT_BLACKLIST = {
    ("قانون العقوبات",                                        "3"),
    ("قانون الوقاية من الجرائم المتصلة بتكنولوجيات",         "2"),
    ("قانون الوقاية من الجرائم المتصلة بتكنولوجيات",         "3"),
    ("القانون المدني",                                         "1"),
    ("القانون التجاري",                                        "1"),
}

def _is_blacklisted(art: dict) -> bool:
    law = art.get("law_name", "")
    num = str(art.get("article_number", ""))
    return any(
        frag in law and num == num_key
        for frag, num_key in _CONTEXT_BLACKLIST
    )

def _build_context(retrieved: list[dict]) -> str:
    if not retrieved:
        return "لا توجد مواد قانونية ذات صلة."

    filtered = [a for a in retrieved if not _is_blacklisted(a)]
    retrieved = filtered if filtered else retrieved

    parts = []
    for art in retrieved:
        law  = art.get("law_name",       "قانون غير معروف")
        num  = art.get("article_number", "N/A")
        header = f"【{law} — المادة {num}】"
        if art.get("title"):
            header += f" ({art['title']})"

        body_lines = [art.get("text_original", "")]
        if art.get("legal_conditions_summary"):
            body_lines.append(f"الشروط: {art['legal_conditions_summary']}")
        if art.get("penalties_summary"):
            body_lines.append(f"العقوبة: {art['penalties_summary']}")

        parts.append(f"{header}\n" + "\n".join(body_lines))

    return "\n\n---\n\n".join(parts)

# ── CJK range cleanup ───────────────────────────────────────────────────────────
_CJK_RANGE = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+')
def _sanitize_arabic(text: str) -> str:
    return _CJK_RANGE.sub('[...]‏', text)

# ── Raw generator call ──────────────────────────────────────────────────────────
def raw_qwen_generate(question: str, retrieved: list[dict]) -> str:
    context = _build_context(retrieved)
    user_prompt = f"=== المواد القانونية ذات الصلة ===\n\n{context}\n\n=== سؤال المستخدم ===\n\n{question}\n\n=== الإجابة ==="
    
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMP,
            "num_ctx": OLLAMA_NUM_CTX,
        },
        "messages": messages,
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    try:
        req = urllib.request.Request(url, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        res = body["message"]["content"].strip()
        return _sanitize_arabic(res)
    except Exception as e:
        return f"Error connecting to local Ollama: {e}"

# ── Main runner ─────────────────────────────────────────────────────────────────
def main():
    input_file  = PROJECT_ROOT / "questions.txt"
    output_file = PROJECT_ROOT / "answers_qwen_rag_baseline.txt"

    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    questions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(("1.", "2.", "3.", "4.", "5.", "🟦", "🟨", "🟩", "🟪")):
            continue
        questions.append(line)

    print(f"Loaded {len(questions)} questions for unoptimized Qwen 7B baseline.")
    
    with open(output_file, "a", encoding="utf-8") as out:
        # Resume support
        already_done = 0
        if output_file.exists():
            content = output_file.read_text(encoding="utf-8")
            already_done = content.count("[User]:")
            if already_done > 0:
                print(f"Resuming from question {already_done + 1} (skipping {already_done} already done).")
                questions = questions[already_done:]
        else:
            out.write(f"=== Qwen RAG Results (Unoptimized Baseline) — Generator: {OLLAMA_MODEL} ===\n\n")

        for i, q in enumerate(questions, 1):
            print(f"[{i + already_done}/{len(questions) + already_done}] Processing: {q}", flush=True)
            start = time.time()

            try:
                chunks = qwen_retrieve(q, top_k=3)
                answer = raw_qwen_generate(q, chunks)
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(error_trace, flush=True)
                answer = f"Error during processing: {e}\n{error_trace}"
                chunks = []

            elapsed = time.time() - start

            out.write(f"[User]: {q}\n")
            out.write(f"ANSWER:\n{answer}\n")
            out.write(f"(Time taken: {elapsed:.2f} seconds)\n")
            out.write("SOURCES:\n")
            for idx, chunk in enumerate(chunks, 1):
                law_name    = chunk.get("law_name",       "قانون غير معروف")
                article_num = chunk.get("article_number", "N/A")
                score       = chunk.get("score",          0)
                out.write(f"[{idx}] {law_name} - المادة {article_num}  (score={score:.4f})\n")
            out.write("\n" + "=" * 50 + "\n\n")
            out.flush()

    print(f"\nDone! All baseline answers saved to {output_file}")


if __name__ == "__main__":
    main()
