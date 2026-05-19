#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import time
import json
import traceback
from pathlib import Path
import requests

# Must be set BEFORE any torch/cuda imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Ensure UTF-8 output
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from bm25_rag.bm25_retriever import bm25_retrieve
from dense_rag.bge_retriever import dense_retrieve
from hybrid_rag.hybrid_retriever import reciprocal_rank_fusion
from hybrid_rag.reranker import rerank_candidates

SYSTEM_PROMPT = """أنت مستشار قانوني خبير ومتخصص في القانون الجزائري.
مهمتك هي الإجابة على أسئلة المستخدمين استناداً حصراً إلى المواد القانونية المقدَّمة.
لضمان أعلى درجات الدقة، يجب عليك التفكير بعمق وتحليل المواد خطوة بخطوة قبل الإجابة.

القواعد الصارمة:
1. أجب دائماً باللغة العربية الفصحى بأسلوب قانوني رصين ومفصل.
2. فكر خطوة بخطوة (Chain of Thought): اشرح كيف توصلت إلى الاستنتاج من خلال تفسير كل مادة على حدة، واربطها بالسؤال بشكل منطقي.
3. الاستشهاد الإلزامي: ادعم كل حُكم أو معلومة تستخرجها بالصيغة التالية بالضبط: [المادة {رقم المادة}، {اسم القانون}].
4. يُمنع منعاً باتاً استخدام أو اختراع أي معلومة خارج نصوص المواد المرفقة.
5. التعامل مع نقص المعلومات:
   - إذا كانت المواد تُثبت أن الفعل ليس جريمة، أو تنفي الحكم، فهذه تعتبر إجابة (وضّح ذلك استناداً للنصوص).
   - أما إذا كانت المواد المرفقة غير ذات صلة تماماً بالسؤال، فقل بوضوح: "لا تتوفر معلومات كافية في المواد المقدمة."
6. الهيكلة: رتّب إجابتك بشكل منظم دائماً باستخدام العناوين التالية:
   (١) التحليل القانوني: (اشرح الموقف وفسر المواد)
   (٢) الحكم الرئيسي: (الإجابة المباشرة مع الاستشهاد)
   (٣) الشروط: (إن نصت عليها المواد)
   (٤) العقوبات: (إن وُجدت صراحة في المواد)
"""

def check_ollama_connection(model_name: str) -> bool:
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            if model_name in models:
                print(f"[Ollama] Connected. Model '{model_name}' ready.")
                return True
            else:
                print(f"[Ollama] Connected but '{model_name}' not found.")
                print(f"[Ollama] Available models: {models}")
                print(f"[Ollama] Run: ollama pull {model_name}")
                return False
    except Exception as e:
        print(f"[Ollama] Cannot connect: {e}")
        print("[Ollama] Make sure Ollama is running: ollama serve")
        return False

def build_messages(question: str, context_chunks: list[str]) -> list:
    # Take top 5 chunks
    top_chunks = context_chunks[:5]

    context_text = "\n\n".join([
        f"المادة {i+1}:\n{chunk}"
        for i, chunk in enumerate(top_chunks)
    ])

    user_content = f"""بناءً على المواد القانونية الجزائرية التالية:

{context_text}

السؤال: {question}

تذكر: يجب أن تكون إجابتك باللغة العربية الفصحى فقط.
أجب على السؤال أعلاه باللغة العربية الفصحى فقط،
مستنداً إلى المواد القانونية المذكورة.
الإجابة:"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content}
    ]

def run_query_with_timing(question: str) -> dict:
    timings = {}

    # Stage 1: BM25 retrieval
    t0 = time.time()
    bm25_results = bm25_retrieve(question, top_k=100)
    timings["bm25_retrieval"] = round(time.time() - t0, 2)

    # Stage 2: Dense retrieval
    t0 = time.time()
    dense_results = dense_retrieve(question, top_k=100)
    timings["dense_retrieval"] = round(time.time() - t0, 2)

    # Stage 3: RRF fusion
    t0 = time.time()
    fused = reciprocal_rank_fusion(
        bm25_results, dense_results,
        bm25_weight=0.3, dense_weight=0.7
    )[:30]  # Increased to 30 to restore high precision (from previous 10)
    timings["rrf_fusion"] = round(time.time() - t0, 2)

    # Stage 4: Reranking
    t0 = time.time()
    reranked = rerank_candidates(question, fused, top_k=5)
    timings["reranking"] = round(time.time() - t0, 2)

    # Stage 5: LLM generation
    t0 = time.time()
    # Ensure text is properly formatted for the chunks
    def build_doc_text(doc: dict) -> str:
        law_name = doc.get("law_name", "")
        art_num  = doc.get("article_number", "")
        title    = doc.get("title", "")
        original = doc.get("text_original", "")
        expl     = doc.get("text_explanation", "")
        summary  = doc.get("summary", "")
        kws_raw  = doc.get("keywords", "")
        kws      = " ".join(kws_raw) if isinstance(kws_raw, list) else str(kws_raw)
        return f"[{law_name} - المادة {art_num}] {title}. {original} {expl} {summary} {kws}".strip()

    messages = build_messages(question, [build_doc_text(r) for r in reranked])
    
    # NOTE: user must run this once: ollama pull gemma2:9b
    model_name = "gemma2:9b"
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model_name,
                "options": {
                    "num_gpu": 28,      # Partially offload 28 layers to GPU so it fits in 6GB VRAM
                    "num_thread": 6,    # CPU threads for the remaining 14 layers
                    "num_ctx": 2048,    # Lower context window to guarantee VRAM safety
                    "temperature": 0.1, # low temp for factual legal answers
                    "top_p": 0.9,
                },
                "messages": messages,
                "stream": False
            },
            timeout=180
        )
        response.raise_for_status()
        answer = response.json()["message"]["content"]
    except Exception as e:
        answer = f"Error calling Ollama: {e}"
        
    timings["llm_generation"] = round(time.time() - t0, 2)

    timings["total"] = round(sum(timings.values()), 2)

    # Print timing breakdown
    print("\n[Pipeline Timing]")
    for stage, seconds in timings.items():
        print(f"  {stage:<20}: {seconds}s")

    return {
        "question": question,
        "answer": answer,
        "sources": reranked,
        "timings": timings
    }

def main():
    model_name = "gemma2:9b"
    if not check_ollama_connection(model_name):
        return

    input_file = PROJECT_ROOT / "algerian_law_ragas_dataset_v3.json"
    output_file = PROJECT_ROOT / "answers_fused.txt"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found.") 
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    questions = [item["question"] for item in dataset]
    print(f"Loaded {len(questions)} questions. Starting generation...")
    
    with open(output_file, "w", encoding="utf-8") as out:
        for i, q in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Processing: {q}", flush=True)
            
            try:
                res = run_query_with_timing(q)
                answer = res["answer"]
                chunks = res["sources"]
                elapsed = res["timings"]["total"]
            except Exception as e:
                traceback.print_exc()
                answer = f"Error during processing: {e}"
                chunks = []
                elapsed = 0.0
                
            out.write(f"[User]: {q}\n")
            out.write(f"ANSWER:\n{answer}\n")
            out.write(f"(Time taken: {elapsed:.2f} seconds)\n")
            out.write("SOURCES:\n")
            for idx, chunk in enumerate(chunks, 1):
                law_name = chunk.get("law_name", "قانون غير معروف")
                article_num = chunk.get("article_number", "N/A")
                out.write(f"[{idx}] {law_name} - المادة {article_num}\n")
            
            out.write("\n" + "="*50 + "\n\n")
            out.flush()
            
    print(f"\nDone! All {len(questions)} answers have been saved to {output_file}")

if __name__ == "__main__":
    main()
