import json
import re
import time
import unicodedata
from pathlib import Path
import requests
import argparse
import filelock
import os

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

DATASET_PATH = "algerian_law_ragas_dataset_v3.json"
MODELS = {
    "Graph RAG": "answers_graph_rag.txt",
    "Qwen RAG": "answers_qwen_rag.txt",
    "CamELBERT RAG": "answers_camelbert_rag.txt"
}
OLLAMA_URL = "http://localhost:11434/api/generate"
JUDGE_MODEL = "gemma2:9b"

def parse_txt_results(file_path):
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    blocks = content.split("==================================================")
    results = {}
    
    for block in blocks:
        block = block.strip()
        if not block or ("===" in block and "Results" in block):
            continue
        
        q_match = re.search(r"\[User\]:\s*(.*)", block)
        if not q_match:
            continue
        question = q_match.group(1).strip()
        
        ans_start = block.find("ANSWER:")
        if ans_start == -1:
            continue
        ans_start += len("ANSWER:")
        ans_end = block.find("(Time taken:")
        if ans_end == -1:
            ans_end = block.find("SOURCES:")
        
        answer = block[ans_start:ans_end].strip() if ans_end != -1 else block[ans_start:].strip()
        
        sources_str = ""
        sources_start = block.find("SOURCES:")
        if sources_start != -1:
            sources_str = block[sources_start + len("SOURCES:"):].strip()
        
        sources = []
        for line in sources_str.split('\n'):
            line = line.strip()
            if not line: continue
            if line.startswith("["):
                # Remove leading [N] index
                cleaned = re.sub(r'^\[\d+\]\s*', '', line)
                # Strip ALL trailing parenthetical metadata blocks:
                # handles (score=...), (graph_score=..., pagerank=...), (time=...) etc.
                cleaned = re.sub(r'\s*\([^)]*\)', '', cleaned).strip()
                if cleaned:
                    sources.append(cleaned)
        
        results[question] = {
            "answer": answer,
            "sources": sources
        }
    return results

def llm_judge(question, ground_truth, generated_answer):
    if not generated_answer or generated_answer.strip() == "" or "Error" in generated_answer:
        return 0.0
        
    prompt = f"""أنت خبير في القانون الجزائري. قيم مدى دقة إجابة النظام مقارنة بالإجابة النموذجية الصحيحة (Ground Truth).
    
السؤال: {question}
الإجابة النموذجية الصحيحة: {ground_truth}
إجابة النظام: {generated_answer}

قيم إجابة النظام. يجب أن تكون إجابتك عبارة عن رقم عشري واحد فقط (float) بين 0.0 و 1.0 (حيث 1.0 تعني دقيقة تماماً و 0.0 تعني خاطئة تماماً). لا تكتب أي نص آخر مع الرقم.
التقييم:"""
    
    try:
        if ANTHROPIC_API_KEY:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=20,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            score_text = message.content[0].text.strip()
        else:
            resp = requests.post(OLLAMA_URL, json={
                "model": JUDGE_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            }, timeout=600)
            
            score_text = resp.json().get("response", "").strip()
            
        match = re.search(r"0\.\d+|1\.0|0|1", score_text)
        if match:
            return float(match.group(0))
        return 0.0
    except Exception as e:
        print(f"Error calling LLM judge: {e}")
        return 0.0

# Arabic diacritics (tashkeel) Unicode range to strip for robust comparison
_ARABIC_DIACRITICS_RE = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0640\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]')

# Letter-variant map: alef variants, tah marbuta, alef maqsura
# Ensures 'جزاىري' (GT) matches 'جزائري' (corpus) — a silent match failure otherwise.
_ARABIC_VARIANTS = str.maketrans({
    '\u0623': '\u0627',  # أ → ا
    '\u0625': '\u0627',  # إ → ا
    '\u0622': '\u0627',  # آ → ا
    '\u0649': '\u064A',  # ى → ي
})

def _normalize_source(s: str) -> str:
    """NFKC-normalize, strip Arabic diacritics, normalize letter variants, collapse whitespace."""
    s = unicodedata.normalize('NFKC', s)
    s = _ARABIC_DIACRITICS_RE.sub('', s)
    s = s.translate(_ARABIC_VARIANTS)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def calculate_metrics(retrieved_sources, gt_sources):
    # Normalize both sides
    ret_norm_raw = [_normalize_source(s) for s in retrieved_sources]
    gt_norm = [_normalize_source(s) for s in gt_sources]

    # Deduplicate retrieved sources while preserving rank order.
    # Duplicates in retrieved results inflate the denominator and deflate Precision.
    seen = set()
    ret_norm = []
    for r in ret_norm_raw:
        if r not in seen:
            seen.add(r)
            ret_norm.append(r)

    # Deduplicate GT sources (defensive)
    gt_norm = list(dict.fromkeys(gt_norm))

    hits = 0
    mrr = 0.0
    gt_set = set(gt_norm)
    for i, r in enumerate(ret_norm):
        if r in gt_set:
            hits += 1
            if mrr == 0.0:
                mrr = 1.0 / (i + 1)

    hit_rate = 1.0 if hits > 0 else 0.0
    precision = hits / len(ret_norm) if ret_norm else 0.0
    recall = hits / len(gt_norm) if gt_norm else 0.0

    return precision, hit_rate, mrr, recall

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Models")
    parser.add_argument("--model", type=str, help="Specify a single model to evaluate (e.g., 'Graph RAG')")
    args = parser.parse_args()

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} questions from Ground Truth dataset.")
    
    # Load all models' outputs
    model_results = {}
    for model_name, path in MODELS.items():
        if args.model and model_name != args.model:
            continue
        model_results[model_name] = parse_txt_results(path)
        print(f"Loaded {len(model_results[model_name])} answers for {model_name}.")

    summary = {}
    
    for model_name, results in model_results.items():
        if not results:
            continue
            
        print(f"\nEvaluating {model_name}...")
        model_safe = model_name.replace(' ', '_')
        cache_path = Path(f"eval_llm_cache_{model_safe}.json")
        lock_path = f"{cache_path}.lock"
        
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                llm_cache = json.load(f)
            print(f"Loaded {len(llm_cache)} cached LLM judgments.")
        else:
            llm_cache = {}

        metrics = {
            "Precision": 0.0,
            "Hit Rate": 0.0,
            "MRR": 0.0,
            "Recall": 0.0,
            "Legal Accuracy": 0.0
        }
        
        # Build a normalized question -> raw question index for robust lookup
        norm_to_result_key = {_normalize_source(k): k for k in results}

        count = 0
        for i, item in enumerate(dataset):
            q = item.get("question", "").strip()
            gt = item.get("ground_truth", "")
            gt_articles = item.get("articles", [])

            # Use normalized comparison so minor whitespace differences don't break lookup
            q_norm = _normalize_source(q)
            result_key = norm_to_result_key.get(q_norm)
            if result_key is None:
                continue
                
            model_ans = results[result_key]["answer"]
            model_src = results[result_key]["sources"]
            
            # Retrieval metrics
            p, hr, mrr, r = calculate_metrics(model_src, gt_articles)
            metrics["Precision"] += p
            metrics["Hit Rate"] += hr
            metrics["MRR"] += mrr
            metrics["Recall"] += r
            
            # Generation Accuracy (LLM Judge)
            print(f"  [{i+1}/{len(dataset)}] Judging answer... ", end="", flush=True)
            
            cache_key = f"{model_name}:::{q}"
            if cache_key in llm_cache:
                acc = llm_cache[cache_key]
                print(f"Score: {acc} (cached)")
            else:
                acc = llm_judge(q, gt, model_ans)
                print(f"Score: {acc}")
                llm_cache[cache_key] = acc
                
                lock = filelock.FileLock(lock_path)
                with lock:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(llm_cache, f, ensure_ascii=False, indent=2)
            
            metrics["Legal Accuracy"] += acc
            
            count += 1
            
        if count > 0:
            for k in metrics:
                metrics[k] /= count
            
            summary[model_name] = metrics
            print(f"Done evaluating {model_name}. Averaged over {count} questions.")
        
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)
    
    md_lines = []
    md_lines.append("# RAG Models Evaluation")
    md_lines.append("")
    md_lines.append(f"**Dataset**: {len(dataset)} questions")
    md_lines.append(f"**Judge Model**: {JUDGE_MODEL}")
    md_lines.append("")
    md_lines.append("| Model | Precision | Hit Rate | MRR | Recall | Legal Accuracy |")
    md_lines.append("|-------|-----------|----------|-----|--------|----------------|")
    
    for m, s in summary.items():
        md_lines.append(f"| {m} | {s['Precision']:.4f} | {s['Hit Rate']:.4f} | {s['MRR']:.4f} | {s['Recall']:.4f} | {s['Legal Accuracy']:.4f} |")
        
    md_text = "\n".join(md_lines)
    print(md_text)
    
    out_suffix = f"_{args.model.replace(' ', '_')}" if args.model else ""
    
    with open(f"evaluation_final_results{out_suffix}.md", "w", encoding="utf-8") as f:
        f.write(md_text)
        
    with open(f"evaluation_final_results{out_suffix}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
