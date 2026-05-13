import json
import re
import time
from pathlib import Path
import requests

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
                cleaned = re.sub(r'^\[\d+\]\s*', '', line)
                cleaned = cleaned.split("  (")[0].split(" (score")[0].strip()
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
        resp = requests.post(OLLAMA_URL, json={
            "model": JUDGE_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        }, timeout=60)
        
        score_text = resp.json().get("response", "").strip()
        match = re.search(r"0\.\d+|1\.0|0|1", score_text)
        if match:
            return float(match.group(0))
        return 0.0
    except Exception as e:
        print(f"Error calling LLM judge: {e}")
        return 0.0

def calculate_metrics(retrieved_sources, gt_sources):
    ret_norm = [re.sub(r'\s+', ' ', s).strip() for s in retrieved_sources]
    gt_norm = [re.sub(r'\s+', ' ', s).strip() for s in gt_sources]
    
    hits = 0
    mrr = 0.0
    for i, r in enumerate(ret_norm):
        if r in gt_norm:
            hits += 1
            if mrr == 0.0:
                mrr = 1.0 / (i + 1)
                
    hit_rate = 1.0 if hits > 0 else 0.0
    precision = hits / len(ret_norm) if ret_norm else 0.0
    recall = hits / len(gt_norm) if gt_norm else 0.0
    
    return precision, hit_rate, mrr, recall

def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} questions from Ground Truth dataset.")
    
    # Load all models' outputs
    model_results = {}
    for model_name, path in MODELS.items():
        model_results[model_name] = parse_txt_results(path)
        print(f"Loaded {len(model_results[model_name])} answers for {model_name}.")

    summary = {}
    
    for model_name, results in model_results.items():
        if not results:
            continue
            
        print(f"\nEvaluating {model_name}...")
        metrics = {
            "Precision": 0.0,
            "Hit Rate": 0.0,
            "MRR": 0.0,
            "Recall": 0.0,
            "Legal Accuracy": 0.0
        }
        
        count = 0
        for i, item in enumerate(dataset):
            q = item.get("question", "").strip()
            gt = item.get("ground_truth", "")
            gt_articles = item.get("articles", [])
            
            if q not in results:
                continue
                
            model_ans = results[q]["answer"]
            model_src = results[q]["sources"]
            
            # Retrieval metrics
            p, hr, mrr, r = calculate_metrics(model_src, gt_articles)
            metrics["Precision"] += p
            metrics["Hit Rate"] += hr
            metrics["MRR"] += mrr
            metrics["Recall"] += r
            
            # Generation Accuracy (LLM Judge)
            print(f"  [{i+1}/{len(dataset)}] Judging answer... ", end="", flush=True)
            acc = llm_judge(q, gt, model_ans)
            print(f"Score: {acc}")
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
    
    with open("evaluation_final_results.md", "w", encoding="utf-8") as f:
        f.write(md_text)
        
    with open("evaluation_final_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
