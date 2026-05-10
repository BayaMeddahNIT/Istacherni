import json
import re
import os
import sys
import time
import urllib.request
import urllib.error
import argparse
from pathlib import Path
import csv

# Add the project root to the python path to allow imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout uses UTF-8 to prevent UnicodeEncodeError in Windows terminals
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIGURATION ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
JUDGE_MODEL = "qwen2.5:7b"
DATASET_PATH = PROJECT_ROOT / "algerian_law_ragas_dataset_v3.json"
ANSWERS_FILE = PROJECT_ROOT / "answers_gemma2_finetuned.txt"
OUTPUT_CSV = PROJECT_ROOT / "custom_judge_results.csv"

# --- UTILS: Parse Answers File ---
def normalize_arabic(text: str) -> str:
    """Normalize Arabic text: remove diacritics, unify chars, lowercase."""
    text = re.sub(r'[\u0617-\u061A\u064B-\u065F]', '', text)
    text = re.sub(r'[آأإ]', 'ا', text)
    text = text.replace('ة', 'ه')
    text = text.replace('ى', 'ي')
    text = text.replace('ؤ', 'و')
    text = text.replace('ئ', 'ي')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def parse_answer_file(path: Path) -> dict:
    """
    Parse an answer .txt file and return a dict keyed by normalized question.
    """
    results = {}
    try:
        raw = path.read_text(encoding='utf-8', errors='replace')
    except FileNotFoundError:
        print(f"Error: Answer file not found: {path}")
        return results

    blocks = re.split(r'={40,}', raw)
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        q_match = re.search(r'\[User\]:\s*(.+?)(?:\r?\n)', block)
        if not q_match:
            continue
        question = q_match.group(1).strip()

        ans_match = re.search(r'ANSWER:\r?\n(.*?)(?:SOURCES:|$)', block, re.DOTALL)
        if not ans_match:
            ans_match = re.search(r'ANSWER:\r?\n(.*)', block, re.DOTALL)
        answer = ans_match.group(1).strip() if ans_match else ""

        answer = re.sub(r'\(Time taken:.*?\)', '', answer).strip()

        sources_match = re.search(r'SOURCES:\r?\n(.*)', block, re.DOTALL)
        sources_text = sources_match.group(1).strip() if sources_match else ""

        norm_q = normalize_arabic(question)
        results[norm_q] = {
            "question": question,
            "answer": answer,
            "sources_text": sources_text,
        }
    return results

# --- OLLAMA JUDGE ---
def call_ollama_judge(question: str, ground_truth: str, generated_answer: str) -> dict:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    
    prompt = f"""You are an expert Algerian Legal Evaluator. Your task is to evaluate the quality of a generated answer compared to a ground truth answer.

QUESTION: {question}

GROUND TRUTH (EXPECTED ANSWER):
{ground_truth}

GENERATED ANSWER (TO EVALUATE):
{generated_answer}

EVALUATION CRITERIA:
1. Legal Accuracy (1-5): Does the generated answer accurately reflect the legal principles in the ground truth? Are there any hallucinations? (1=Completely wrong, 5=Perfectly accurate)
2. Citation Precision (1-5): Did the generated answer cite the correct laws and articles exactly as specified in the ground truth? (1=No citations or wrong citations, 5=All required citations are perfectly mentioned)

INSTRUCTIONS:
Return ONLY a valid JSON object in English with the following keys, AND DO NOT output any Chinese text. The reasoning should be a short explanation in Arabic.
{{
  "legal_accuracy": <int>,
  "citation_precision": <int>,
  "reasoning": "<short string explaining the scores in Arabic>"
}}
"""
    
    payload = json.dumps({
        "model": JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0 # Make it deterministic
        }
    }).encode("utf-8")
    
    headers = {"Content-Type": "application/json"}
    
    try:
        req = urllib.request.Request(url, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            response_text = data.get("response", "{}")
            return json.loads(response_text)
    except Exception as e:
        print(f"  -> Judge API Error: {e}")
        return {
            "legal_accuracy": 0,
            "citation_precision": 0,
            "reasoning": f"Error: {str(e)}"
        }

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true", help="Only evaluate the first 3 questions")
    args = parser.parse_args()

    print(f"Loading dataset from {DATASET_PATH.name}...")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    print(f"Loading generated answers from {ANSWERS_FILE.name}...")
    answers_map = parse_answer_file(ANSWERS_FILE)
    print(f"Found {len(answers_map)} parsed answers.")
    
    records = []
    
    # Load existing records if any to resume
    start_idx = 0
    if OUTPUT_CSV.exists() and not args.test_mode:
        with open(OUTPUT_CSV, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            records = list(reader)
            start_idx = len(records)
            print(f"Resuming from question {start_idx + 1}...")
    else:
        # Create new CSV
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question", "legal_accuracy", "citation_precision", "reasoning"
            ])
            writer.writeheader()

    limit = 3 if args.test_mode else len(dataset)
    
    total_acc = sum(int(r.get("legal_accuracy", 0)) for r in records)
    total_cit = sum(int(r.get("citation_precision", 0)) for r in records)
    
    print("\nStarting evaluation...")
    for i in range(start_idx, min(limit, len(dataset))):
        item = dataset[i]
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        
        norm_q = normalize_arabic(question)
        entry = answers_map.get(norm_q)
        
        # Fuzzy match if exact fails
        if not entry:
            for k in answers_map:
                if k in norm_q or norm_q in k:
                    entry = answers_map[k]
                    break
        
        print(f"[{i+1}/{limit}] Q: {question}")
        if not entry:
            print("  -> Could not find generated answer for this question. Skipping.")
            continue
            
        gen_answer = entry["answer"] + "\nSOURCES:\n" + entry["sources_text"]
        
        start_time = time.time()
        judge_result = call_ollama_judge(question, ground_truth, gen_answer)
        elapsed = time.time() - start_time
        
        acc_score = judge_result.get("legal_accuracy", 0)
        cit_score = judge_result.get("citation_precision", 0)
        reasoning = judge_result.get("reasoning", "")
        
        print(f"  -> Legal Accuracy: {acc_score}/5 | Citation Precision: {cit_score}/5 ({elapsed:.1f}s)")
        print(f"  -> Reasoning: {reasoning}")
        
        row = {
            "question": question,
            "legal_accuracy": acc_score,
            "citation_precision": cit_score,
            "reasoning": reasoning
        }
        
        records.append(row)
        total_acc += acc_score
        total_cit += cit_score
        
        # Save progressively
        with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=["question", "legal_accuracy", "citation_precision", "reasoning"])
            writer.writerow(row)
            
    # Summary
    if records:
        print("\n" + "="*50)
        print("  LLM-AS-JUDGE EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Evaluated     : {len(records)}")
        print(f"Avg Legal Accuracy  : {total_acc / len(records):.2f} / 5")
        print(f"Avg Citation Precision : {total_cit / len(records):.2f} / 5")
        print("="*50)

if __name__ == "__main__":
    main()
