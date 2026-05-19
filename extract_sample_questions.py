"""
extract_sample_questions.py
Writes the 40-question stratified sample to questions.txt.
Called by run_full_evaluation.ps1 before each benchmark run.
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

SAMPLE = "algerian_law_ragas_dataset_v3_sample40.json"
OUTPUT = "questions.txt"

with open(SAMPLE, encoding="utf-8") as f:
    data = json.load(f)

with open(OUTPUT, "w", encoding="utf-8") as out:
    for item in data:
        out.write(item["question"].strip() + "\n")

print(f"Written {len(data)} questions to {OUTPUT}")
