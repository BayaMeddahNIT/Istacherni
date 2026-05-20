import json
import glob
import sys
from pathlib import Path

def main():
    print("Merging evaluation results...")
    
    # Only load files that end with a model name, avoiding the base file if it's old
    # e.g., evaluation_final_results_Graph_RAG.json
    files = glob.glob("evaluation_final_results_*.json")
    results = {}
    
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            results.update(data)
            
    if not results:
        print("No result files found. Make sure the evaluation scripts have finished.")
        sys.exit(1)
        
    md_lines = []
    md_lines.append("# Merged RAG Models Evaluation")
    md_lines.append("")
    md_lines.append("| Model | Precision | Hit Rate | MRR | Recall | Legal Accuracy |")
    md_lines.append("|-------|-----------|----------|-----|--------|----------------|")
    
    for m, s in results.items():
        md_lines.append(f"| {m} | {s['Precision']:.4f} | {s['Hit Rate']:.4f} | {s['MRR']:.4f} | {s['Recall']:.4f} | {s['Legal Accuracy']:.4f} |")
        
    md_text = "\n".join(md_lines)
    print("\n" + md_text)
    
    out_file = "evaluation_final_results_MERGED.md"
    with open(out_file, "w", encoding="utf-8") as out:
        out.write(md_text)
        
    print(f"\nSaved merged results to {out_file}")

if __name__ == "__main__":
    main()
