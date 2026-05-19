import json

with open('evaluation_comparison.json', 'r', encoding='utf-8') as f:
    r = json.load(f)

mA = r['reranker_A_bgem3_unified']
mB = r['reranker_B_generic_ce']
mC = r['reranker_C_finetuned_ce']

with open('eval_summary.md', 'w', encoding='utf-8') as out:
    out.write('# Evaluation Results\n\n')
    out.write('| Metric | BGE-M3 Unified | Generic CE | Fine-Tuned CE |\n')
    out.write('|---|---|---|---|\n')
    out.write(f'| Precision@1 | {mA["precision_at_1"]*100:.2f}% | {mB["precision_at_1"]*100:.2f}% | {mC["precision_at_1"]*100:.2f}% |\n')
    out.write(f'| MRR | {mA["mrr"]:.4f} | {mB["mrr"]:.4f} | {mC["mrr"]:.4f} |\n')
    out.write(f'| Hit Rate @5 | {mA["hit_rate_5"]*100:.2f}% | {mB["hit_rate_5"]*100:.2f}% | {mC["hit_rate_5"]*100:.2f}% |\n')
    out.write(f'| NDCG@5 | {mA["ndcg_5"]:.4f} | {mB["ndcg_5"]:.4f} | {mC["ndcg_5"]:.4f} |\n\n')

    improved = [q for q in r['per_query'] if q['A_to_C_improved']]
    regressed = [q for q in r['per_query'] if q['A_to_C_regressed']]
    
    out.write('## Improved Queries (A -> C)\n')
    for q in improved:
        out.write(f'- Rank {q["reranker_A_rank"]} -> {q["reranker_C_rank"]}: {q["query"]}\n')
        
    out.write('\n## Regressed Queries (A -> C)\n')
    for q in regressed:
        out.write(f'- Rank {q["reranker_A_rank"]} -> {q["reranker_C_rank"]}: {q["query"]}\n')
