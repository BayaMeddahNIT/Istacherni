# Full 127-question benchmark — qwen2:7b generator
# Resume-safe: each run_*.py script skips already-answered questions

# Clean only if starting fresh (comment these out to resume a partial run)
Remove-Item answers_graph_rag.txt    -ErrorAction SilentlyContinue
Remove-Item answers_qwen_rag.txt     -ErrorAction SilentlyContinue
Remove-Item answers_camelbert_rag.txt -ErrorAction SilentlyContinue

Write-Host "GENERATING ANSWERS - Full 127 questions x 3 models"
Write-Host "Generator: qwen2:7b | Est. total runtime: ~26 hours (Graph RAG retrieval-bound)"

Write-Host "[1/3] Graph RAG..."
.\.venv\Scripts\python.exe run_graph_rag.py > run_graph_rag_log.txt 2>&1

Write-Host "[2/3] Qwen RAG..."
.\.venv\Scripts\python.exe run_qwen_rag.py > run_qwen_rag_log.txt 2>&1

Write-Host "[3/3] CamELBERT RAG..."
.\.venv\Scripts\python.exe run_camelbert_rag.py > run_camelbert_rag_log.txt 2>&1

Write-Host ""
Write-Host "DONE - Upload answers_graph_rag.txt, answers_qwen_rag.txt, answers_camelbert_rag.txt for scoring."
