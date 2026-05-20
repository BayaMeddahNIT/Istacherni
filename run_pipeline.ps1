Write-Host "Setting UTF-8 Encodings..."
$env:PYTHONIOENCODING="utf-8"

Write-Host "Clearing old answers..."
Clear-Content answers_graph_rag.txt -ErrorAction SilentlyContinue

Write-Host "Running Graph RAG generation (will take ~6 hours)..."
cmd.exe /c ".\.venv\Scripts\python.exe run_graph_rag.py > run_graph_rag_local_log.txt 2>&1"

Write-Host "Running LLM Evaluation (will take ~3 hours)..."
cmd.exe /c ".\.venv\Scripts\python.exe evaluate_rag_models.py > eval_log.txt 2>&1"

Write-Host "Done!"
