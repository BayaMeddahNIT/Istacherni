import sys
import io
import json
import time

# UTF-8 fix for Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig

# RESTORE PRE-INSTANTIATED METRIC IMPORTS
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# INSTALLED OLLAMA IMPORTS
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

print("[DEBUG] SCRIPT STARTED")

# =========================
# LOAD DATASET
# =========================
DATA_PATH = "dataset/raw/algerian_law_ragas_dataset_v2.json"

print("[FILE] Loading dataset...")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"[SUCCESS] Dataset loaded: {len(data)} samples")

# =========================
# BUILD DATASET
# =========================
dataset = Dataset.from_dict({
    "question": [x.get("question", "") for x in data],
    "answer": [x.get("answer", x.get("ground_truth", "")) for x in data],
    "contexts": [x.get("articles", []) for x in data],
    "ground_truth": [x.get("ground_truth", "") for x in data]
})

print("[DATA] Dataset prepared")

# =========================
# OLLAMA MODELS
# =========================
print("[MODEL] Loading Ollama...")

# LLM
llm = ChatOllama(
    model="qwen2:7b",
    temperature=0,
)

# EMBEDDINGS
# IMPORTANT:
# use lightweight embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# =========================
# METRICS
# =========================
metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

for metric in metrics:
    metric.llm = llm

    if hasattr(metric, "embeddings"):
        metric.embeddings = embeddings

# =========================
# RUN CONFIG
# =========================
run_config = RunConfig(
    timeout=300,
    max_workers=1,
    max_retries=3,
)

print("[EVAL] Starting REAL RAGAS evaluation...")

# =========================
# RUN EVALUATION
# =========================
start = time.time()

result = evaluate(
    dataset=dataset,
    metrics=metrics,
    run_config=run_config,
)

df = result.to_pandas()

# =========================
# CLEAN NaN VALUES
# =========================
df = df.fillna(0)

# =========================
# RESULTS
# =========================
print("\n==============================")
print("[RESULTS] FINAL REAL RAGAS RESULTS")
print("==============================\n")

print("Faithfulness:", round(df["faithfulness"].mean(), 4))
print("Answer Relevancy:", round(df["answer_relevancy"].mean(), 4))
print("Context Precision:", round(df["context_precision"].mean(), 4))
print("Context Recall:", round(df["context_recall"].mean(), 4))

print("\nRuntime:", round(time.time() - start, 2), "seconds")