from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Initialize local LLM and Embeddings using Ollama qwen2:7b
eval_llm = ChatOllama(model="qwen2:7b")
eval_embeddings = OllamaEmbeddings(model="qwen2:7b")

# Configure Ragas metrics to use Ollama
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
for m in metrics:
    m.llm = eval_llm
    if hasattr(m, "embeddings"):
        m.embeddings = eval_embeddings

# Simple test dataset (2 questions)
data = {
    "question": [
        "هل النصب في التجارة يُعاقب عليه القانون؟",
        "شخص باعني منتج مزور، هل هذا يعتبر جريمة؟"
    ],
    "answer": [
        "نعم، النصب في التجارة جريمة مُجرَّمة صراحةً بموجب المادة 372 من قانون العقوبات الجزائري.",
        "نعم، بيع منتج مزور يُشكّل جريمة غش تجاري بموجب المادة 429 من قانون العقوبات."
    ],
    "contexts": [
        ["المادة 372 من قانون العقوبات الجزائري (الأمر 66-156) تنص على عقوبة الحبس."],
        ["المادة 429 من قانون العقوبات تنص على معاقبة كل من يخدع أو يحاول خدع المتعاقد."]
    ],
    "ground_truth": [
        "نعم، النصب في التجارة جريمة مُجرَّمة صراحةً بموجب المادة 372 من قانون العقوبات الجزائري.",
        "نعم، بيع منتج مزور يُشكّل جريمة غش تجاري بموجب المادة 429 من قانون العقوبات."
    ]
}

dataset = Dataset.from_dict(data)

print("Starting evaluation...")
result = evaluate(
    dataset=dataset,
    metrics=metrics
)
print("Result:")
print(result)
