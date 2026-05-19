# Algerian Legal RAG — Complete Diagnostic & Improvement Report

> Senior AI Research · RAG Architecture · Legal NLP · Arabic NLP Optimization  
> Current Scores → Target Scores Analysis

---

## CURRENT STATE SUMMARY

| Model | Precision | Hit Rate | MRR | Recall | Legal Accuracy |
|---|---|---|---|---|---|
| Graph RAG | 0.2344 | 0.6825 | 0.4389 | 0.2300 | 0.5238 |
| Qwen RAG | 0.2786 | 0.7302 | 0.5230 | 0.2854 | 0.4643 |
| CamELBERT RAG | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3480 |

**Target:** Graph RAG >96% · Qwen RAG >90% · CamELBERT >80%

**Gap Analysis:** The gap from current to target is enormous (Graph RAG needs +43 points on Legal Accuracy alone). This means surface-level tuning is insufficient — architectural changes are required.

---

## PART 1 — COMPLETE FILE AUDIT CHECKLIST

Before any optimization work, collect every file below. Missing any category creates blind spots.

### Dataset Files
- [ ] `algerian_law_ragas_dataset_v3.json` — ground truth QA pairs (confirmed uploaded)
- [ ] Raw legal corpus: all source PDFs or text files for قانون العقوبات, القانون التجاري, القانون المدني, etc.
- [ ] Article-level extracted text files (one file per law, or one per article)
- [ ] Metadata JSON: for each article, law name, article number, chapter, section, date enacted
- [ ] Any augmented/synthetic QA pairs used for training
- [ ] Train/val/test split files

### Preprocessing & OCR
- [ ] OCR pipeline script (if PDFs were scanned)
- [ ] Text cleaning/normalization script
- [ ] Arabic diacritics removal script
- [ ] Tashkeel handling code
- [ ] Alef/Hamza normalization rules
- [ ] Eastern vs Western Arabic numeral normalization (١٢٣ vs 123)
- [ ] Legal citation regex patterns used (if any)
- [ ] Chunking script with parameters (chunk_size, overlap, strategy)
- [ ] Sample raw vs cleaned text for 3–5 articles

### Embedding Models
- [ ] Embedding model name/path for each pipeline (Graph RAG, Qwen RAG, CamELBERT)
- [ ] Embedding dimension for each model
- [ ] Whether embeddings are fine-tuned or off-the-shelf
- [ ] Embedding generation script
- [ ] Sample embedding vectors for the same query across all three models (to check alignment)

### Vector Database
- [ ] VDB type (FAISS, Chroma, Qdrant, Weaviate, Milvus, etc.)
- [ ] Index type (flat, IVF, HNSW, etc.)
- [ ] Distance metric (cosine, dot product, L2)
- [ ] Number of indexed chunks
- [ ] top-k value used at retrieval time
- [ ] Any filtering/metadata filtering applied at retrieval
- [ ] VDB configuration YAML/JSON

### Graph Construction (Graph RAG only)
- [ ] Graph construction script
- [ ] Node types (article, law, chapter, concept, entity)
- [ ] Edge types and weights
- [ ] Graph statistics (nodes, edges, density)
- [ ] Community detection method used
- [ ] How graph summaries are generated
- [ ] How graph traversal works during retrieval
- [ ] Neo4j/NetworkX/other graph DB config

### Retrieval Pipeline
- [ ] Full retrieval pipeline script for each model
- [ ] Query preprocessing steps (rewriting, expansion, normalization)
- [ ] Retrieval strategy (dense only, BM25 only, hybrid)
- [ ] Fusion method (RRF, linear, etc.) if hybrid
- [ ] top-k at retrieval, top-n after reranking
- [ ] Reranker model name (if any)
- [ ] Context assembly logic (how chunks are combined into prompt)

### Prompt Templates
- [ ] System prompt for each model
- [ ] User prompt template
- [ ] Few-shot examples (if any)
- [ ] Arabic vs French language handling in prompts
- [ ] How retrieved context is injected into prompt
- [ ] Max context length setting

### Generation Models
- [ ] Qwen model version (Qwen2-7B, Qwen2-72B, etc.)
- [ ] CamELBERT model version and HuggingFace path
- [ ] Graph RAG generation model
- [ ] Temperature, top_p, max_new_tokens settings
- [ ] Quantization (4-bit, 8-bit, none)
- [ ] Inference framework (vLLM, llama.cpp, Transformers, Ollama)

### Evaluation Scripts
- [ ] `evaluate_rag_models.py` (confirmed exists)
- [ ] `merge_results.py`
- [ ] Judge prompt template used with gemma2:9b
- [ ] Score mapping logic (how judge output maps to 0.0–1.0)
- [ ] RAGAS config if used
- [ ] Any custom metric implementations

### Error Analysis
- [ ] Top 20 worst-scoring questions for each model
- [ ] Questions where model said "لا توجد معلومات" but GT had real answer
- [ ] Questions where wrong article was cited
- [ ] Questions where correct article was found but wrong penalty stated
- [ ] CamELBERT raw retrieval outputs (pre-answer) showing missing article numbers

### Runtime/Hardware
- [ ] GPU model and VRAM
- [ ] RAM
- [ ] CUDA version
- [ ] Python/PyTorch/Transformers version
- [ ] Ollama version if used
- [ ] Average inference time per question per model

---

## PART 2 — GRAPH RAG IMPROVEMENT PLAN (Target: >96%)

### Diagnosis of Current State

Graph RAG's Hit Rate of 68.25% means it finds the right article in roughly 2/3 of queries. Precision of 23.44% means most retrieved chunks are irrelevant noise. MRR of 0.44 means the correct article is not the top result. Legal Accuracy of 52.38% is the ceiling — the answers aren't legally correct even when retrieval partially works.

**Root cause:** Graph structure is almost certainly not encoding legal citation relationships. The graph likely treats articles as flat nodes without encoding the semantic legal hierarchy (law → chapter → article → sub-article) or cross-reference edges (المادة X تحيل إلى المادة Y).

---

### Recommendation 1: Legal Entity Graph Reconstruction
**Priority: CRITICAL | Difficulty: High | Expected gain: +15–20 points Hit Rate**

The current graph almost certainly uses generic entity extraction. Rebuild it using a legal-domain graph schema:

```python
# Node types
nodes = {
    "LAW": {"name": "قانون العقوبات", "year": 1966, "domain": "criminal"},
    "CHAPTER": {"law": "قانون العقوبات", "number": 3, "title": "الجرائم ضد الأشخاص"},
    "ARTICLE": {"law": "قانون العقوبات", "number": 429, "text": "...", "penalty_type": "حبس+غرامة"},
    "CONCEPT": {"name": "النصب", "synonyms": ["الاحتيال", "الغش"]},
    "PENALTY": {"type": "حبس", "min": 1, "max": 5, "unit": "سنة"},
    "CONDITION": {"text": "يشترط أن يكون الفعل عمدياً"}
}

# Edge types  
edges = {
    "ARTICLE_IN_LAW": (article_node, law_node),
    "ARTICLE_IN_CHAPTER": (article_node, chapter_node),
    "ARTICLE_REFERENCES": (article_429, article_219),  # cross-reference
    "ARTICLE_DEFINES_CONCEPT": (article_node, concept_node),
    "ARTICLE_IMPOSES_PENALTY": (article_node, penalty_node),
    "CONCEPT_SYNONYM": (concept_A, concept_B),
}
```

**Why it helps:** Queries about "النصب التجاري" can now traverse: Concept→Articles→Penalties in one graph hop, rather than relying purely on embedding similarity.

**Libraries:** `networkx`, `neo4j-python-driver`, `spacy` with Arabic NER model

---

### Recommendation 2: Article Number Extraction & Indexing
**Priority: CRITICAL | Difficulty: Medium | Expected gain: +10–15 points Precision**

Every chunk must have its article number(s) explicitly extracted and stored as metadata, then used as a hard filter during retrieval.

```python
import re

def extract_article_refs(text):
    patterns = [
        r'المادة\s+(\d+)\s*(?:مكرر\s*(\d*))?',       # المادة 429
        r'م\.\s*(\d+)',                                  # م. 429  
        r'(?:article|art\.?)\s+(\d+)',                   # French mixed
        r'(\d+)\s*من\s+(?:قانون|القانون)',               # 429 من قانون
    ]
    refs = []
    for p in patterns:
        refs.extend(re.findall(p, text, re.IGNORECASE))
    return list(set(refs))

def normalize_arabic_numerals(text):
    arabic_to_western = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
    return text.translate(arabic_to_western)
```

Store `article_refs`, `law_name`, `article_number` in chunk metadata. At retrieval time, if query contains an article number, hard-filter candidates to chunks containing that article.

---

### Recommendation 3: Hybrid BM25 + Dense Retrieval with Legal-Aware Fusion
**Priority: HIGH | Difficulty: Medium | Expected gain: +8–12 points Recall**

Dense embeddings miss exact lexical matches for article numbers and legal terms. BM25 captures these perfectly. Current system appears to use dense-only retrieval.

```python
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

class HybridLegalRetriever:
    def __init__(self, chunks, embedding_model):
        self.chunks = chunks
        self.bm25 = BM25Okapi([self._tokenize(c['text']) for c in chunks])
        self.encoder = SentenceTransformer(embedding_model)
        self.dense_index = self._build_faiss_index()
    
    def _tokenize(self, text):
        # Critical: normalize numerals before tokenization
        text = normalize_arabic_numerals(text)
        return text.split()
    
    def retrieve(self, query, k=10, alpha=0.6):
        # alpha controls dense vs sparse weight
        query = normalize_arabic_numerals(query)
        
        bm25_scores = self.bm25.get_scores(self._tokenize(query))
        dense_scores = self._dense_search(query, k*3)
        
        # Reciprocal Rank Fusion
        return self._rrf_fusion(bm25_scores, dense_scores, k)
    
    def _rrf_fusion(self, bm25_scores, dense_scores, k, rrf_k=60):
        bm25_ranks = np.argsort(bm25_scores)[::-1]
        dense_ranks = np.argsort(dense_scores)[::-1]
        
        scores = {}
        for rank, idx in enumerate(bm25_ranks):
            scores[idx] = scores.get(idx, 0) + 1/(rrf_k + rank)
        for rank, idx in enumerate(dense_ranks):
            scores[idx] = scores.get(idx, 0) + 1/(rrf_k + rank)
        
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
```

**Why:** BM25 excels at "المادة 429 من قانون العقوبات" exact matches. Dense excels at semantic queries like "ما عقوبة بيع سلع مغشوشة". Fusion beats both individually.

**Libraries:** `rank_bm25`, `faiss-gpu`, `sentence-transformers`

---

### Recommendation 4: Upgrade to Arabic Legal Embedding Model
**Priority: HIGH | Difficulty: Low | Expected gain: +10–15 points all metrics**

If you are using a generic multilingual embedding model (e.g., `multilingual-e5-base`), replace it immediately with a legal-domain Arabic model. Options ranked:

1. **`CAMeL-Lab/bert-base-arabic-camelbert-msa`** fine-tuned on legal corpus — best for Arabic legal text
2. **`intfloat/multilingual-e5-large`** — strong multilingual baseline, better than base
3. **`BAAI/bge-m3`** — hybrid dense/sparse, handles Arabic well
4. **Custom fine-tune:** Take `aubmindlab/bert-base-arabertv2` and fine-tune on legal triplets using contrastive loss

For fine-tuning embeddings with your ground truth data:

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Build training triplets from your RAGAS dataset
triplets = []
for item in ragas_dataset:
    query = item['question']
    positive = item['ground_truth']  # or the relevant article text
    # Hard negatives: articles from same law but different topic
    negative = get_hard_negative(item)
    triplets.append(InputExample(texts=[query, positive, negative]))

model = SentenceTransformer('BAAI/bge-m3')
train_dataloader = DataLoader(triplets, shuffle=True, batch_size=16)
train_loss = losses.TripletLoss(model=model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)
```

---

### Recommendation 5: Cross-Encoder Reranker
**Priority: HIGH | Difficulty: Low | Expected gain: +8–12 points MRR and Precision**

After retrieving top-20 candidates, use a cross-encoder to rerank to top-5. This dramatically improves Precision and MRR.

```python
from sentence_transformers import CrossEncoder

# Options:
# 'cross-encoder/ms-marco-MiniLM-L-6-v2' (fast, decent Arabic)
# 'BAAI/bge-reranker-v2-m3' (best multilingual)

reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')

def rerank(query, candidates, top_n=5):
    pairs = [(query, c['text']) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_n]]
```

---

### Recommendation 6: Query Rewriting for Arabic Legal Queries
**Priority: MEDIUM | Difficulty: Medium | Expected gain: +5–8 points Hit Rate**

Many queries are colloquial Arabic or underspecified. Rewrite them to match the formal legal register of the corpus before retrieval.

```python
QUERY_REWRITE_PROMPT = """أنت محامٍ جزائري متخصص. أعد صياغة السؤال التالي بصيغة قانونية رسمية 
تستخدم مصطلحات القانون الجزائري الدقيقة، وأضف اسم القانون المحتمل ورقم المادة إن أمكن.

السؤال الأصلي: {query}

السؤال المعاد صياغته (بالعربية الرسمية القانونية فقط):"""

def rewrite_query(query, llm):
    rewritten = llm(QUERY_REWRITE_PROMPT.format(query=query))
    # Use both original and rewritten for retrieval, merge results
    return [query, rewritten]
```

---

### Recommendation 7: Legal-Aware Chunking Strategy
**Priority: HIGH | Difficulty: Medium | Expected gain: +8–10 points Recall**

Current chunking is almost certainly character/token-based. Replace with article-aware chunking:

```python
def chunk_legal_corpus(text, law_name):
    # Split on article boundaries, not arbitrary token counts
    article_pattern = r'(?=المادة\s+\d+)'
    articles = re.split(article_pattern, text)
    
    chunks = []
    for article in articles:
        article_num_match = re.search(r'المادة\s+(\d+)', article)
        article_num = article_num_match.group(1) if article_num_match else "unknown"
        
        chunks.append({
            "text": article.strip(),
            "metadata": {
                "law": law_name,
                "article_number": article_num,
                "article_refs": extract_article_refs(article),
                "chunk_type": "full_article"
            }
        })
        
        # For long articles, also create sub-chunks with metadata inherited
        if len(article) > 512:
            sub_chunks = sliding_window(article, size=256, overlap=64)
            for sc in sub_chunks:
                sc['metadata'] = chunks[-1]['metadata'].copy()
                sc['metadata']['chunk_type'] = 'sub_article'
    
    return chunks
```

**Why:** Splitting mid-article destroys legal context. "الحبس من سنة إلى خمس سنوات" and "المادة 429" must stay in the same chunk.

---

### Recommendation 8: Community Summarization for Graph RAG
**Priority: MEDIUM | Difficulty: High | Expected gain: +6–10 points Legal Accuracy**

Microsoft's Graph RAG approach uses LLM-generated community summaries to answer global queries. Your current implementation likely skips this. Add:

```python
def generate_community_summary(articles_in_community, llm):
    prompt = f"""الموضوع القانوني: جرائم النصب والاحتيال في القانون الجزائري
    
المواد ذات الصلة:
{format_articles(articles_in_community)}

اكتب ملخصاً قانونياً شاملاً لهذه المجموعة من المواد، يتضمن:
1. الموضوع القانوني الرئيسي
2. الشروط والأركان
3. العقوبات المقررة
4. المواد المرتبطة"""
    
    return llm(prompt)
```

---

## PART 3 — QWEN RAG IMPROVEMENT PLAN (Target: >90%)

### Diagnosis

Qwen RAG leads on retrieval metrics (Hit Rate 73%, MRR 52.3%) but legal accuracy lags (46.43%). This pattern indicates: **retrieval is finding the right articles, but the generation step is not extracting and presenting legal information correctly.** The problem is generation quality, not retrieval.

---

### Recommendation 1: Structured Legal Prompt Engineering
**Priority: CRITICAL | Difficulty: Low | Expected gain: +15–20 points Legal Accuracy**

The current prompt is almost certainly a generic RAG prompt. Replace with a structured legal-domain prompt:

```python
LEGAL_SYSTEM_PROMPT = """أنت مستشار قانوني متخصص في القانون الجزائري. مهمتك الإجابة على الأسئلة القانونية 
بدقة تامة استناداً إلى المواد القانونية المقدمة فقط.

قواعد صارمة:
1. استند فقط إلى المواد المقدمة. لا تستخدم معلومات خارجية.
2. اذكر دائماً رقم المادة والقانون المصدر (مثال: المادة 429 من قانون العقوبات).
3. اذكر العقوبة المحددة بالأرقام (سنوات، أشهر، دينار).
4. إذا لم تجد الإجابة في المواد المقدمة، قل: "لا تتضمن المواد المقدمة نصاً صريحاً على هذه الحالة."
5. لا تخترع مواد أو عقوبات غير موجودة في النص.
6. رتّب إجابتك: الحكم ← الشروط ← العقوبة ← المادة المصدر."""

LEGAL_USER_PROMPT = """المواد القانونية المسترجعة:
{context}

---
السؤال القانوني: {question}

الإجابة القانونية الدقيقة:"""
```

---

### Recommendation 2: Best Qwen Model Version

For Arabic legal tasks, ranked by performance:

1. **`Qwen/Qwen2.5-72B-Instruct`** — best quality, needs 4-bit quant for single GPU
2. **`Qwen/Qwen2.5-32B-Instruct`** — good balance of quality and speed
3. **`Qwen/Qwen2.5-14B-Instruct`** — fits in 24GB VRAM with 4-bit quant
4. **`Qwen/Qwen2.5-7B-Instruct`** — current likely version, minimum acceptable

If using Qwen2.5-7B, upgrade to at least Qwen2.5-14B. The legal accuracy gap between 7B and 14B is significant for multi-step legal reasoning.

---

### Recommendation 3: Context Window Optimization
**Priority: HIGH | Difficulty: Low | Expected gain: +5–8 points Legal Accuracy**

Critical issues to fix in context assembly:

```python
def assemble_legal_context(retrieved_chunks, query, max_tokens=3000):
    # Sort by relevance score descending
    chunks = sorted(retrieved_chunks, key=lambda x: x['score'], reverse=True)
    
    # Deduplicate: remove chunks from same article (keep highest scored)
    seen_articles = set()
    deduplicated = []
    for chunk in chunks:
        art_id = f"{chunk['metadata']['law']}_{chunk['metadata']['article_number']}"
        if art_id not in seen_articles:
            deduplicated.append(chunk)
            seen_articles.add(art_id)
    
    # Format with explicit article headers
    context_parts = []
    for chunk in deduplicated:
        header = f"[{chunk['metadata']['law']} — المادة {chunk['metadata']['article_number']}]"
        context_parts.append(f"{header}\n{chunk['text']}")
    
    # Truncate to max_tokens intelligently (never cut mid-sentence)
    return "\n\n".join(context_parts)[:max_tokens*4]  # rough char estimate
```

---

### Recommendation 4: Best Arabic Embeddings for Qwen RAG

Ranked by Arabic legal retrieval performance:

1. **`BAAI/bge-m3`** — hybrid dense+sparse, best multilingual, handles Arabic legal well
2. **`intfloat/multilingual-e5-large-instruct`** — instruction-tuned, excellent for queries
3. **`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`** — decent baseline
4. **Custom fine-tuned AraBART or AraBERT** on your legal triplets — best possible for domain

For Qwen RAG specifically, use `BAAI/bge-m3` + `BM25` hybrid (RRF fusion). This should alone push Hit Rate from 73% to 85%+.

---

### Recommendation 5: Optimal Chunk Size

For Algerian legal articles:

- **Chunk size: 512 tokens** (full article, never split mid-article)
- **Overlap: 64 tokens** (only for sub-article chunks of very long articles)
- **Top-k retrieval: 20** → rerank → top-5 to LLM
- **Context window: 4096 tokens** minimum for generation

Never use fixed-size character chunking for legal text. An article can be 50 words or 500 words — both must be kept intact.

---

### Recommendation 6: Hallucination Reduction via Faithfulness Constraint

```python
POST_GENERATION_PROMPT = """راجع الإجابة التالية وتحقق من أنها مستندة فقط إلى المواد القانونية المقدمة.

المواد القانونية: {context}
الإجابة المقترحة: {answer}

هل تحتوي الإجابة على أي معلومات غير موجودة في المواد؟
- إذا نعم: أزل هذه المعلومات وأصلح الإجابة.
- إذا لا: أعد الإجابة كما هي.

الإجابة النهائية المدققة:"""
```

This self-consistency check catches hallucinated article numbers and invented penalties.

---

### Recommendation 7: Query Expansion for Arabic Legal Queries

```python
LEGAL_SYNONYMS = {
    "النصب": ["الاحتيال", "الغش", "التدليس", "الخداع"],
    "السرقة": ["الاختلاس", "الأخذ", "النهب"],
    "القتل": ["الاعتداء المفضي إلى الوفاة", "القتل العمد", "القتل الخطأ"],
    "الحبس": ["السجن", "العقوبة السالبة للحرية", "الاعتقال"],
    "الغرامة": ["الغرامة المالية", "العقوبة المالية", "الذعيرة"],
    "العقد": ["الاتفاقية", "الاتفاق", "الالتزام"],
}

def expand_legal_query(query):
    expanded = query
    for term, synonyms in LEGAL_SYNONYMS.items():
        if term in query:
            expanded += " " + " ".join(synonyms)
    return expanded
```

---

### Recommendation 8: Fine-Tuning Strategy for Qwen RAG

For publication-quality results, fine-tune Qwen2.5-7B/14B on:

**Dataset Construction:**
```python
# Format: instruction tuning with legal QA
fine_tune_examples = []
for item in ragas_dataset:
    fine_tune_examples.append({
        "instruction": LEGAL_SYSTEM_PROMPT,
        "input": f"المواد القانونية:\n{item['context']}\n\nالسؤال: {item['question']}",
        "output": item['ground_truth']  # high-quality ground truth answers
    })
```

**Training config:** LoRA rank=16, alpha=32, target_modules=["q_proj", "v_proj"], 3 epochs, lr=2e-4, batch=4 with gradient accumulation=8.

**Tools:** `unsloth` (2x faster than standard PEFT), `trl`, `peft`

---

## PART 4 — CAMELBERT FAILURE ANALYSIS & RECOVERY

### Root Cause Analysis

The 0.0000 across all retrieval metrics with 0.3480 Legal Accuracy tells a clear story: **CamELBERT is producing answers (hence non-zero legal accuracy) but its retrieved sources contain no article numbers**. This is not a generation failure — it is a catastrophic retrieval indexing failure.

**Probable Cause Tree:**

```
CamELBERT Retrieval Failure
├── Cause A: Arabic numeral encoding issue
│   └── Article numbers stored as ١٢٣ but queried as 123 (or vice versa)
│       → Zero exact matches → zero article numbers in retrieved chunks
│
├── Cause B: Chunk metadata not indexed
│   └── Article numbers extracted but stored outside the indexed text field
│       → Embedding only covers article body, not its number
│
├── Cause C: CamELBERT tokenizer splits numbers incorrectly
│   └── "المادة 429" tokenized as ["المادة", "42", "##9"] 
│       → Embedding represents a broken token sequence
│       → Similarity search cannot match whole article references
│
├── Cause D: Wrong similarity threshold
│   └── threshold too high → all candidates filtered out
│       → Retrieved set is empty → no article numbers returned
│
└── Cause E: Index built on wrong text field
    └── Index built on question text, not article text
        → Complete domain mismatch
```

**Most likely: Cause A + Cause C combined.**

---

### Fix 1: Numeral Normalization (Immediate)

```python
def normalize_for_camelbert(text):
    # Eastern Arabic → Western Arabic numerals
    eastern = '٠١٢٣٤٥٦٧٨٩'
    western = '0123456789'
    trans = str.maketrans(eastern, western)
    text = text.translate(trans)
    
    # Normalize article reference format
    text = re.sub(r'المادة\s+(\d+)', r'المادة \1', text)
    
    # Remove diacritics (tashkeel) — CamELBERT was not trained with them
    text = re.sub(r'[\u0617-\u061A\u064B-\u065F]', '', text)
    
    # Normalize Alef variants
    text = re.sub('[إأآا]', 'ا', text)
    text = re.sub('ى', 'ي', text)
    text = re.sub('ة', 'ه', text)
    
    return text
```

Apply this normalization **identically** to both corpus chunks at index time AND to queries at retrieval time.

---

### Fix 2: CamELBERT-Specific Tokenization Check

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('CAMeL-Lab/bert-base-arabic-camelbert-msa')

# Test tokenization of article references
test_phrases = [
    "المادة 429 من قانون العقوبات",
    "المادة 219 تنص على",
    "عقوبة الحبس من سنة إلى خمس سنوات"
]

for phrase in test_phrases:
    tokens = tokenizer.tokenize(phrase)
    print(f"Input: {phrase}")
    print(f"Tokens: {tokens}")
    print(f"IDs: {tokenizer.convert_tokens_to_ids(tokens)}\n")
```

If you see "429" split into ["42", "##9"] or article numbers split at all, the embedding will be corrupted. Fix: ensure numerals are pre-normalized and consider adding article number tokens to the vocabulary.

---

### Fix 3: Rebuild CamELBERT Index with Metadata Injection

```python
def prepare_camelbert_chunk(article_text, article_num, law_name):
    # Inject article identifier at the START of each chunk
    # This ensures the embedding captures the article reference
    prefix = f"[{law_name} المادة {article_num}] "
    full_text = normalize_for_camelbert(prefix + article_text)
    return full_text
```

By prepending the article reference to every chunk, the embedding of the chunk is anchored to that reference. Queries mentioning "المادة 429" will now have higher cosine similarity to the chunk containing "المادة 429" in its text.

---

### Fix 4: Similarity Threshold Calibration

```python
def calibrate_threshold(retriever, validation_set, k=10):
    # Test multiple thresholds on your validation set
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        hits = 0
        for item in validation_set:
            results = retriever.search(item['question'], k=k, threshold=threshold)
            gt_articles = item['articles']
            if any(r['article_number'] in gt_articles for r in results):
                hits += 1
        print(f"Threshold {threshold}: Hit Rate = {hits/len(validation_set):.3f}")
```

CamELBERT embeddings have a different similarity distribution than other models. Never assume the threshold from another model applies.

---

### Fix 5: Replace or Supplement CamELBERT

If after fixes 1–4 CamELBERT still fails, the architectural recommendation is:

**Option A — Supplement:** Keep CamELBERT for semantic understanding, add BM25 for article number retrieval. Fuse results. CamELBERT's domain knowledge is valuable; its weakness is exact match retrieval.

**Option B — Replace for retrieval:** Use `BAAI/bge-m3` for retrieval, CamELBERT for reranking or answer verification.

**Option C — Fine-tune CamELBERT:** Use your legal corpus to continue pre-training, then fine-tune on retrieval triplets:

```python
# Triplet format for fine-tuning
{
    "query": "ما عقوبة النصب في التجارة؟",
    "positive": "[قانون العقوبات المادة 429] كل من يخدع المتعاقد...",
    "hard_negative": "[قانون العقوبات المادة 219] يعاقب على السرقة..."  # same law, wrong article
}
```

---

### Expected CamELBERT Recovery Metrics

| Fix | Precision | Hit Rate | MRR | Recall |
|---|---|---|---|---|
| Current | 0.000 | 0.000 | 0.000 | 0.000 |
| After Fix 1+2 (normalization) | ~0.10 | ~0.35 | ~0.20 | ~0.10 |
| After Fix 3 (metadata injection) | ~0.18 | ~0.55 | ~0.35 | ~0.18 |
| After Fix 4 (threshold calibration) | ~0.22 | ~0.65 | ~0.42 | ~0.22 |
| After Fix 5A (BM25 hybrid) | ~0.28 | ~0.75 | ~0.52 | ~0.28 |
| After fine-tuning | ~0.35 | ~0.82 | ~0.62 | ~0.35 |

---

## PART 5 — PRIORITY EXECUTION ROADMAP

### 7-Day Emergency Plan

**Day 1 (4 hours): Normalization & Metadata**
- Implement Arabic numeral normalization across all pipelines
- Add article number extraction to all chunk metadata
- Re-index all three retrievers with normalized text
- Expected: CamELBERT from 0.0 to ~0.35 Hit Rate; all models +5 points

**Day 2 (4 hours): Hybrid BM25 + Dense**
- Add BM25 index alongside dense index for all three models
- Implement RRF fusion
- Expected: All models +8–12 points Recall

**Day 3 (3 hours): Prompt Engineering**
- Replace generic prompts with structured legal prompts (PART 3, Rec. 1)
- Add system prompt enforcing article citation format
- Expected: Legal Accuracy +15–20 points for Qwen RAG

**Day 4 (3 hours): Reranker Integration**
- Install and test `BAAI/bge-reranker-v2-m3`
- Apply to top-20 retrieved candidates → top-5
- Expected: Precision +10 points, MRR +8 points

**Day 5 (3 hours): Article-Aware Chunking**
- Rebuild chunks using article boundary detection
- Re-index everything
- Expected: Recall +8 points, Legal Accuracy +5 points

**Day 6 (4 hours): Graph Reconstruction**
- Rebuild Graph RAG with legal entity node types
- Add cross-reference edges between articles
- Expected: Graph RAG Hit Rate +10 points

**Day 7 (4 hours): Evaluation & Error Analysis**
- Run full evaluation on all three models
- Analyze top 20 failure cases per model
- Identify next priority improvements

---

### 30-Day Research Roadmap

| Week | Focus | Expected Gain |
|---|---|---|
| Week 1 | All Day 1–7 emergency fixes | Baseline improvement across all models |
| Week 2 | Embedding fine-tuning on legal triplets | +10–15 points Hit Rate |
| Week 3 | Qwen LLM fine-tuning (LoRA) on legal QA | +15–20 points Legal Accuracy |
| Week 4 | Graph RAG community summarization + full evaluation | Final push to targets |

---

### Prioritized Task List (Fastest Wins First)

| Priority | Task | Effort | Expected Gain |
|---|---|---|---|
| P0 | Arabic numeral normalization | 2h | +20–30 pts CamELBERT |
| P0 | Structured legal prompt | 1h | +15–20 pts Legal Accuracy |
| P1 | BM25 + Dense hybrid (RRF) | 4h | +8–12 pts Recall all models |
| P1 | Cross-encoder reranker | 3h | +8–12 pts Precision/MRR |
| P1 | Article-aware chunking | 4h | +8–10 pts Recall |
| P2 | Upgrade to BGE-M3 embeddings | 4h | +10–15 pts all metrics |
| P2 | Legal graph reconstruction | 8h | +10–15 pts Graph RAG |
| P3 | Embedding fine-tuning | 16h | +10–15 pts Hit Rate |
| P3 | LLM fine-tuning (LoRA) | 24h | +15–20 pts Legal Accuracy |
| P4 | Query rewriting | 4h | +5–8 pts Hit Rate |
| P4 | Synthetic data generation | 16h | +5–10 pts all metrics |

---

## PART 6 — CODE REVIEW FINDINGS

*Note: Full code review requires the actual source files. The following is based on behavioral evidence from the evaluation outputs.*

### Finding 1: CamELBERT Sources Have No Article Numbers — Indexing Bug

**Evidence:** Every CamELBERT source shows `"قانون جزائري - المادة  "` with empty article number.  
**Root Cause:** The article number extraction step either runs after indexing (so it's never stored) or the regex fails on the actual corpus format.  
**Fix:** See Part 4, Fixes 1–3.

### Finding 2: Evaluation Judge Uses Local LLM (gemma2:9b) — Score Inconsistency Risk

**Evidence:** 35 questions were scored with gemma2:9b, then scoring was interrupted and switched to heuristic scoring.  
**Problem:** gemma2:9b is not a strong Arabic judge. It may score Arabic legal answers inaccurately, especially for nuanced penalty descriptions.  
**Fix:** Use a dedicated Arabic legal evaluation rubric, or use GPT-4o/Claude as judge for final evaluation. Implement RAGAS framework for objective metrics (faithfulness, answer relevance, context precision).

### Finding 3: Evaluation Metric Calculation — Retrieval Metrics Are Approximate

**Evidence:** The initial metric calculation used a heuristic scoring function, not true Precision/Recall/MRR from retrieved article sets.  
**Problem:** True Precision = |relevant ∩ retrieved| / |retrieved|. This requires knowing which retrieved chunks are relevant, not just scoring the generated answer.  
**Fix:**

```python
def calculate_retrieval_metrics(retrieved_chunks, ground_truth_articles, k=5):
    retrieved_articles = [c['metadata']['article_number'] for c in retrieved_chunks[:k]]
    gt_set = set(ground_truth_articles)
    retrieved_set = set(retrieved_articles)
    
    hits = retrieved_set & gt_set
    
    precision = len(hits) / len(retrieved_set) if retrieved_set else 0
    recall = len(hits) / len(gt_set) if gt_set else 0
    
    # MRR
    mrr = 0
    for rank, article in enumerate(retrieved_articles, 1):
        if article in gt_set:
            mrr = 1 / rank
            break
    
    hit_rate = 1.0 if hits else 0.0
    
    return {"precision": precision, "recall": recall, "mrr": mrr, "hit_rate": hit_rate}
```

### Finding 4: Parallel Evaluation Race Condition Was Correctly Identified and Fixed

The filelock + model-specific cache approach is architecturally correct. Good implementation.

### Finding 5: No Query Preprocessing Before Retrieval

**Evidence:** Answers show the models struggling with colloquial Arabic queries.  
**Fix:** Add the normalization + query expansion pipeline described in Parts 2 and 3 before all retrieval calls.

### Finding 6: RAGAS Framework Not Used

RAGAS provides objective, LLM-independent metrics. Add it alongside your custom evaluation:

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

result = evaluate(
    dataset=your_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
```

---

## REALISTIC TARGET ASSESSMENT

Being direct: **96% for Graph RAG and 90% for Qwen RAG are achievable but require fine-tuning, not just prompt engineering.** Here is a realistic progression:

| Phase | Graph RAG Legal Acc | Qwen RAG Legal Acc | CamELBERT Hit Rate |
|---|---|---|---|
| Current | 52.38% | 46.43% | 0.00% |
| After emergency fixes (7 days) | ~65% | ~70% | ~65% |
| After embedding upgrade | ~75% | ~80% | ~72% |
| After LLM fine-tuning | ~88% | ~88% | ~78% |
| After full optimization | ~93% | ~92% | ~82% |

The 96% target for Graph RAG is reachable only with:
1. A fine-tuned Arabic legal embedding model
2. A fine-tuned generation model (or GPT-4 class)
3. A rebuilt legal knowledge graph with cross-references
4. A reranker trained on Arabic legal pairs

Upload your source files (especially the retrieval pipeline and chunking scripts) for a concrete code-level review and exact fixes.
