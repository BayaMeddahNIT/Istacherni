# Istacherni: RAG Architectures Comparison & Selection

## 1. Introduction
The **Istacherni** project aimed to build a highly accurate Algerian Legal retrieval system. To achieve this, we developed, tested, and benchmarked four distinct Retrieval-Augmented Generation (RAG) architectures. This report outlines the mechanics of each architecture, their strengths and weaknesses in the context of Algerian law, and the justification for our foundational architectural choices.

---

## 2. The Four RAG Architectures

### 2.1 Standard Dense RAG (Vector Search)
*   **How it works:** Uses deep learning embedding models (e.g., BGE-M3) to convert both the user's query and the legal articles into high-dimensional numerical vectors. It retrieves articles based on "Semantic Similarity" (Cosine Distance) in a vector database like ChromaDB.
*   **Strengths:** Excellent at understanding the *meaning* or *concept* behind a query, even if the user uses synonyms or colloquial phrasing instead of strict legal terminology.
*   **Weaknesses in Legal Context:** Dense vectors naturally struggle with "exact match" requirements. If a user asks for "المادة 15" (Article 15), the dense model might retrieve Article 115 or Article 150 because the semantic concept of "an article number" is similar, leading to fatal citation errors.

### 2.2 Graph RAG
*   **How it works:** Transforms the legal dataset into a Knowledge Graph using `NetworkX`. Legal articles, domains (e.g., Commercial Law), penalties, and concepts become interconnected "Nodes." Retrieval relies on traversing "Edges" using algorithms like PageRank to find connected legal concepts.
*   **Strengths:** Highly effective for "multi-hop" reasoning. For example, if a query asks how a specific commercial crime affects civil liability, Graph RAG can trace the path from the Penal Code node to the Civil Code node.
*   **Weaknesses in Legal Context:** Extremely complex to construct and maintain. Building accurate relationships between thousands of specific Algerian laws requires intense manual mapping or flawless LLM extraction, which is prone to scaling issues.

### 2.3 Agentic RAG
*   **How it works:** Replaces the standard linear retrieval pipeline with an autonomous "Agent" (powered by Gemini 2.5 Flash). The agent is given tools (e.g., `search_bm25`, `search_dense`) and iteratively decides which tool to use, reading the results, and deciding if it needs to search again before answering.
*   **Strengths:** Capable of self-correction. If the first search returns irrelevant laws, the agent realizes the mistake and formulates a new search query automatically.
*   **Weaknesses in Legal Context:** Extremely slow and API-intensive. A single query might take 15-30 seconds and multiple API calls, which is unacceptable for a production-level, fast-response user interface.

### 2.4 BM25 (Sparse Keyword Retrieval)
*   **How it works:** A statistical algorithm that ranks documents based on the frequency and rarity of exact keywords appearing in both the query and the document. We enhanced this with custom Arabic tokenization to handle diacritics and prefixes.
*   **Strengths:** Absolute precision on exact terminology, Article numbers, and Law names. Highly computationally efficient.
*   **Weaknesses:** Lacks semantic understanding. If the user searches for "سرقة" (theft) but the law uses "اختلاس" (embezzlement), BM25 will fail to find the connection.

---

## 3. Why We Chose BM25 (Integrated into Hybrid RAG)

While our final system architecture is a **Hybrid RAG** (combining BGE-M3 and BM25 via Reciprocal Rank Fusion), **BM25 forms the critical, non-negotiable foundation** of the Istacherni dataset retrieval for several mandatory reasons:

### Reason 1: The "Article Number" Problem
In Algerian Law, the difference between "المادة 374" (Article 374) and "المادة 375" (Article 375) could mean the difference between a fine and 5 years in prison. Dense AI models treat numbers as fuzzy semantic concepts and frequently mix them up. **BM25 guarantees exact matching.** If the user searches for "المادة 374", BM25 forcefully pushes that exact article to the top of the results.

### Reason 2: Deterministic Legal Terminology
Algerian law (extracted from the JORADP Official Gazette) uses highly specific, rigid phrasing. Lawyers and legal professionals search using exact terminology. BM25 excels at matching these precise, multi-word legal phrases exactly as they are written in the code, whereas Dense RAG might retrieve a conceptually similar but legally irrelevant article.

### Reason 3: Computational Speed & Independence
BM25 relies entirely on term frequencies (TF-IDF variations). It requires no GPU, no heavy neural network weights, and executes in milliseconds on standard hardware. By pairing BM25 with BGE-M3, we ensured that even if the semantic AI model fails to find a connection, the ultra-fast keyword engine acts as a flawless safety net.

### Conclusion
By relying on **BM25** as our core exact-match engine, and pairing it with **Dense RAG** to handle semantic synonyms, we achieved a retrieval architecture that is both highly intelligent (Dense) and legally precise (BM25), eliminating the hallucination risks of Agentic RAG and the immense overhead of Graph RAG.
