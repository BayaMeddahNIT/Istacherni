"""
camelbert_classifier.py
-----------------------
Zero-shot domain classifier using CamelBERT embeddings.
Provides a fast, sub-second alternative to LLM-based query classification.
"""

# ── Offline Enforcement ────────────────────────────────────────────────────────
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Mapping of domains to rich Arabic descriptions to improve semantic matching
DOMAINS = {
    "Civil Law": "القانون المدني شروط صحة العقد الرضا الأهلية المحل السبب الالتزام حقوق وواجبات البيع الإيجار التعويض",
    "Penal Code": "قانون العقوبات الجرائم الجنايات الجنح المخالفات السرقة القتل السجن الحبس الغرامة المصادرة",
    "Labor Law": "قانون العمل العمال الموظف الأجر الطرد التسريح النقابات عقد العمل العطلة حقوق العمال",
    "Commercial Law": "القانون التجاري الشركات السجل التجاري التاجر الإفلاس الشيكات السندات الأعمال التجارية",
    "Code of Civil and Administrative Procedures": "قانون الإجراءات المدنية والإدارية الدعوى المحكمة الطعن الاستئناف الإثبات الإجراءات التنفيذ",
}

class CamelBERTClassifier:
    def __init__(self, model_name="CAMeL-Lab/bert-base-arabic-camelbert-msa"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.domain_embeddings = self._precompute_domain_embeddings()
        
    def _mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _embed(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings.cpu().numpy()

    def _precompute_domain_embeddings(self):
        keys = list(DOMAINS.keys())
        descriptions = [DOMAINS[k] for k in keys]
        embeddings = self._embed(descriptions)
        return {k: emb for k, emb in zip(keys, embeddings)}

    def classify(self, query: str) -> tuple[str, float, dict]:
        """
        Classifies the query and returns (best_domain, confidence_score, all_scores).
        """
        query_emb = self._embed([query])[0]
        
        # Calculate cosine similarity manually avoiding sklearn dependency
        best_domain = ""
        best_score = -1.0
        scores = {}
        
        norm_q = np.linalg.norm(query_emb)
        if norm_q == 0: norm_q = 1e-9
            
        for domain, emb in self.domain_embeddings.items():
            norm_e = np.linalg.norm(emb)
            if norm_e == 0: norm_e = 1e-9
            
            sim = np.dot(query_emb, emb) / (norm_q * norm_e)
            scores[domain] = float(sim)
            if sim > best_score:
                best_score = float(sim)
                best_domain = domain
                
        return best_domain, best_score, scores

# Singleton pattern
_classifier = None

def get_camelbert_classifier() -> CamelBERTClassifier:
    global _classifier
    if _classifier is None:
        model_name = os.getenv("CAMELBERT_MODEL", "CAMeL-Lab/bert-base-arabic-camelbert-msa")
        print(f"[CamelBERT] Initialising query classifier ({model_name})...")
        _classifier = CamelBERTClassifier(model_name)
    return _classifier

def get_camelbert_domain(query: str, threshold: float = 0.6) -> str:
    """Convenience method to get domain only if confidence meets threshold."""
    try:
        classifier = get_camelbert_classifier()
        best_domain, score, _ = classifier.classify(query)
        if score >= threshold:
            print(f"[CamelBERT] Confident domain match: {best_domain} (score: {score:.3f})")
            return best_domain
        else:
            print(f"[CamelBERT] Low confidence for {best_domain} (score: {score:.3f} < {threshold}). Falling back.")
            return ""
    except Exception as e:
        print(f"[CamelBERT] Error during classification: {e}")
        return ""
