from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import time
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from data_types import Query, Passage
from helpers import adaptive_encode


@dataclass(frozen=True)
class RetrievalResult:
    passage_id: str
    text: str
    score: float
    rank: int


class Retriever(ABC):
    @abstractmethod
    def index(self, corpus: List[Passage]) -> None: ...
    @abstractmethod
    def retrieve(self, query: str, k: int = 20) -> List[RetrievalResult]: ...
    @property
    @abstractmethod
    def name(self) -> str: ...


# A: Dense Contriever
class SystemA_Contriever(Retriever):
    def __init__(self, device: str = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[System A] Loading Contriever on {device}...")
        self.encoder = SentenceTransformer("facebook/contriever", device=device)
        self.corpus: List[Passage] = []
        self.embeddings: Optional[torch.Tensor] = None

    @property
    def name(self) -> str:
        return "Dense Contriever Retrieval"

    def index(self, corpus: List[Passage]) -> None:
        print(f"\n[System A] Indexing {len(corpus)} passages...")
        self.corpus = corpus
        texts = [p.text for p in corpus]
        t0 = time.time()
        self.embeddings = adaptive_encode(self.encoder, texts, batch_start=256)
        print(f"✓ Indexed in {time.time()-t0:.2f}s")

    def retrieve(self, query: str, k: int = 20) -> List[RetrievalResult]:
        assert self.embeddings is not None, "Call .index(corpus) before retrieve()"
        q = self.encoder.encode(query, convert_to_tensor=True)        # [d]
        scores = torch.matmul(q, self.embeddings.T)                   # [N]
        topk = min(k, scores.shape[-1])
        top_scores, top_idx = torch.topk(scores, topk)                # [k], [k]
        return [
            RetrievalResult(
                passage_id=self.corpus[int(i)].id,
                text=self.corpus[int(i)].text,
                score=float(s),
                rank=r,
            )
            for r, (i, s) in enumerate(zip(top_idx.cpu().tolist(), top_scores.cpu().tolist()), start=1)
        ]


# B: BM25 + BGE reranker 
class SystemB_HybridBM25BGE(Retriever):
    def __init__(self, device: str = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[System B] Loading BGE reranker on {device}...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-large", device=device)
        self.corpus: List[Passage] = []
        self.bm25: Optional[BM25Okapi] = None

    @property
    def name(self) -> str:
        return "Hybrid BM25 + BGE Reranker"

    def index(self, corpus: List[Passage]) -> None:
        print(f"\n[System B] Building BM25 index for {len(corpus)} passages...")
        self.corpus = corpus
        tokenized = [p.text.lower().split() for p in tqdm(corpus, desc="Tokenizing")]
        t0 = time.time()
        self.bm25 = BM25Okapi(tokenized)
        print(f"✓ Indexed in {time.time()-t0:.2f}s")

    def retrieve(self, query: str, k: int = 20) -> List[RetrievalResult]:
        assert self.bm25 is not None, "Call .index(corpus) before retrieve()"
        q_tok = query.lower().split()
        bm25_scores = self.bm25.get_scores(q_tok)                     # [N]
        top_100 = np.argsort(bm25_scores)[-100:][::-1]                # best 100
        pairs = [[query, self.corpus[int(i)].text] for i in top_100]
        rerank_scores = self.reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        topk_idx = np.argsort(rerank_scores)[-k:][::-1]
        return [
            RetrievalResult(
                passage_id=self.corpus[int(top_100[i])].id,
                text=self.corpus[int(top_100[i])].text,
                score=float(rerank_scores[i]),
                rank=r,
            )
            for r, i in enumerate(topk_idx, start=1)
        ]


# C: Dense BGE-large 
class SystemC_BGE(Retriever):
    def __init__(self, device: str = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[System C] Loading BGE-large on {device}...")
        self.encoder = SentenceTransformer("BAAI/bge-large-en", device=device)
        self.corpus: List[Passage] = []
        self.embeddings: Optional[torch.Tensor] = None

    @property
    def name(self) -> str:
        return "Dense BGE-large"

    def index(self, corpus: List[Passage]) -> None:
        print(f"\n[System C] Indexing {len(corpus)} passages...")
        self.corpus = corpus
        texts = [p.text for p in corpus]
        t0 = time.time()
        self.embeddings = adaptive_encode(self.encoder, texts, batch_start=128)
        print(f"✓ Indexed in {time.time()-t0:.2f}s")

    def retrieve(self, query: str, k: int = 20) -> List[RetrievalResult]:
        assert self.embeddings is not None, "Call .index(corpus) before retrieve()"
        q = self.encoder.encode(query, convert_to_tensor=True)        # [d]
        scores = torch.matmul(q, self.embeddings.T)                   # [N]
        topk = min(k, scores.shape[-1])
        top_scores, top_idx = torch.topk(scores, topk)
        return [
            RetrievalResult(
                passage_id=self.corpus[int(i)].id,
                text=self.corpus[int(i)].text,
                score=float(s),
                rank=r,
            )
            for r, (i, s) in enumerate(zip(top_idx.cpu().tolist(), top_scores.cpu().tolist()), start=1)
        ]


# D: Reflective filter on A's candidates 
class SystemD_ReflectiveContriever(Retriever):
    def __init__(
        self,
        device: str = None,
        top_k_candidates: int = 60,
        thr_relevance: float = 0.55,
        thr_support: float = 0.55,
        alpha_blend: float = 0.50,
        max_passages_to_llm: int = 12,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[System D] Loading Contriever on {device}...")
        self.encoder = SentenceTransformer("facebook/contriever", device=device)
        self.corpus: List[Passage] = []
        self.embeddings: Optional[torch.Tensor] = None
        self.top_k_candidates = int(top_k_candidates)
        self.thr_relevance = float(thr_relevance)
        self.thr_support   = float(thr_support)
        self.alpha_blend   = float(alpha_blend)
        self.max_passages_to_llm = int(max_passages_to_llm)

    @property
    def name(self) -> str:
        return "Reflective Evidence Filtering (Contriever)"

    def index(self, corpus: List[Passage]) -> None:
        print(f"\n[System D] Indexing {len(corpus)} passages (Contriever)...")
        self.corpus = corpus
        texts = [p.text for p in corpus]
        t0 = time.time()
        self.embeddings = adaptive_encode(self.encoder, texts, batch_start=256)
        print(f"✓ Indexed in {time.time()-t0:.2f}s")

    def _squash_base(self, x: torch.Tensor) -> float:
        # soften dot-product scale to ~0..1
        return (1.0 / (1.0 + torch.exp(-x / 5.0))).item()

    def retrieve(self, query: str, k: int = 20) -> List[RetrievalResult]:
        if self.embeddings is None:
            raise RuntimeError("System D not indexed. Call .index(corpus) first.")

        # base candidates (dot product, unnormalized)
        q_raw = self.encoder.encode(query, convert_to_tensor=True)
        base = torch.matmul(q_raw, self.embeddings.T)              
        topK = int(min(self.top_k_candidates, base.numel()))
        base_vals, base_idx = torch.topk(base, topK)                 

        # reflective critic (cosine on those candidates)
        q_norm = F.normalize(q_raw, dim=-1).unsqueeze(0)      
        cand_norm = F.normalize(self.embeddings[base_idx], dim=-1)   
        cos = torch.matmul(q_norm, cand_norm.T).clamp(-1, 1)         

        kept = []
        for j, idx in enumerate(base_idx.tolist()):
            rel = float((cos[0, j].item() + 1.0) / 2.0)              # [-1,1] → [0,1]
            sup = max(0.0, min(1.0, 0.95 * rel + 0.025))            
            if rel >= self.thr_relevance and sup >= self.thr_support:
                fused = self.alpha_blend * (0.5 * (rel + sup)) + (1.0 - self.alpha_blend) * self._squash_base(base_vals[j])
                kept.append((int(idx), float(fused)))

        
        if not kept:
            kept = [(int(i), float(self._squash_base(s))) for i, s in zip(base_idx.tolist(), base_vals)]
        kept.sort(key=lambda x: x[1], reverse=True)
        kept = kept[: min(self.max_passages_to_llm, k)]

        results: List[RetrievalResult] = []
        for rank, (pi, fused) in enumerate(kept, start=1):
            p = self.corpus[int(pi)]
            results.append(RetrievalResult(passage_id=p.id, text=p.text, score=float(fused), rank=rank))
        return results


print(" Retrieval Systems ready")
