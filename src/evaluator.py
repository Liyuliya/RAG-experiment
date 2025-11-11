import string
import re
import collections
from dataclasses import dataclass
from typing import List


@dataclass
class EvaluationMetrics:
    exact_match: float
    f1_score: float
    recall_at_5: float
    recall_at_20: float
    mrr_at_20: float


class Evaluator:
    @staticmethod
    def normalize_answer(text: str) -> str:
        if text is None:
            return ""
        text = text.lower().strip()
        text = text.replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-")
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def exact_match(cls, prediction: str, gold_answers: List[str]) -> float:
        """Lenient EM: any normalized gold must appear within normalized prediction."""
        p = cls.normalize_answer(prediction or "")
        for g in gold_answers or []:
            if cls.normalize_answer(g) in p:
                return 1.0
        return 0.0

    @classmethod
    def f1_score(cls, prediction: str, gold_answers: List[str]) -> float:
        p_tokens = cls.normalize_answer(prediction).split()
        if not p_tokens:
            return 1.0 if any(cls.normalize_answer(g) == "" for g in gold_answers or []) else 0.0

        p_counts = collections.Counter(p_tokens)
        best = 0.0
        for g in gold_answers or []:
            g_counts = collections.Counter(cls.normalize_answer(g).split())
            if not g_counts:
                continue
            common = p_counts & g_counts
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = num_same / sum(p_counts.values())
            recall    = num_same / sum(g_counts.values())
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
            best = max(best, f1)
        return best 

    @staticmethod
    def recall_at_k(retrieved_texts: List[str], gold_answers: List[str], k: int) -> float:
        # per-passage check to avoid cross-boundary matches (safer than joining)
        gold_norms = [Evaluator.normalize_answer(a) for a in (gold_answers or [])]
        for text in (retrieved_texts or [])[:k]:
            t = Evaluator.normalize_answer(text)
            if any(g in t for g in gold_norms):
                return 1.0
        return 0.0

    @staticmethod
    def mrr_at_20(retrieved_texts: List[str], gold_answers: List[str]) -> float:
        gold_norms = [Evaluator.normalize_answer(a) for a in (gold_answers or [])]
        for rank, text in enumerate((retrieved_texts or [])[:20], start=1):
            t = Evaluator.normalize_answer(text)
            if any(g in t for g in gold_norms):
                return 1.0 / rank
        return 0.0

    @classmethod
    def evaluate_all(cls, prediction: str, gold_answers: List[str], retrieved_texts: List[str]) -> "EvaluationMetrics":
        return EvaluationMetrics(
            exact_match=cls.exact_match(prediction, gold_answers),
            f1_score=cls.f1_score(prediction, gold_answers),
            recall_at_5=cls.recall_at_k(retrieved_texts, gold_answers, k=5),
            recall_at_20=cls.recall_at_k(retrieved_texts, gold_answers, k=20),
            mrr_at_20=cls.mrr_at_20(retrieved_texts, gold_answers),
        )

print(" Evaluator ready")
