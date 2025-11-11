from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path
import json
import time
import numpy as np
from tqdm import tqdm

import config
from llm_interface import LLMInterface, CostTracker
from evaluator import Evaluator


def validate_config():
    assert config.D_CANDIDATES >= config.TOP_K_TO_LLM, "System D must retrieve more than it passes"
    assert 0 < config.D_THR_RELEVANCE <= 1, "Invalid D_THR_RELEVANCE"
    assert 0 < config.D_THR_SUPPORT <= 1, "Invalid D_THR_SUPPORT"
    assert 0 < config.BUDGET_USD, "Budget must be > 0"
    assert config.TOP_K_TO_LLM <= 20, "Too many docs to LLM — will waste tokens"
    print(" Config validated successfully.")

validate_config()


@dataclass
class ExperimentResult:
    query_id: str
    question: str
    prediction: str
    gold_answers: List[str]
    exact_match: float
    f1_score: float
    recall_at_5: float
    recall_at_20: float
    mrr_at_20: float
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    cost_usd: float


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


class ExperimentRunner:
    def __init__(
        self,
        output_dir: Path = Path("./results"),
        checkpoint_every: int = 100,
        llm: Optional[LLMInterface] = None,
        top_k_to_llm: Optional[int] = None,   # if None → use config.TOP_K_TO_LLM
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_every = checkpoint_every
        self.evaluator = Evaluator()
        self.llm = llm or LLMInterface()
        self.top_k_to_llm = top_k_to_llm or config.TOP_K_TO_LLM
        
        # CRITICAL: Set evaluation k for recall metrics
        self.eval_k = 20  # Always retrieve 20 for metrics

    def run_system(self, system, queries: List, corpus: List) -> List[Dict]:
        """
        FIXED VERSION:
        - Retrieves k=20 passages for evaluation
        - Sends only top_k_to_llm (12) to LLM
        - Passes all 20 to evaluator for recall@20
        """
        # fresh cost tracker per system
        self.llm.tracker = CostTracker(budget_limit=self.llm.tracker.budget_limit)

        sys_name = _safe_name(system.name)
        checkpoint_file = self.output_dir / f"{sys_name}_checkpoint.json"

        # resume if checkpoint exists
        if checkpoint_file.exists():
            print(f"Loading checkpoint: {checkpoint_file}")
            with open(checkpoint_file) as f:
                completed = json.load(f)
            start_idx = len(completed)
        else:
            completed = []
            start_idx = 0

        # index once per system/corpus
        if start_idx == 0:
            print(f"\n{'='*60}")
            print(f"Indexing corpus for {system.name}")
            print(f"{'='*60}")
            system.index(corpus)

        print(f"\n{'='*60}")
        print(f"Running: {system.name}")
        print(f"Queries: {len(queries)} (starting from {start_idx})")
        print(f"Retrieving k={self.eval_k} for evaluation (sending {self.top_k_to_llm} to LLM)")
        print(f"{'='*60}")

        results = completed.copy()

        for idx in tqdm(range(start_idx, len(queries)), desc=system.name):
            q = queries[idx]
 
            t0 = time.time()
            retrieved_all = system.retrieve(q.question, k=self.eval_k) #k=20
            retrieval_latency = (time.time() - t0) * 1000.0


            contexts_for_llm = [r.text for r in retrieved_all[: self.top_k_to_llm]]
            
            all_retrieved_texts = [r.text for r in retrieved_all]

            llm_resp = self.llm.answer_with_context(q.question, contexts_for_llm)

         
            metrics = self.evaluator.evaluate_all(
                prediction=llm_resp.answer,
                gold_answers=list(q.answers),
                retrieved_texts=all_retrieved_texts,  # <-- Pass all 20!
            )

            result = ExperimentResult(
                query_id=q.id,
                question=q.question,
                prediction=llm_resp.answer,
                gold_answers=list(q.answers),
                exact_match=metrics.exact_match,
                f1_score=metrics.f1_score,
                recall_at_5=metrics.recall_at_5,
                recall_at_20=metrics.recall_at_20,
                mrr_at_20=metrics.mrr_at_20,
                retrieval_latency_ms=retrieval_latency,
                llm_latency_ms=llm_resp.latency_ms,
                total_latency_ms=retrieval_latency + llm_resp.latency_ms,
                cost_usd=llm_resp.cost_usd,
            )

            results.append(asdict(result))

       
            if (idx + 1) % self.checkpoint_every == 0:
                self._save_checkpoint(checkpoint_file, results)
                print(f"\n Checkpoint: {idx + 1}/{len(queries)}")
                print(f" Cost so far: ${self.llm.tracker.total_cost:.4f}")

        # final save
        final_json = self.output_dir / f"{sys_name}_results.json"
        with open(final_json, "w") as f:
            json.dump(results, f, indent=2)

        if checkpoint_file.exists():
            checkpoint_file.unlink()

        print(f"\n {system.name} complete!")
        print(f" Total cost: ${self.llm.tracker.total_cost:.4f}")

        return results

    @staticmethod
    def _save_checkpoint(path: Path, data: List) -> None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        tmp.rename(path)

    @staticmethod
    def aggregate_results(results: List[Dict]) -> Dict[str, float]:
        if not results:
            return {
                "exact_match": 0.0, "f1_score": 0.0,
                "recall@5": 0.0, "recall@20": 0.0,
                "mrr@20": 0.0, "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0, "total_cost_usd": 0.0
            }

        em   = [r["exact_match"] for r in results]
        f1   = [r["f1_score"] for r in results]
        r5   = [r["recall_at_5"] for r in results]
        r20  = [r["recall_at_20"] for r in results]
        mrr  = [r["mrr_at_20"] for r in results]
        lat  = [r["total_latency_ms"] for r in results]
        cost = [r["cost_usd"] for r in results]

        return {
            "exact_match": float(np.mean(em) * 100.0),
            "f1_score": float(np.mean(f1) * 100.0),
            "recall@5": float(np.mean(r5) * 100.0),
            "recall@20": float(np.mean(r20) * 100.0),
            "mrr@20": float(np.mean(mrr)),
            "latency_p50_ms": float(np.percentile(lat, 50)),
            "latency_p95_ms": float(np.percentile(lat, 95)),
            "total_cost_usd": float(sum(cost)),
        }


print(" FIXED ExperimentRunner ready")
