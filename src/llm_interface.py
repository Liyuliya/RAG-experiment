import time
from dataclasses import dataclass
from typing import List


@dataclass
class LLMResponse:
    answer: str
    latency_ms: float
    tokens_used: int
    cost_usd: float

class CostTracker:
    def __init__(self, budget_limit: float = 5.0):
        self.total_cost = 0.0
        self.budget_limit = float(budget_limit)
        self.call_count = 0

    def track(self, cost: float):
        self.total_cost += float(cost)
        self.call_count += 1
        if self.total_cost > self.budget_limit * 0.9:
            print(f"⚠️ Approaching budget limit: ${self.total_cost:.2f}/{self.budget_limit:.2f}")

    def summary(self) -> str:
        return f"Total: ${self.total_cost:.4f} | Calls: {self.call_count}"

class LLMInterface:
    # prices per 1M tokens (adjust if needed)
    PRICE_INPUT = 0.150
    PRICE_OUTPUT = 0.600

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 256,
        max_retries: int = 3
    ):
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.max_retries = int(max_retries)

        api_key = UserSecretsClient().get_secret("rag").strip()
        self.client = OpenAI(api_key=api_key)
        self.tracker = CostTracker()

    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        numbered = "\n".join([f"[Doc {i+1}] {c}" for i, c in enumerate(contexts)])
        return (
            "You are a truthful assistant. Ground the answer ONLY in the documents.\n"
            "Cite doc numbers in square brackets where used (e.g., [Doc 2]).\n"
            "If evidence is insufficient, say so.\n\n"
            f"{numbered}\n\nQuestion: {question}\nAnswer:"
        )

    def answer_with_context(self, question: str, contexts: List[str]) -> LLMResponse:
        prompt = self._build_prompt(question, contexts)

        last_err = None
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                latency_ms = (time.time() - start) * 1000.0

                usage = response.usage
                cost = (
                    (usage.prompt_tokens / 1_000_000) * self.PRICE_INPUT
                    + (usage.completion_tokens / 1_000_000) * self.PRICE_OUTPUT
                )
                self.tracker.track(cost)

                return LLMResponse(
                    answer=response.choices[0].message.content.strip(),
                    latency_ms=latency_ms,
                    tokens_used=usage.total_tokens,
                    cost_usd=float(cost),
                )
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                print(f"Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)

        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")
