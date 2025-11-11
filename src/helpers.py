
import sys, types, torch
from dataclasses import dataclass
from typing import Tuple


data_types = types.ModuleType("data_types")

@dataclass(frozen=True)
class Query:
    id: str
    question: str
    answers: Tuple[str, ...]

@dataclass
class Passage:
    id: str
    text: str
    title: str
    source_doc_id: int

data_types.Query = Query
data_types.Passage = Passage
sys.modules["data_types"] = data_types

#  helpers module
helpers = types.ModuleType("helpers")

def adaptive_encode(model, texts, *, batch_start=256, min_batch=16, **encode_kwargs):
    """
    Robust wrapper for SentenceTransformer.encode().
    Tries batch_start, halves on CUDA OOM until min_batch, then raises.
    Works on CPU too (skips empty_cache).
    """
    bs = int(batch_start)
    last_err = None
    is_cuda = torch.cuda.is_available()

    while bs >= min_batch:
        try:
            if is_cuda:
                torch.cuda.empty_cache()
            return model.encode(
                texts,
                batch_size=bs,
                show_progress_bar=True,
                convert_to_tensor=True,
                **encode_kwargs
            )
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            bs //= 2
            print(f" OOM: retrying with batch_size={bs}")
    raise last_err if last_err else RuntimeError("adaptive_encode failed without CUDA OOM")

helpers.adaptive_encode = adaptive_encode
sys.modules["helpers"] = helpers

print("In-notebook modules ready: data_types, helpers")
