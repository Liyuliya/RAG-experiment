from pathlib import Path
import torch


DATA_DIR = Path("/dpr-wikipedia-2018-100k")
CACHE_DIR   = Path("./cache")
RESULTS_DIR = Path("./results")
ARTIFACTS_DIR = Path("./artifacts")

SEED = 12

N_QUERIES   = 1000   
N_PASSAGES  = 100_000 

CHUNK_SIZE  = 100      
STRIDE      = 50         

#  Retrieval (shared)
EVAL_K       = 20      
TOP_K_TO_LLM = 12      
# Dense: Contriever
A_BATCH_SIZE = 256

# Hybrid: BM25 + BGE reranker
B_BM25_CANDIDATES = 100   
B_RERANK_BATCH    = 32

# Dense: BGE-large
C_BATCH_SIZE = 128

# D 
D_CANDIDATES     = 60     # top-K candidates from dense before filtering
D_THR_RELEVANCE  = 0.55  
D_THR_SUPPORT    = 0.55
D_ALPHA_BLEND    = 0.50  

# LLM 
LLM_MODEL       = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS  = 64
LLM_MAX_RETRIES = 3

# cost tracking
BUDGET_USD      = 5.00

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EMB_BATCH  = 256  # default encode batch; A uses 256, C overrides to 128

TEST_MODE = False       
TEST_SLICE_PASSAGES = 10_000

import types, sys
_config = types.ModuleType("config")
for _k, _v in list(globals().items()):
    if _k.isupper(): setattr(_config, _k, _v)
sys.modules["config"] = _config
print("✓ config module registered for import")
