from pathlib import Path
from typing import List
import pickle


SEED = 12
DATA_DIR = Path("/dpr-wikipedia-2018-100k")
LOCAL_CACHE  = Path("./cache"); LOCAL_CACHE.mkdir(parents=True, exist_ok=True)



def _query_from_dict(d: dict) -> Query:
    ans = d.get("answers", d.get("answer", []))
    if isinstance(ans, dict) and "text" in ans:
        ans = ans["text"]
    if isinstance(ans, str):
        ans = [ans]
    return Query(
        id=str(d.get("id", "")),
        question=str(d.get("question", d.get("title", ""))),
        answers=tuple(ans),
    )

def _coerce_queries(objs) -> List[Query]:
    out = []
    for o in objs:
        if isinstance(o, Query):
            out.append(o)
        elif isinstance(o, dict):
            out.append(_query_from_dict(o))
        else:
            out.append(Query(
                id=str(getattr(o, "id", "")),
                question=str(getattr(o, "question", "")),
                answers=tuple(getattr(o, "answers", ()))
            ))
    return out

def _coerce_passages(objs) -> List[Passage]:
    out = []
    for o in objs:
        if isinstance(o, Passage):
            out.append(o)
        elif isinstance(o, dict):
            out.append(Passage(
                id=str(o.get("id", "")),
                text=str(o.get("text", "")),
                title=str(o.get("title", "")),
                source_doc_id=int(o.get("source_doc_id", 0)),
            ))
        else:
            out.append(Passage(
                id=str(getattr(o, "id", "")),
                text=str(getattr(o, "text", "")),
                title=str(getattr(o, "title", "")),
                source_doc_id=int(getattr(o, "source_doc_id", 0)),
            ))
    return out

def load_queries_fixed(n_samples: int = 1000) -> List[Query]:
    
    queries_pkl = DATA_DIR / f"queries_nq_open_validation_{n_samples}.pkl"
    if queries_pkl.exists():
        with open(queries_pkl, "rb") as f:
            qs = pickle.load(f)
        return _coerce_queries(qs)

    cache_pkl = LOCAL_CACHE / f"queries_nq_open_validation_{n_samples}.pkl"
    if cache_pkl.exists():
        with open(cache_pkl, "rb") as f:
            qs = pickle.load(f)
        return _coerce_queries(qs)

    # Build once from HF (for GitHub users)
    from datasets import load_dataset
    ds = load_dataset("nq_open", split="validation", trust_remote_code=True)
    sampled = ds.shuffle(seed=SEED).select(range(min(n_samples, len(ds))))
    queries = [_query_from_dict(item) for item in sampled]
    with open(cache_pkl, "wb") as f:
        pickle.dump(queries, f)
    return queries

def ensure_tsv(tsv_path: Path) -> Path:
    if tsv_path.exists():
        return tsv_path
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    url = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
    gz = tsv_path.with_suffix(".gz")
    if not gz.exists():
        subprocess.run(["wget", url, "-O", str(gz)], check=True)
    subprocess.run(["gunzip", "-f", str(gz)], check=True)
    return tsv_path

def load_corpus_fixed(n_passages: int = 100_000) -> List[Passage]:
    queries_pkl = DATA_DIR / f"corpus_dpr_2018_psgs_w100_{n_passages}.pkl"
    if queries_pkl.exists():
        with open(queries_pkl, "rb") as f:
            cps = pickle.load(f)
        return _coerce_passages(cps)

    cache_pkl = LOCAL_CACHE / f"corpus_dpr_2018_psgs_w100_{n_passages}.pkl"
    if cache_pkl.exists():
        with open(cache_pkl, "rb") as f:
            cps = pickle.load(f)
        return _coerce_passages(cps)

    # Build from TSV (GitHub path)
    import pandas as pd
    from tqdm import tqdm

    tsv = ensure_tsv(LOCAL_CACHE / "psgs_w100.tsv")
    df = pd.read_csv(tsv, sep="\t")
    if n_passages is not None:
        df = df.iloc[:n_passages]

    passages = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building passages"):
        passages.append(Passage(
            id=str(row["pid"]),
            text=str(row["text"]),
            title=str(row["title"]),
            source_doc_id=int(row["docid"]) if "docid" in row else 0,
        ))

    with open(cache_pkl, "wb") as f:
        pickle.dump(passages, f)
    return passages
