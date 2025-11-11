from pathlib import Path
import pickle

print("-"*70)
print("FULL BENCHMARK: 1000 QUERIES")
print("-"*70)

# 
print("\n🔍 Checking available retrieval systems...")

classes_to_find = {
    'SystemA': ['SystemA_Contriever', 'SystemA_DenseContriever', 'DenseContriever'],
    'SystemB': ['SystemB_HybridBM25BGE', 'SystemB_HybridBM25Rerank', 'HybridBM25Rerank'],
    'SystemC': ['SystemC_BGE', 'SystemC_DenseBGELarge', 'DenseBGELarge'],
    'SystemD': ['SystemD_ReflectiveContriever', 'ReflectiveFilterContriever']
}

found_classes = {}
for key, names in classes_to_find.items():
    for name in names:
        if name in globals():
            found_classes[key] = globals()[name]
            print(f"  ✓ {key}: {name}")
            break
    if key not in found_classes:
        print(f"  ✗ {key}: NOT FOUND - Check your retriever_systems cell!")

if len(found_classes) < 4:
    print("\n ERROR: Some retrieval systems not defined!")
    print("Make sure you've run the retriever_systems.py cell")
    raise RuntimeError("Missing retrieval system classes")


DATA_DIR = Path("/dpr-wikipedia-2018-100k")

with open(DATA_DIR / "queries_nq_open_validation_1000.pkl", "rb") as f:
    queries_raw = pickle.load(f)
with open(DATA_DIR / "corpus_dpr_2018_psgs_w100_100000.pkl", "rb") as f:
    corpus_raw = pickle.load(f)

# Coerce to proper types
def _coerce_queries(qs):
    out = []
    for d in qs:
        if isinstance(d, Query):
            out.append(d)
            continue
        ans = d.get("answers", d.get("answer", []))
        if isinstance(ans, dict) and "text" in ans:
            ans = ans["text"]
        if isinstance(ans, str):
            ans = [ans]
        out.append(Query(
            id=str(d.get("id", "")),
            question=str(d.get("question", d.get("title", ""))),
            answers=tuple(ans)
        ))
    return out

def _coerce_corpus(cps):
    out = []
    for d in cps:
        if isinstance(d, Passage):
            out.append(d)
            continue
        out.append(Passage(
            id=str(d.get("pid", d.get("id", ""))),
            text=str(d.get("text", "")),
            title=str(d.get("title", "")),
            source_doc_id=int(d.get("docid", d.get("source_doc_id", 0)))
        ))
    return out

queries = _coerce_queries(queries_raw)
corpus = _coerce_corpus(corpus_raw[:100_000])

print(f"\n✓ Loaded {len(queries)} queries, {len(corpus)} passages")


TEST_N = 1000 
test_queries = queries[:TEST_N]

print(f"\nRunning: {len(test_queries)} queries")
print(f"Corpus coverage: ~40% (from verification)")
print(f"Expected Recall@20: 10-20%")
print(f"Expected time: ~2 hours")
print(f"Expected cost: ~$0.50")
print("="*70)


systems = {
    "Dense (Contriever)": found_classes['SystemA'](),
    "Hybrid (BM25 + BGE)": found_classes['SystemB'](),
    "Dense (BGE-large)": found_classes['SystemC'](),
    "Reflective Filter (Contriever)": found_classes['SystemD'](
        top_k_candidates=60,
        max_passages_to_llm=12,
        thr_relevance=0.50,
        thr_support=0.50,
        alpha_blend=0.30,
    ),
}

#share emb A -> D
print("\nSharing System A embeddings with System D...")
sysA = systems["Dense (Contriever)"]
sysD = systems["Reflective Filter (Contriever)"]
sysA.index(corpus)
sysD.corpus, sysD.embeddings = sysA.corpus, sysA.embeddings

# Run all systems
runner = ExperimentRunner(
    output_dir=Path("./results"),
    checkpoint_every=100,
    top_k_to_llm=12
)

summaries = {}
for name, system in systems.items():
    print(f"\n{'='*70}")
    print(name.upper())
    print('='*70)
    results = runner.run_system(system, test_queries, corpus)
    summaries[name] = runner.aggregate_results(results)


def g(sysn, key): 
    return summaries[sysn][key]

print("\n" + "-"*70)
print(f"FINAL RESULTS ({len(test_queries)} queries)")
print("-"*70)
print(f"{'Metric':<15} {'Dense(Contr.)':>16} {'Hybrid':>12} {'BGE-large':>12} {'Reflective':>12}")
print("-"*70)

print(f"{'Exact Match':<15} {g('Dense (Contriever)','exact_match'):>15.1f}% "
      f"{g('Hybrid (BM25 + BGE)','exact_match'):>11.1f}% "
      f"{g('Dense (BGE-large)','exact_match'):>11.1f}% "
      f"{g('Reflective Filter (Contriever)','exact_match'):>11.1f}%")

print(f"{'F1 Score':<15} {g('Dense (Contriever)','f1_score'):>15.1f}% "
      f"{g('Hybrid (BM25 + BGE)','f1_score'):>11.1f}% "
      f"{g('Dense (BGE-large)','f1_score'):>11.1f}% "
      f"{g('Reflective Filter (Contriever)','f1_score'):>11.1f}%")

print(f"{'Recall@5':<15} {g('Dense (Contriever)','recall@5'):>15.1f}% "
      f"{g('Hybrid (BM25 + BGE)','recall@5'):>11.1f}% "
      f"{g('Dense (BGE-large)','recall@5'):>11.1f}% "
      f"{g('Reflective Filter (Contriever)','recall@5'):>11.1f}%")

print(f"{'Recall@20':<15} {g('Dense (Contriever)','recall@20'):>15.1f}% "
      f"{g('Hybrid (BM25 + BGE)','recall@20'):>11.1f}% "
      f"{g('Dense (BGE-large)','recall@20'):>11.1f}% "
      f"{g('Reflective Filter (Contriever)','recall@20'):>11.1f}%")

print(f"{'MRR@20':<15} {g('Dense (Contriever)','mrr@20'):>15.3f} "
      f"{g('Hybrid (BM25 + BGE)','mrr@20'):>11.3f} "
      f"{g('Dense (BGE-large)','mrr@20'):>11.3f} "
      f"{g('Reflective Filter (Contriever)','mrr@20'):>11.3f}")

print(f"{'Cost($)':<15} {g('Dense (Contriever)','total_cost_usd'):>15.4f} "
      f"{g('Hybrid (BM25 + BGE)','total_cost_usd'):>11.4f} "
      f"{g('Dense (BGE-large)','total_cost_usd'):>11.4f} "
      f"{g('Reflective Filter (Contriever)','total_cost_usd'):>11.4f}")

print("\n BENCHMARK COMPLETE! ")
print(f"Total cost: ${sum(s['total_cost_usd'] for s in summaries.values()):.4f}")
