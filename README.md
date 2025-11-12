# The 61% Problem: Why RAG Fails Even When It Works

Systematic comparison of four retrieval architectures revealing that 61.3% 
of RAG failures occur during generation, not retrieval**—even when the correct 
answer appears in retrieved passages.

Key Finding: At ~30% recall, swapping retrieval architectures provides 
minimal gains. The bottleneck is generation: how models interpret and use 
evidence, not how they find it.



## Overview

This repository contains code and experiments from the Medium article 
[  ]

Experimental Setup:
- **Dataset:** 1,000 Natural Questions  
- **Corpus:** 100k Wikipedia passages (2018)
- **Generator:** GPT-4o-mini
- **Architectures Tested:**
  1. Dense (BGE-large)
  2. Dense (Contriever)  
  3. Hybrid (BM25 + BGE reranker)
  4. Multi-stage Contriever (embedding-based threshold filtering)

**Results:**
- All four architectures achieved 24-34% recall (Recall@20)
- Generation accuracy remained 12-15% (Exact Match)  
- Generation-stage gap: 12-19 percentage points across architectures
- BGE-large: 336 retrieved cases → 206 generation failures (61.3%)

-
## Quick Start

### Installation
```bash
git clone https://github.com/Liyuliya/RAG-experiment.git
cd RAG-experiment
pip install -r requirements.txt
```

**Requirements:** Python 3.10+ (tested on 3.10.13)

### Configuration

Create a `.env` file in the repository root:
```bash
OPENAI_API_KEY=sk-your-key-here
```

### Run Experiments
```bash
# Evaluate all four retrieval architectures
python evaluate_rag.py --dataset natural_questions --num_samples 1000

# Run specific retriever
python evaluate_rag.py --retriever bge-large

# Analyze generation failures
python analyze_failures.py --system bge-large
```

---

## Repository Structure
```
RAG-experiment/
├── data/
│   ├── natural_questions/      # NQ dataset samples
│   └── wikipedia_passages/     # 100k passage corpus
├── retrievers/
│   ├── bge_retriever.py        # BGE-large dense retrieval
│   ├── contriever_retriever.py # Contriever dense retrieval
│   ├── hybrid_retriever.py     # BM25 + BGE reranking
│   └── multistage_retriever.py # Multi-stage filtering
├── evaluation/
│   ├── compute_metrics.py      # Recall@20, Exact Match
│   └── analyze_gap.py          # Retrieval-generation gap analysis
├── evaluate_rag.py             # Main evaluation script
├── analyze_failures.py         # Failure pattern analysis
├── requirements.txt
└── README.md
```

## Key Results



### Generation Failure Breakdown (BGE-large)

- **Total questions:** 1,000
- **Successfully retrieved:** 336 (33.6%)
- **Generation success:** 130 (38.7% of retrieved)
- **Generation failures:** 206 (61.3% of retrieved)

The model fails to use retrieved evidence more than half the time.


### Basic Evaluation
```python
from retrievers import BGERetriever
from evaluation import evaluate_rag



## Computational Requirements

**Time:** ~2-4 hours for full evaluation (1,000 questions × 4 architectures)

**Costs:**
- Retrieval: Free (local models)
- Generation (GPT-4o-mini): ~$3-5 for full experiment

**Hardware:**
- CPU: Any modern processor
- RAM: 8GB minimum, 16GB recommended
- GPU: Optional (speeds up embedding computation)


## Acknowledgments

**Models Used:**
- **BGE-large** - [BAAI](https://github.com/FlagOpen/FlagEmbedding)
- **Contriever** - [Meta AI](https://github.com/facebookresearch/contriever)
- **GPT-4o-mini** - [OpenAI](https://openai.com)

**Dataset:**
- **Natural Questions** - Google Research

---

## Related Work

- 📝 [Medium Article] - Full analysis and discussion
- 📊 Part 2: Inside the 206 Failures *(coming soon)*


## License

MIT License - see [LICENSE](LICENSE) file for details.


## Contributing

Found an issue or want to improve the code? PRs welcome!

For questions about the methodology, see the [Medium article]
or open an issue.


**⭐ If you find this work useful, please consider starring the repository!**
