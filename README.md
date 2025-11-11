# RAG-experiment
# The 61% Problem: Why RAG Fails Even When It Works

Code and experiments for the Medium article  
[*The 61% Problem: Why RAG Fails Even When It Works*]

-🧠 Overview
This project compares four retrieval architectures on 1,000 **Natural Questions** using a 100k-passage 2018 **Wikipedia** corpus, with **GPT-4o-mini** as the generator.  
It quantifies generation failures even when relevant evidence is retrieved — **61.3% of retrieved cases still fail**.

Retrievers evaluated:
- BGE-large (dense)
- Contriever (dense)
- Hybrid (BM25 + BGE reranker)
- Reflective-Contriever (Self-RAG–inspired heuristic filter)



## Setup

git clone https://github.com/Liyuliya/RAG-experiment.git
cd RAG-experiment

Python 3.10+ recommended (tested on 3.10.13)

pip install -r requirements.txt

Main libraries:
torch>=2.2.0
transformers>=4.44.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
rank-bm25>=0.2.2
openai>=1.40.0
python-dotenv>=1.0.0
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.4.0
tqdm>=4.66.0
matplotlib>=3.8.0
tabulate>=0.9.0

Create a .env file in the repository root:
OPENAI_API_KEY=sk-your-key
The notebook and scripts automatically load it with:
os.getenv("OPENAI_API_KEY")





BGE & Contriever retrievers — BAAI, Meta AI

GPT-4o-mini — OpenAI
