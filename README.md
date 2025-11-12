# The 61% Problem: Why RAG Fails Even When It Works

Systematic comparison of four retrieval architectures revealing that **61.3% 
of RAG failures occur during generation, not retrieval**—even when the correct 
answer appears in retrieved passages.

📊 At ~30% recall, swapping retrieval architectures provides 
minimal gains. The bottleneck is generation: how models interpret and use 
evidence, not how they find it.

## Overview
This repository contains code and experiments from the Medium article 
[The 61% Problem: Why RAG Fails Even When It Works]

**Experimental Setup:**
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
# Run all four retrieval architectures
python src/main.py

# Run specific retriever
python src/main.py --retriever bge-large

# Run with custom sample size
python src/main.py --num_samples 100

# Analyze generation failures for a specific system
python src/experiment_runner.py --analyze --system bge-large
```

## Repository Structure
```
RAG-experiment/
├── src/
│   ├── __init__.py             
│   ├── config.py               
│   ├── data_loader.py          
│   ├── retriever_systems.py     
│   ├── llm_interface.py         
│   ├── evaluator.py          
│   ├── experiment_runner.py   
│   ├── helpers.py            
│   └── main.py                
├── data/                     
├── results/                   
├── .env                    
├── .gitignore
├── requirements.txt
└── README.md
```

**Note:** The `data/` folder should contain Natural Questions and Wikipedia passages. 
See `src/data_loader.py` for expected format.



## Computational Requirements

**Full experiment:** ~2-4 hours, ~$3-5 in API costs (GPT-4o-mini)  
**Hardware:** 8GB+ RAM, GPU optional


## Related Work
- 📝 [Medium Article](link-here) - Full analysis and discussion
- 📊 Part 2: Inside the 206 Failures *(coming soon)*



## License
MIT License - see [LICENSE](LICENSE) file for details.



## Contributing
Found an issue or want to improve the code? Pull requests welcome!

For questions about the methodology, see the [Medium article](link-here) 
or open an issue.

**⭐ If you find this work useful, please consider starring the repository!**
