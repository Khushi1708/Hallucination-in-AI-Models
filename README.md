This project implements Semantic Entropy (Farquhar et al., Nature, 2024) end to end and adds a verification layer on top: a retrieval-based check against live Wikipedia and a deterministic rule engine for known facts. It ships with a Streamlit dashboard, a command-line tool, a Python API, and a test suite.

How it works
Step 1  User input             "Who invented the telephone?"
   │
Step 2  Sample N responses     N = 3–12 answers at temperature τ (default 0.8)
   │
Step 3  Embed responses        Sentence-Transformers (all-MiniLM-L6-v2),
   │                           with a TF-IDF fallback if the model is unavailable
   │
Step 4  Cluster by meaning     Cosine similarity + agglomerative clustering
   │                           → "Bell invented it" (75%), "Meucci invented it" (25%)
   │
Step 5  Semantic entropy       H = -Σ P(Cₖ) ln P(Cₖ)   →  normalized, plus a confidence score
   │
Step 6  Verification layer     Formal rule engine first, then Wikipedia RAG cross-check
   │
Step 7  Final audit            Answer · confidence · entropy level · verified / hallucination flag

If the sampled answers all mean the same thing, entropy is low and the model is probably confident. If they split into several different meanings, entropy is high and the answer is suspect. The verification layer then checks the dominant answer against external evidence, so a confidently wrong answer can still be flagged.

Features
Semantic entropy pipeline – sampling, embedding, clustering, entropy, confidence score, and a Low / Moderate / High uncertainty rating.
Multiple generation backends – OpenAI (ChatGPT), Google Gemini, Kimi (Moonshot), local Hugging Face models, and an offline simulation mode that needs no API keys.
Two-stage verification
Formal Rule Engine: deterministic checks against a built-in knowledge base (capitals, famous inventions and discoveries, a few space facts).
RAG verification: retrieves a Wikipedia summary and compares it to the answer by embedding similarity.
Interactive Streamlit dashboard – 2D cluster map (PCA), cosine similarity heatmap, entropy math breakdown, verification cards, and a full 7-step trace.
CLI with human-readable and JSON output.
Unit tests for the embedder, clustering, entropy, rule verifier, and the full pipeline.
Installation

Prerequisites: Python 3.9+ and pip.

bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

The first run downloads the all-MiniLM-L6-v2 embedding model. If the download isn't possible, the embedder automatically falls back to TF-IDF vectors.

API key configuration

API keys are only needed for the real LLM backends. Simulation mode works without them.

Copy the example file and fill in the keys you have:

bash
cp .env.example .env
env
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
KIMI_API_KEY=your-kimi-key
KIMI_BASE_URL=https://api.moonshot.cn/v1

⚠️ Never commit your .env file. It is listed in .gitignore. If a key has ever been pushed or shared, revoke it and create a new one.

Supported backends:

Mode	Provider	Default / example models
chatgpt / openai	OpenAI	gpt-4o-mini, gpt-3.5-turbo, gpt-4o
gemini	Google	gemini-1.5-flash, gemini-pro
kimi	Moonshot AI	moonshot-v1-8k, moonshot-v1-32k
huggingface	Local model	HuggingFaceTB/SmolLM-135M-Instruct
simulation	Built-in	Deterministic offline benchmark scenarios
Usage
1. Streamlit dashboard
bash
streamlit run app.py

Choose a backend, the number of samples, the similarity threshold, and the temperature in the sidebar, then enter a question or pick a preset benchmark.

2. Command line
bash
# Offline simulation (no API keys needed)
python main.py --question "Who invented the telephone?" --mode simulation

# Real backends
python main.py --question "What is the capital of Germany?" --mode gemini
python main.py --question "Who discovered penicillin?" --mode kimi --samples 10

# Raw JSON output
python main.py -q "Who invented the telephone?" --json
Option	Default	Description
--question, -q	Who invented the telephone?	Question to analyze
--samples, -n	8	Number of sampled responses
--temperature, -t	0.8	Sampling temperature
--mode, -m	simulation	simulation, openai, chatgpt, gemini, kimi, huggingface
--model	backend default	Specific model name
--api-key	from .env	Override the API key
--threshold	0.75	Cosine similarity threshold for merging clusters
--json	off	Print results as JSON
3. Python API
python
from core.pipeline import SemanticEntropyPipeline

pipeline = SemanticEntropyPipeline(generator_mode="simulation")
result = pipeline.run("Who invented the telephone?", n_samples=8)

print("Dominant answer:", result["dominant_answer"])
print("Confidence:", result["entropy"]["confidence_score"])
print("Status:", result["overall_status"]["status"])
print("Verified ground truth:", result["verification"]["ground_truth_answer"])

You can also pass your own answers with custom_samples=[...] to analyze responses you collected elsewhere.

Running the tests
bash
pytest tests/ -v
The math

Pairwise similarity between embeddings:

$$\text{Sim}(\mathbf{v}_i, \mathbf{v}_j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\|\,\|\mathbf{v}_j\|}$$

Cluster probabilities over $K$ semantic clusters and $N$ samples:

$$P(C_k) = \frac{|C_k|}{N}$$

Semantic entropy:

$$SE = -\sum_{k=1}^{K} P(C_k)\,\ln P(C_k)$$

Normalized entropy and confidence:

$$SE_{norm} = \frac{SE}{\ln N} \in [0,1] \qquad \text{Confidence} = (1 - SE_{norm}) \times 100\%$$

Default risk thresholds on $SE_{norm}$: Low below 0.35, Moderate from 0.35 to 0.65, High above 0.65.

Project structure
.
├── app.py                 # Streamlit dashboard
├── main.py                # Command-line interface
├── requirements.txt
├── .env.example           # Template for API keys
├── core/
│   ├── generator.py       # Multi-backend response sampling + simulation mode
│   ├── embedder.py        # Sentence-Transformers embeddings with TF-IDF fallback
│   ├── clustering.py      # Similarity matrix, agglomerative clustering, PCA coords
│   ├── entropy.py         # Semantic entropy, confidence, risk level
│   ├── verifier.py        # Formal rule engine + Wikipedia RAG verification
│   └── pipeline.py        # 7-step orchestration
└── tests/
    └── test_pipeline.py
Limitations
The formal rule engine covers only a small hard-coded set of facts, so most questions are verified through Wikipedia retrieval only.
RAG verification compares the answer to the top Wikipedia summary by embedding similarity. This is a coarse signal and can misjudge questions where the top article isn't the best source.
Offline simulation mode uses scripted answers for a few benchmark questions; it demonstrates the pipeline but isn't a real hallucination test.
SemanticEntropyPipeline.run() also queries ChatGPT, Gemini, and Kimi for the side-by-side model comparison, so real API calls (and costs) can occur even when the main mode is simulation.
Semantic entropy works best for short factual questions with a single correct answer.
Reference

Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. Nature, 630, 625–630.
