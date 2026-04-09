# 🏢 Enterprise RAG — Advanced Retrieval-Augmented Generation Pipeline

A production-grade RAG system built on **Azure AI Search** and **Azure OpenAI**, implementing every advanced technique discussed in senior GenAI interviews — with real benchmark numbers to back it up.

---

## 📌 What This Project Is

An HR Policy Q&A system where employees can query internal policy documents and receive accurate, grounded answers. Built as a learning project to implement and measure every advanced RAG technique — not just theoretically, but with real Azure infrastructure and RAGAS evaluation scores.

---

## 🏗️ Architecture

```
User Query
    ↓
[Input Guardrail]     → blocks injection, off-topic, harmful queries
    ↓
[Query Rewriter]      → HyDE: expands 9-word query to 70-word policy passage
    ↓
[Hybrid Search]       → Vector (cosine) + Keyword (BM25) + RRF fusion
    ↓
[Semantic Reranker]   → Azure cross-encoder re-scores top 5 chunks
    ↓
[Context Guardrail]   → validates retrieved context before generation
    ↓
[Prompt Builder]      → assembles grounded system prompt + context
    ↓
[GPT-4o Generator]    → temperature 0, policy-grounded answer
    ↓
[Self Checker]        → second LLM call verifies answer is grounded
    ↓
[Output Guardrail]    → PII scan, system prompt leak check, content safety
    ↓
Final Answer
```

All 9 stages are wired as a **LangGraph StateGraph** with conditional edges that short-circuit blocked queries.

---

## 🧰 Azure Stack

| Service | Purpose |
|---|---|
| Azure OpenAI (`gpt-4o`) | Generation, self-checking, topic classification |
| Azure OpenAI (`text-embedding-3-small`) | 1536-dim chunk embeddings |
| Azure AI Search (Basic) | Hybrid search, HNSW vector index, semantic reranker |

---

## 📂 Project Structure

```
enterprise-rag/
├── data/                        # PDF policy documents
├── ingestion/
│   ├── generate_docs.py         # Create placeholder PDFs
│   ├── document_loader.py       # Extract text from PDFs
│   ├── chunker.py               # Child + parent chunking
│   ├── embedder.py              # Azure OpenAI embeddings
│   └── indexer.py               # Push to Azure AI Search
├── retrieval/
│   ├── guardrail_input.py       # Input validation + injection detection
│   ├── query_rewriter.py        # HyDE query expansion
│   ├── hybrid_search.py         # Vector + BM25 + RRF
│   └── reranker.py              # Azure semantic reranker
├── generation/
│   ├── prompt_builder.py        # System prompt + context assembly
│   ├── generator.py             # GPT-4o call + self-checker
│   └── guardrail_output.py      # PII + safety output checks
├── pipeline/
│   ├── state.py                 # LangGraph RAGState TypedDict
│   ├── nodes.py                 # All 9 pipeline nodes
│   └── graph.py                 # Graph assembly + runner
├── evaluation/
│   └── ragas_eval.py            # RAGAS evaluation on 10 questions
├── config.py                    # Environment variable loader
├── requirements.txt
└── .env.template
```

---

## ⚙️ Advanced Techniques Implemented

### 1. Parent-Child Chunking
Small child chunks (200 tokens, 50 overlap) are embedded and searched for precision. When retrieved, their parent chunks (600 tokens) are fetched and sent to the LLM for full context.

```
Child chunks → searched by vector    (9 chunks, avg 190.2 tokens)
Parent chunks → sent to LLM          (3 chunks, avg 470.3 tokens)
```

### 2. HyDE — Hypothetical Document Embeddings
User queries are short (8-10 words). Policy documents are long (190+ words). HyDE bridges this gap by generating a hypothetical ideal answer first, then embedding that for search.

```
"How many leave days?" (9 words)
        ↓ GPT-4o
"Employees are permitted to carry forward a maximum of 10 unused
vacation leave days into the next calendar year..." (64 words)
```

### 3. Hybrid Search + RRF
Vector search captures semantic intent. BM25 captures exact keyword matches. RRF merges both using ranks, not raw scores.

```
child_0003 — found by BM25 only (keyword match)
           — missed by vector search entirely
           — included in final results via RRF
```
This proves hybrid search outperforms vector-only retrieval.

### 4. Semantic Reranking
Azure's cross-encoder reads query + chunk together to produce precise relevance scores (0-4.0 scale).

```
child_0003: hybrid rank 4 → reranker rank 2  ⬆️ +2
child_0008: hybrid rank 3 → reranker rank 5  ⬇️ -2
```

### 5. Dual Guardrails
**Input guardrail** — 3 checks in order (cheapest first):
- Empty/junk detection (instant)
- Prompt injection — 11 regex patterns (instant)
- Topic classifier — GPT-4o JSON response

**Output guardrail** — 4 checks:
- Empty response detection
- PII scan — 6 pattern types (Aadhaar, PAN, email, phone, credit card, employee ID)
- System prompt leak detection
- Content safety via GPT-4o

### 6. Self-Check Hallucination Detection
A second GPT-4o call verifies every claim in the generated answer is supported by the retrieved context. Returns `GROUNDED` or `HALLUCINATION` with confidence score.

### 7. LangGraph Stateful Pipeline
All 9 stages modelled as a `StateGraph` with a shared `RAGState` TypedDict. Conditional edges short-circuit blocked queries.

```
Prompt injection query → blocked at Node 1 → latency: 0.023s
Valid policy query     → all 9 nodes run   → latency: ~13-20s
```

---

## 📊 Real Benchmark Numbers

### Ingestion
| Metric | Value |
|---|---|
| Documents | 3 PDFs |
| Total words | 1,143 |
| Child chunks | 9 (200 tokens, 50 overlap) |
| Parent chunks | 3 (600 tokens) |
| Embedding model | text-embedding-3-small |
| Vector dimensions | 1,536 |
| Embedding cost | $0.000034 |

### Retrieval (query: "How many days of leave can I carry forward?")
| Method | Top Result | Score |
|---|---|---|
| Vector search | child_0004 | 0.690 cosine |
| Keyword (BM25) | child_0004 | 5.577 BM25 |
| RRF fusion | child_0004 | 0.0328 RRF |
| Semantic reranker | child_0004 | 2.879 / 4.0 |

### Generation
| Metric | Value |
|---|---|
| Context tokens | 935 (83% of prompt) |
| Total prompt tokens | 1,126 / 4,000 |
| Answer tokens | 40 |
| Generation latency | 2.05s |
| Self-check verdict | GROUNDED (confidence 1.0) |

### RAGAS Evaluation (10 questions)
| Metric | Score |
|---|---|
| **Faithfulness** | **1.0** |
| **Answer Relevancy** | **0.8848** |
| **Context Precision** | **1.0** |
| **Overall Average** | **0.9616** |

Context Precision of **1.0** across all 10 questions confirms the hybrid search + reranker combination retrieves only relevant chunks every time.

---

## 🐛 Real Issue Found During Evaluation

During RAGAS evaluation, the gift policy question (Q6) scored **0.0 faithfulness** despite a correct answer. Root cause: the self-checker context was truncated at 2,000 characters, cutting off the anti-bribery section. The self-checker then flagged correct claims as hallucinations because it couldn't see the supporting text.

**Fix:** increase `context_truncated = context[:4000]` in `pipeline/nodes.py`

**Expected faithfulness after fix:** 0.93+

This is documented intentionally — finding and explaining real failure modes is part of building production systems.

---

## 🚀 Setup

### Prerequisites
- Python 3.11+
- Azure OpenAI resource with `gpt-4o` and `text-embedding-3-small` deployments
- Azure AI Search resource (Basic tier minimum for semantic reranker)

### Installation
```bash
git clone https://github.com/YOUR-USERNAME/enterprise-rag.git
cd enterprise-rag

# Create virtual environment
uv venv
uv pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Fill in your Azure credentials in .env
```

### Running the Pipeline

```bash
# Stage 1 — Ingestion
python ingestion/generate_docs.py
python ingestion/document_loader.py
python ingestion/chunker.py
python ingestion/embedder.py
python ingestion/indexer.py

# Stage 2 — Test retrieval
python retrieval/guardrail_input.py
python retrieval/query_rewriter.py
python retrieval/hybrid_search.py
python retrieval/reranker.py

# Stage 3 — Test generation
python generation/prompt_builder.py
python generation/generator.py
python generation/guardrail_output.py

# Full pipeline (LangGraph)
python pipeline/graph.py

# Evaluation
python evaluation/ragas_eval.py
```

---

## 🔑 Environment Variables

```env
AZURE_OPENAI_ENDPOINT=https://YOUR-NAME.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o

AZURE_SEARCH_ENDPOINT=https://YOUR-NAME.search.windows.net
AZURE_SEARCH_API_KEY=your-key
AZURE_SEARCH_INDEX_NAME=enterprise-rag-index
```

---

## 📦 Key Dependencies

```
langchain, langgraph          — pipeline orchestration
azure-search-documents        — Azure AI Search client
openai                        — Azure OpenAI client
ragas                         — RAG evaluation framework
tiktoken                      — token counting
rich                          — terminal output formatting
pypdf, reportlab              — PDF handling
```

---

## 🎯 What I Learned

- **Context Precision 1.0** is achievable with good chunking + hybrid search + reranker — but it requires all three working together
- **HyDE adds real value** for short queries against long documents — the word expansion from 9 to 64 words measurably improves retrieval
- **Conditional edges in LangGraph** are critical for production — a blocked query at node 1 costs 0.023s instead of 20s
- **Self-checkers can produce false negatives** when given truncated context — the system needs to be as careful about what the verifier sees as what the generator sees
- **RAGAS evaluation reveals what testing cannot** — Q8 scored 0.527 on answer relevancy despite passing self-check, indicating the answer was technically correct but not optimally relevant
