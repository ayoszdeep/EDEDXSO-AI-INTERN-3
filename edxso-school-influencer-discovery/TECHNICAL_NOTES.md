# Technical Note: EDXSO School Influencer Discovery Tool

**Author:** EDXSO AI Engineering Intern Candidate  
**Project:** LinkedIn School Influencer Discovery Tool  
**Date:** November 2026  
**Document Version:** 1.0  

---

## 1. System Architecture

The application is structured into decoupled, modular components designed for high maintainability, unit testability, and zero cross-component coupling:

```
edxso-school-influencer-discovery/
├── app.py                  # Main Streamlit reactive UI & user controls
├── data_loader.py          # CSV loading, validation, sanitization & feedback logging
├── profile_analyzer.py     # NLP/Rule extraction of role families, past roles, and tags
├── embedding_engine.py     # Dense vector embedding generation & cosine similarity
├── ranker.py               # Hybrid 5-component weighted ranker with renormalization
├── explanations.py         # Deterministic Explainable AI (XAI) justification generator
├── data/
│   ├── profiles.csv        # 50 source-verified real professional records
│   └── feedback.csv        # Evaluator relevance feedback loop
└── tests/
    ├── conftest.py         # Pytest configuration & path registration
    ├── test_data.py        # Dataset schema and integrity tests
    ├── test_analyzer.py    # Role and indicator extraction tests
    ├── test_ranker.py      # Ranking formula and renormalization tests
    └── test_integration.py # End-to-end integration test with Precision@20
```

### Component Interaction Workflow:
1. **Input Interface (`app.py`):** Captures reference profile details either manually or via pre-fills (e.g. Dr. Amit Sharma).
2. **Analysis Layer (`profile_analyzer.py`):** Maps unstructured biography and role strings into canonical taxonomies (`role_family`, `school_ecosystem_tags`, `leadership_evidence`, `multi_school_evidence`, `decision_maker_evidence`).
3. **Data Access (`data_loader.py`):** Loads, validates, and deduplicates the candidate dataset (`data/profiles.csv`), guaranteeing no invalid rows or missing source links.
4. **Vector Embedding Layer (`embedding_engine.py`):** Converts unified professional representations into 384-dimensional dense vectors and computes pairwise cosine similarity against all candidates using an in-memory cache.
5. **Hybrid Ranker (`ranker.py`):** Fuses semantic similarity with structural school indicators, role compatibility, network footprint, and geographic proximity. Omits the reference person if present in the candidate collection.
6. **Explanation Generator (`explanations.py`):** Compiles evidence-backed reasoning bullets and transparent point contribution breakdowns for every ranked candidate.

---

## 2. Data Retrieval, Provenance & Responsible Access

### Ethical & Legal Compliance Protocols
The tool strictly adheres to LinkedIn's User Agreement and the **Digital Personal Data Protection Act (DPDP Act, 2023 / DPDP Rules, 2025)**:
- **No Scraping or Bot Extensions:** The application **does not** scrape LinkedIn using Selenium, Playwright, Puppeteer, or browser plugins.
- **Lawfully Accessible Public Information:** Profiles were curated from publicly accessible institutional sources:
  - Official school governance pages (e.g., The Doon School, Modern School, NPS Group, Heritage Schools).
  - National education conference speaker directories (e.g., FICCI ARISE, ScooNews Global Education Summit, EducationWorld).
  - Published professional biographies and education trust portals.
- **Data Minimisation & Privacy:** Only strictly necessary professional information is retained. No private phone numbers, home addresses, or personal contact details are collected or stored.
- **Provenance Traceability:** Every candidate record contains an immutable `source_url`, `source_date`, and `data_status`.

---

## 3. AI / ML Approach: Semantic Embeddings & Vector Similarity

Naive keyword search fails to capture semantic equivalence (e.g., understanding that a *"Founding Director of Step by Step School advising school promoters"* is conceptually equivalent to a *"Former Principal advising school owners"*).

### Model Selection: `sentence-transformers/all-MiniLM-L6-v2`
- **Architecture:** 6-layer MiniLM transformer with 384-dimensional output embeddings.
- **Normalization:** Embeddings are L2-normalized upon generation:
  $$\|v\|_2 = 1$$
- **Cosine Similarity:** For normalized vectors, cosine similarity reduces to a fast dot product:
  $$\text{Cosine Similarity}(A, B) = A \cdot B$$
- **In-Memory Caching:** Hashes of candidate text strings are cached in-memory (`_CACHE`), ensuring that re-ranking candidate pools requires zero redundant transformer forward passes.
- **Fault-Tolerant Fallback:** In the event of network connectivity interruptions during model initialization, the engine seamlessly activates a scikit-learn TF-IDF subword n-gram fallback, guaranteeing uninterrupted uptime.

---

## 4. Hybrid Ranking Methodology & Dynamic Renormalization

To balance semantic nuance with concrete domain credentials, the system implements a **5-factor weighted hybrid scoring formula**:

| Component | Weight ($w_k$) | Measurement Metric |
| :--- | :---: | :--- |
| **Semantic Similarity** | **45%** | Cosine similarity of 384-d dense embeddings |
| **School Ecosystem Tags** | **20%** | Jaccard similarity: $J(A, B) = \frac{\|A \cap B\|}{\|A \cup B\|}$ |
| **Role & Experience Proximity** | **15%** | Role-family transition matrix + experience delta bonus |
| **Multi-School Exposure** | **15%** | Documented multi-campus reach ($1.0$ for $10+$ schools; $0.5$ for single campus) |
| **Location Alignment** | **5%** | Metro education hub and regional proximity |

### Dynamic Missing-Data Renormalization
A critical design requirement is that missing or unverified candidate information must **never** be penalized as a false negative. If an attribute (such as multi-school experience or precise city location) is unverified in the public record, it is assigned `None`.

The final score $S \in [0, 100]$ is computed dynamically over available components:
$$S = \frac{\sum_{k \in \text{available}} w_k \cdot S_k}{\sum_{k \in \text{available}} w_k} \times 100$$

This guarantees that candidates with incomplete public profiles are fairly evaluated on their verified credentials without score skew.

---

## 5. Limitations & Production Considerations

1. **Curated vs. Universal Member Search:** The current MVP operates over a curated dataset of 50 verified professionals. Direct discovery across all 1+ billion LinkedIn members requires approved access to the official **LinkedIn Member Search API** or licensed enterprise talent aggregators (e.g., People Data Labs).
2. **Vector Database Scalability:** For scaling beyond 10,000 candidates, in-memory array operations should be replaced with **PostgreSQL + pgvector** or a dedicated vector database (Milvus / Pinecone), leveraging HNSW indexing for sub-millisecond retrieval.
3. **Active Learning Ranking Tuning:** The current weights (45/20/15/15/5) were empirically calibrated. In production, logged evaluator feedback (`data/feedback.csv`) can be used to train a learning-to-rank algorithm (e.g., LambdaMART) for continuous relevance optimization.

---

## 6. Setup & Execution Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Git

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/your-username/edxso-school-influencer-discovery.git
cd edxso-school-influencer-discovery

# 2. Setup virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate # On Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify test suite
pytest tests/ -v

# 5. Launch dashboard
streamlit run app.py
```
The dashboard opens at `http://localhost:8501`.
