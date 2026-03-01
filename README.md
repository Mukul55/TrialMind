# TrialMind — Clinical Trial Protocol Optimizer

Evidence-based protocol optimization powered by 450,000+ historical trials. TrialMind is a production-grade RAG system that helps protocol designers make data-driven decisions before trials begin, preventing costly amendments and recruitment failures.

## Architecture

```
trialmind/
├── ingestion/          # Data ingestion from 4 sources
│   ├── aact_ingestion.py         # ClinicalTrials.gov AACT PostgreSQL (primary)
│   ├── pubmed_ingestion.py       # Published trial results
│   ├── fda_reviews_ingestion.py  # FDA drug approval records
│   ├── who_ictrp_ingestion.py    # WHO international registry
│   └── profile_builder.py       # Unified TrialProfile builder
├── processing/         # Data processing pipeline
│   ├── chunker.py                # 5 specialized chunk types per trial
│   ├── embedder.py               # PubMedBERT embeddings + ChromaDB
│   ├── eligibility_processor.py  # Eligibility criteria analysis
│   └── endpoint_normalizer.py    # Endpoint terminology normalization
├── retrieval/          # Multi-collection RAG retrieval
│   ├── query_router.py           # Intent classification + strategy selection
│   ├── retrieval_engine.py       # Hybrid semantic + filtered retrieval
│   ├── reranker.py               # Cross-encoder reranking
│   └── benchmark_builder.py     # Quantitative comparison tables
├── synthesis/          # LLM synthesis layer
│   ├── protocol_analyzer.py      # Main analysis orchestrator
│   ├── report_generator.py       # Structured report generation
│   └── prompts.py                # All system prompts (6 specialized)
├── api/                # FastAPI REST API
│   ├── main.py
│   ├── models.py
│   └── routes.py
├── ui/                 # Streamlit web interface
│   └── app.py
├── evaluation/         # Evaluation framework
│   ├── evaluator.py              # Automated scoring pipeline
│   └── golden_set.json          # 30+ test cases with ground truth
└── utils/              # Supporting utilities
    ├── logger.py
    ├── rate_limiter.py
    └── pdf_exporter.py
```

### Key Design Decisions

**5-Collection ChromaDB Architecture**
Each trial is chunked into 5 specialized chunk types stored in separate collections:
- `trial_profiles` — design parameters (sample size, masking, randomization)
- `trial_results` — outcomes, dropout rates, failure reasons
- `eligibility_criteria` — inclusion/exclusion criteria
- `endpoint_data` — primary/secondary endpoints
- `site_enrollment` — geography and enrollment rates

This prevents the common failure mode of mixing all trial information into one chunk.

**Domain-Specific Embeddings**
Uses `pritamdeka/S-PubMedBert-MS-MARCO` — trained on medical QA pairs. Captures clinical synonymy (e.g., "myocardial infarction" = "heart attack") that general-purpose embeddings miss.

**Two-Stage Retrieval**
1. Semantic search across relevant collections (with metadata pre-filtering)
2. Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)

**Intent-Aware Query Routing**
Classifies queries into 7 intent types and routes to specialized collection subsets with tailored synthesis prompts.

## Quick Start

### 1. Prerequisites

- Python 3.11+
- 8GB RAM minimum (16GB recommended for embedding generation)
- GPU optional (speeds up embedding ~10x)
- Free AACT account: https://aact.ctti-clinicaltrials.org/users/sign_up
- Free NCBI API key: https://ncbi.nlm.nih.gov/account (improves PubMed rate limits)
- Anthropic API key for LLM synthesis

### 2. Installation

```bash
git clone <repo>
cd TrialMind
pip install -r requirements.txt
python -m spacy download en_core_sci_lg
cp .env.example .env
# Edit .env with your credentials
```

### 3. Configure Credentials

Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
AACT_DB_USER=your_aact_username
AACT_DB_PASS=your_aact_password
NCBI_API_KEY=your_ncbi_key      # Optional but recommended
FDA_API_KEY=your_fda_key        # Optional but recommended
```

### 4. Run Data Ingestion (one-time, 3-8 hours)

```bash
# Full ingestion
python ingest_all.py

# Faster subset (2015+ trials only)
python ingest_all.py --start-year 2015

# Skip slow sources for testing
python ingest_all.py --skip-aact --max-pubmed 100
```

### 5. Start the API

```bash
uvicorn trialmind.api.main:app --reload --port 8000
```

### 6. Start the UI

```bash
streamlit run trialmind/ui/app.py
```

### 7. Run Evaluation

```bash
python trialmind/evaluation/evaluator.py
# Target: average score > 0.65
```

## Example Queries

```
"What sample sizes are used in Phase 2 NSCLC trials with PD-L1 >= 1% biomarker requirement?"

"What primary endpoints has FDA accepted for Phase 3 heart failure trials in the last 5 years?"

"Which countries enroll fastest for Phase 3 cardiovascular trials?"

"What eligibility criteria most commonly cause recruitment failure in CNS trials?"

"Why are Phase 3 oncology trials terminated early and what warning signs predict this?"

"I have a Phase 2 trial in NSCLC, N=80, single-arm, primary endpoint ORR, excluding
 ECOG PS 2. Is this design consistent with historical practice?"
```

## API Reference

### POST /query

```json
{
  "query": "What sample sizes are typical for Phase 3 NSCLC trials?",
  "protocol": {
    "indication": "Non-small cell lung cancer",
    "phase": "Phase 3",
    "drug_name": "Pembrolizumab",
    "planned_enrollment": 500,
    "primary_endpoint": "Overall Survival"
  }
}
```

### POST /protocol-review

Accepts a full protocol dict, returns comprehensive optimization report.

### GET /benchmark/{indication}?phase=Phase 3

Returns key benchmark metrics for an indication.

### GET /health

Returns system status and collection document counts.

## Data Sources

| Source | Type | Trials | Access |
|--------|------|--------|--------|
| AACT (ClinicalTrials.gov) | PostgreSQL | 450,000+ | Free account |
| PubMed | REST API | 50,000+ abstracts | Free (key optional) |
| openFDA Drugs@FDA | REST API | 5,000+ approvals | Free (key optional) |
| WHO ICTRP | Web | Supplemental | Free |

## Evaluation

The system is evaluated against a golden set of 30 clinical trial design questions with known ground-truth answers. Target score: **>= 0.65** (keyword recall + NCT citation + quantitative content scoring).

Run evaluation after ingestion:
```bash
python trialmind/evaluation/evaluator.py
```

## Implementation Notes

1. **AACT connection is critical** — without it the system has limited value. Ensure credentials are configured before running.

2. **Never hallucinate NCT IDs** — the system validates that all cited NCT IDs appear in retrieved context.

3. **Metadata filtering before semantic search** — phase, indication, and year filters applied at ChromaDB query level for precision.

4. **Quantitative synthesis** — all prompts enforce numeric, evidence-cited output.

5. **Pipeline is resumable** — re-running `ingest_all.py` skips already-indexed documents.

## License

MIT
