import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys (all free) ──────────────────────────────────────────────────────
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")          # Free at ncbi.nlm.nih.gov/account
FDA_API_KEY  = os.getenv("FDA_API_KEY", "")            # Free at open.fda.gov/apis/authentication
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── AACT PostgreSQL (free account at aact.ctti-clinicaltrials.org) ───────────
AACT_DB_HOST = os.getenv("AACT_DB_HOST", "aact-db.ctti-clinicaltrials.org")
AACT_DB_PORT = int(os.getenv("AACT_DB_PORT", "5432"))
AACT_DB_NAME = os.getenv("AACT_DB_NAME", "aact")
AACT_DB_USER = os.getenv("AACT_DB_USER", "")          # Register free at aact.ctti-clinicaltrials.org
AACT_DB_PASS = os.getenv("AACT_DB_PASS", "")

# ── Vector Store ─────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./trialmind_db"
COLLECTION_TRIAL_PROFILES    = "trial_profiles"
COLLECTION_TRIAL_RESULTS     = "trial_results"
COLLECTION_ELIGIBILITY       = "eligibility_criteria"
COLLECTION_ENDPOINTS         = "endpoint_data"
COLLECTION_SITE_DATA         = "site_enrollment"
COLLECTION_FDA_REVIEWS       = "fda_statistical_reviews"

# ── Embedding Model ──────────────────────────────────────────────────────────
# Domain-specific biomedical embeddings — significantly better than general models
# for clinical trial terminology
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
EMBEDDING_DIMENSION = 768
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Retrieval Parameters ─────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 25            # Retrieve 25 from each collection
TOP_K_AFTER_RERANK = 15         # Keep 15 after reranking
SIMILARITY_THRESHOLD = 0.35

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_MODEL = "claude-opus-4-6"
LLM_MAX_TOKENS = 4096
LLM_TEMPERATURE = 0.1           # Low temperature for factual synthesis

# ── Rate Limits ───────────────────────────────────────────────────────────────
PUBMED_RATE_LIMIT = 10 if NCBI_API_KEY else 3     # requests per second
FDA_API_RATE_LIMIT = 120000 if FDA_API_KEY else 1000  # requests per day
WHO_RATE_LIMIT = 2                                 # conservative

# ── Processing ────────────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 80
BATCH_SIZE_EMBEDDING = 64        # Batch size for embedding generation
MIN_ABSTRACT_LENGTH = 100        # Skip abstracts shorter than this

# ── Data Recency ──────────────────────────────────────────────────────────────
RECENT_YEARS_WINDOW = 10         # Focus on last 10 years of trials
FLAG_OLD_EVIDENCE_YEARS = 5      # Flag evidence older than 5 years
