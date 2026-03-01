"""
Embedding generation and ChromaDB vector store population.

Uses PubMedBERT-based embeddings for biomedical text — significantly outperforms
general-purpose embeddings on clinical terminology retrieval benchmarks.

Model: pritamdeka/S-PubMedBert-MS-MARCO
- Fine-tuned on medical question-answer pairs
- Captures clinical synonymy (e.g., "myocardial infarction" = "heart attack")
- Free, runs locally, no API calls needed
"""

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from tqdm import tqdm
from loguru import logger

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import (
    EMBEDDING_MODEL, RERANKER_MODEL, CHROMA_PERSIST_DIR,
    COLLECTION_TRIAL_PROFILES, COLLECTION_TRIAL_RESULTS, COLLECTION_ELIGIBILITY,
    COLLECTION_ENDPOINTS, COLLECTION_SITE_DATA, COLLECTION_FDA_REVIEWS,
    BATCH_SIZE_EMBEDDING, TOP_K_RETRIEVAL, TOP_K_AFTER_RERANK
)


class TrialMindVectorStore:

    def __init__(self):
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # Persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # Create collections — one per chunk type for precise filtering
        self.collections = {}
        collection_names = [
            COLLECTION_TRIAL_PROFILES,    # Design chunks
            COLLECTION_TRIAL_RESULTS,     # Results chunks
            COLLECTION_ELIGIBILITY,       # Eligibility chunks
            COLLECTION_ENDPOINTS,         # Endpoint chunks
            COLLECTION_SITE_DATA,         # Site/geography chunks
            COLLECTION_FDA_REVIEWS,       # FDA statistical review text
            "pubmed_trial_results"        # PubMed abstracts
        ]

        for name in collection_names:
            self.collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )

        logger.info(f"Initialized {len(self.collections)} ChromaDB collections")

    def embed_texts(self, texts: list) -> np.ndarray:
        """Generate embeddings in batches with progress tracking."""
        all_embeddings = []
        for i in tqdm(
            range(0, len(texts), BATCH_SIZE_EMBEDDING),
            desc="Generating embeddings"
        ):
            batch = texts[i:i+BATCH_SIZE_EMBEDDING]
            embeddings = self.embed_model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True  # Cosine similarity requires normalized vectors
            )
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def add_trial_chunks(self, trial_chunks_list: list):
        """
        Add trial chunks to their respective collections.

        trial_chunks_list: list of dicts from TrialChunker.create_all_chunks()
        Each dict maps chunk_type -> {text, metadata, id}

        Routing:
        - design      → COLLECTION_TRIAL_PROFILES
        - results     → COLLECTION_TRIAL_RESULTS
        - eligibility → COLLECTION_ELIGIBILITY
        - endpoint    → COLLECTION_ENDPOINTS
        - site        → COLLECTION_SITE_DATA
        """
        chunk_type_to_collection = {
            "design": COLLECTION_TRIAL_PROFILES,
            "results": COLLECTION_TRIAL_RESULTS,
            "eligibility": COLLECTION_ELIGIBILITY,
            "endpoint": COLLECTION_ENDPOINTS,
            "site": COLLECTION_SITE_DATA,
        }

        # Group chunks by type for batch processing
        by_type = {ct: [] for ct in chunk_type_to_collection}

        for trial_chunks in trial_chunks_list:
            for chunk_type, chunk in trial_chunks.items():
                if chunk_type in by_type:
                    by_type[chunk_type].append(chunk)

        # Embed and store each chunk type
        for chunk_type, chunks in by_type.items():
            if not chunks:
                continue

            collection_name = chunk_type_to_collection[chunk_type]
            collection = self.collections[collection_name]

            # Check what's already stored (avoid re-embedding)
            existing_ids = set()
            try:
                existing = collection.get(include=[])
                existing_ids = set(existing['ids'])
            except Exception:
                pass

            new_chunks = [c for c in chunks if c['id'] not in existing_ids]

            if not new_chunks:
                logger.info(f"No new chunks for {chunk_type} — all already indexed")
                continue

            logger.info(f"Embedding {len(new_chunks)} {chunk_type} chunks...")

            texts = [c['text'] for c in new_chunks]
            embeddings = self.embed_texts(texts)

            # Add in sub-batches (ChromaDB has insertion limits)
            sub_batch_size = 500
            for i in range(0, len(new_chunks), sub_batch_size):
                sub = new_chunks[i:i+sub_batch_size]
                sub_embeddings = embeddings[i:i+sub_batch_size]

                collection.add(
                    ids=[c['id'] for c in sub],
                    embeddings=sub_embeddings.tolist(),
                    documents=[c['text'] for c in sub],
                    metadatas=[c['metadata'] for c in sub]
                )

            logger.info(f"Added {len(new_chunks)} {chunk_type} chunks to {collection_name}")

    def add_pubmed_records(self, records: list):
        """Add PubMed abstract records to the pubmed collection."""
        collection = self.collections['pubmed_trial_results']

        existing = set()
        try:
            existing = set(collection.get(include=[])['ids'])
        except Exception:
            pass

        new_records = [r for r in records if r['id'] not in existing]
        if not new_records:
            logger.info("No new PubMed records to add")
            return

        texts = [r['text'] for r in new_records]
        embeddings = self.embed_texts(texts)

        metadatas = []
        for r in new_records:
            metadatas.append({
                "pmid": r['pmid'],
                "year": str(r['year'])[:4],
                "journal": r['journal'][:100],
                "mesh_terms_str": r['mesh_terms_str'][:300],
                "pub_types": str(r['pub_types'])[:200],
                "nct_references": str(r['nct_references'])[:100],
                "source": "pubmed"
            })

        for i in range(0, len(new_records), 500):
            sub = new_records[i:i+500]
            sub_embeddings = embeddings[i:i+500]
            collection.add(
                ids=[r['id'] for r in sub],
                embeddings=sub_embeddings.tolist(),
                documents=[r['text'] for r in sub],
                metadatas=metadatas[i:i+500]
            )

        logger.info(f"Added {len(new_records)} PubMed records")

    def add_fda_records(self, records: list):
        """Add FDA approval records to the FDA reviews collection."""
        collection = self.collections[COLLECTION_FDA_REVIEWS]

        existing = set()
        try:
            existing = set(collection.get(include=[])['ids'])
        except Exception:
            pass

        new_records = [r for r in records if r['id'] not in existing]
        if not new_records:
            logger.info("No new FDA records to add")
            return

        texts = [r['text'] for r in new_records]
        embeddings = self.embed_texts(texts)

        metadatas = [r.get('metadata', {}) for r in new_records]

        for i in range(0, len(new_records), 500):
            sub = new_records[i:i+500]
            sub_embeddings = embeddings[i:i+500]
            collection.add(
                ids=[r['id'] for r in sub],
                embeddings=sub_embeddings.tolist(),
                documents=[r['text'] for r in sub],
                metadatas=metadatas[i:i+500]
            )

        logger.info(f"Added {len(new_records)} FDA records")

    def query_collection(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = TOP_K_RETRIEVAL,
        where_filter: dict = None
    ) -> dict:
        """
        Query a single collection with optional metadata filtering.
        Returns ChromaDB results dict.
        """
        collection = self.collections[collection_name]
        query_embedding = self.embed_model.encode(
            [query_text], normalize_embeddings=True
        ).tolist()

        count = collection.count()
        if count == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        kwargs = {
            "query_embeddings": query_embedding,
            "n_results": min(n_results, count),
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            kwargs["where"] = where_filter

        return collection.query(**kwargs)

    def rerank(
        self, query: str, candidates: list, top_k: int = TOP_K_AFTER_RERANK
    ) -> list:
        """
        Rerank retrieved candidates using a cross-encoder.
        Cross-encoders compare query+document jointly — much more accurate than
        bi-encoder similarity scores alone but too slow for first-stage retrieval.

        candidates: list of dicts with 'text', 'metadata', 'distance'
        """
        if not candidates:
            return []

        query_doc_pairs = [[query, c['text']] for c in candidates]
        scores = self.reranker.predict(query_doc_pairs)

        # Sort by reranker score descending
        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, candidate in scored[:top_k]:
            candidate['rerank_score'] = float(score)
            results.append(candidate)

        return results
