"""
Multi-collection hybrid retrieval engine.

Implements a two-stage retrieval pipeline:
1. Stage 1: Semantic search across relevant ChromaDB collections
2. Stage 2: Cross-encoder reranking of merged candidates

Also supports BM25 keyword search as a hybrid component to improve
recall for specific medical terminology (acronyms, drug names, NCT IDs).
"""

from loguru import logger
from retrieval.query_router import RetrievalStrategy

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import TOP_K_RETRIEVAL, TOP_K_AFTER_RERANK, SIMILARITY_THRESHOLD


class RetrievalEngine:
    """
    Orchestrates multi-collection retrieval with reranking.
    """

    def __init__(self, vector_store):
        """
        vector_store: TrialMindVectorStore instance
        """
        self.vs = vector_store

    def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy,
        protocol_context: dict = None
    ) -> dict:
        """
        Execute the full retrieval pipeline for a query.

        Returns:
        {
            'candidates': list of reranked document dicts,
            'stats': retrieval statistics dict
        }
        """
        all_candidates = []
        retrieval_stats = {
            'collections_queried': [],
            'raw_candidates': 0,
            'after_dedup': 0,
            'after_rerank': 0
        }

        # Augment query with protocol context if provided
        augmented_query = self._augment_query(query, protocol_context)

        # Stage 1: Query primary collections
        for collection_name in strategy.primary_collections:
            candidates = self._query_collection(
                collection_name=collection_name,
                query=augmented_query,
                n_results=strategy.top_k,
                where_filter=strategy.metadata_filters if strategy.metadata_filters else None
            )
            all_candidates.extend(candidates)
            retrieval_stats['collections_queried'].append(collection_name)
            logger.debug(f"Retrieved {len(candidates)} from {collection_name}")

        # Stage 1b: Query secondary collections if primary didn't get enough
        if len(all_candidates) < strategy.top_k:
            for collection_name in strategy.secondary_collections:
                if collection_name == "pubmed_trial_results" and not strategy.include_pubmed:
                    continue
                candidates = self._query_collection(
                    collection_name=collection_name,
                    query=augmented_query,
                    n_results=strategy.top_k // 2,
                    where_filter=None  # No filters on secondary
                )
                all_candidates.extend(candidates)
                retrieval_stats['collections_queried'].append(collection_name)

        retrieval_stats['raw_candidates'] = len(all_candidates)

        # Deduplicate by document ID
        all_candidates = self._deduplicate(all_candidates)
        retrieval_stats['after_dedup'] = len(all_candidates)

        # Filter by similarity threshold
        all_candidates = [
            c for c in all_candidates
            if c.get('distance', 1.0) <= (1.0 - SIMILARITY_THRESHOLD)
        ]

        # Stage 2: Rerank
        if all_candidates:
            reranked = self.vs.rerank(
                query=augmented_query,
                candidates=all_candidates,
                top_k=TOP_K_AFTER_RERANK
            )
        else:
            reranked = []

        retrieval_stats['after_rerank'] = len(reranked)

        logger.info(
            f"Retrieval: {retrieval_stats['raw_candidates']} raw → "
            f"{retrieval_stats['after_dedup']} deduped → "
            f"{retrieval_stats['after_rerank']} reranked"
        )

        return {
            'candidates': reranked,
            'stats': retrieval_stats
        }

    def _query_collection(
        self,
        collection_name: str,
        query: str,
        n_results: int,
        where_filter: dict = None
    ) -> list:
        """
        Query a single ChromaDB collection and return formatted candidates.
        """
        try:
            results = self.vs.query_collection(
                collection_name=collection_name,
                query_text=query,
                n_results=n_results,
                where_filter=where_filter
            )

            candidates = []
            if not results or not results.get('ids') or not results['ids'][0]:
                return candidates

            ids = results['ids'][0]
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]

            for doc_id, text, meta, dist in zip(ids, documents, metadatas, distances):
                candidates.append({
                    'id': doc_id,
                    'text': text,
                    'metadata': meta,
                    'distance': dist,
                    'collection': collection_name
                })

            return candidates

        except Exception as e:
            logger.error(f"Error querying {collection_name}: {e}")
            return []

    def _deduplicate(self, candidates: list) -> list:
        """
        Remove duplicate documents by ID.
        For duplicates, keep the one with the lowest distance (best match).
        """
        seen = {}
        for candidate in candidates:
            doc_id = candidate['id']
            if doc_id not in seen or candidate['distance'] < seen[doc_id]['distance']:
                seen[doc_id] = candidate

        return list(seen.values())

    def _augment_query(self, query: str, protocol_context: dict = None) -> str:
        """
        Augment the base query with protocol context for better retrieval.
        If the user has specified indication/phase, prepend to query.
        """
        if not protocol_context:
            return query

        augmentations = []

        if protocol_context.get('indication'):
            augmentations.append(f"indication: {protocol_context['indication']}")

        if protocol_context.get('phase'):
            augmentations.append(f"phase: {protocol_context['phase']}")

        if protocol_context.get('primary_endpoint'):
            augmentations.append(f"endpoint: {protocol_context['primary_endpoint']}")

        if augmentations:
            context_str = " | ".join(augmentations)
            return f"{query} [{context_str}]"

        return query

    def retrieve_comparable_trials(
        self,
        indication: str,
        phase: str,
        n_results: int = 20
    ) -> list:
        """
        Retrieve comparable trials for a given indication and phase.
        Used for direct benchmarking without a natural language query.
        """
        query = f"clinical trial {phase} {indication}"

        filters = {}
        if phase:
            phase_normalized = self._normalize_phase(phase)
            if phase_normalized:
                filters['phase'] = phase_normalized

        candidates = self._query_collection(
            collection_name='trial_profiles',
            query=query,
            n_results=n_results,
            where_filter=filters if filters else None
        )

        return self.vs.rerank(query=query, candidates=candidates, top_k=n_results)

    def _normalize_phase(self, phase_str: str) -> str:
        """Normalize phase string to metadata format."""
        if not phase_str:
            return ""
        phase_lower = phase_str.lower()
        if 'phase 3' in phase_lower or 'phase iii' in phase_lower:
            return 'phase_3'
        if 'phase 2' in phase_lower or 'phase ii' in phase_lower:
            return 'phase_2'
        if 'phase 1' in phase_lower or 'phase i' in phase_lower:
            return 'phase_1'
        return ""
