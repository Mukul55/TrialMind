"""
Cross-encoder reranking module.

The reranker significantly improves retrieval precision over cosine similarity alone.
Cross-encoders jointly encode query+document, capturing relevance signals that
bi-encoders miss (e.g., negations, specific numeric constraints).

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Fast enough for real-time reranking (< 500ms for 25 candidates)
- Good balance of speed vs. accuracy
- Widely used in production RAG systems

For higher accuracy at cost of speed, could upgrade to:
- cross-encoder/ms-marco-electra-base (slower, more accurate)
- Cohere Rerank API (cloud, paid)
"""

from loguru import logger

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import RERANKER_MODEL, TOP_K_AFTER_RERANK


class ClinicalTrialReranker:
    """
    Wraps cross-encoder reranking with clinical trial-specific optimizations.
    """

    def __init__(self):
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading reranker: {RERANKER_MODEL}")
        self.model = CrossEncoder(RERANKER_MODEL)

    def rerank(
        self,
        query: str,
        candidates: list,
        top_k: int = TOP_K_AFTER_RERANK,
        boost_recent: bool = True,
        boost_completed: bool = True
    ) -> list:
        """
        Rerank candidates using cross-encoder scores with optional boosts.

        Boosts:
        - boost_recent: trials from last 5 years get slight score boost
        - boost_completed: completed trials get slight score boost over terminated

        candidates: list of dicts with 'text', 'metadata', etc.
        """
        if not candidates:
            return []

        # Get cross-encoder scores
        query_doc_pairs = [[query, c['text']] for c in candidates]

        try:
            scores = self.model.predict(query_doc_pairs)
        except Exception as e:
            logger.error(f"Reranker error: {e}")
            # Fallback: return candidates sorted by distance
            return sorted(candidates, key=lambda x: x.get('distance', 1.0))[:top_k]

        # Apply domain-specific boosts
        from datetime import datetime
        current_year = datetime.now().year

        final_scores = []
        for score, candidate in zip(scores, candidates):
            adjusted_score = float(score)
            meta = candidate.get('metadata', {})

            if boost_recent:
                start_year = meta.get('start_year', 2000)
                age = current_year - start_year
                if age <= 3:
                    adjusted_score += 0.1
                elif age <= 5:
                    adjusted_score += 0.05
                elif age > 10:
                    adjusted_score -= 0.05

            if boost_completed:
                status = meta.get('status', '')
                if status == 'Completed':
                    adjusted_score += 0.05
                elif status == 'Terminated':
                    adjusted_score -= 0.02  # Slight penalty but still useful

            final_scores.append((adjusted_score, candidate))

        # Sort by adjusted score
        final_scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, candidate in final_scores[:top_k]:
            candidate['rerank_score'] = score
            results.append(candidate)

        return results

    def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        try:
            score = self.model.predict([[query, document]])[0]
            return float(score)
        except Exception as e:
            logger.error(f"Scoring error: {e}")
            return 0.0

    def filter_by_threshold(
        self,
        candidates: list,
        threshold: float = 0.0
    ) -> list:
        """Filter candidates below a rerank score threshold."""
        return [c for c in candidates if c.get('rerank_score', 0) >= threshold]
