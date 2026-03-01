"""
Query Router: Classifies user queries and determines the optimal
retrieval strategy for each question type.

The router solves a critical problem: the same question about clinical trials
could require different retrieval strategies:
- "What sample size is typical?" → needs design chunks + quantitative analysis
- "What eligibility criteria work?" → needs eligibility chunks
- "What endpoints does FDA accept?" → needs endpoint chunks + FDA review data
- "Which countries enroll fastest?" → needs site chunks
- "Why do trials fail?" → needs results chunks + PubMed abstracts

Getting this routing right is the difference between a useful system and
a generic RAG that returns vaguely relevant documents.
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import (
    COLLECTION_TRIAL_PROFILES, COLLECTION_TRIAL_RESULTS, COLLECTION_ELIGIBILITY,
    COLLECTION_ENDPOINTS, COLLECTION_SITE_DATA, COLLECTION_FDA_REVIEWS
)


class QueryIntent(Enum):
    SAMPLE_SIZE_BENCHMARKING = "sample_size"
    ENDPOINT_SELECTION = "endpoint"
    ELIGIBILITY_OPTIMIZATION = "eligibility"
    SITE_SELECTION = "site"
    DROPOUT_RETENTION = "dropout"
    TRIAL_FAILURE_ANALYSIS = "failure"
    PROTOCOL_REVIEW = "protocol_review"   # Comprehensive protocol assessment
    GENERAL_BENCHMARK = "general"


@dataclass
class RetrievalStrategy:
    primary_collections: list      # Query these first
    secondary_collections: list    # Query these if primary insufficient
    metadata_filters: dict         # ChromaDB where clause
    top_k: int
    include_pubmed: bool
    synthesis_mode: str            # Guides LLM synthesis style


class QueryRouter:

    INTENT_PATTERNS = {
        QueryIntent.SAMPLE_SIZE_BENCHMARKING: [
            'sample size', 'enrollment', 'how many patients', 'how many subjects',
            'number of participants', 'power', 'powered', 'underpowered',
            'enrollment target', 'n=', 'cohort size'
        ],
        QueryIntent.ENDPOINT_SELECTION: [
            'endpoint', 'primary outcome', 'outcome measure', 'efficacy outcome',
            'what outcome', 'regulatory endpoint', 'fda accept', 'ema accept',
            'overall survival', 'progression free', 'response rate', 'ORR',
            'PFS', 'OS', 'biomarker endpoint'
        ],
        QueryIntent.ELIGIBILITY_OPTIMIZATION: [
            'eligibility', 'inclusion criteria', 'exclusion criteria',
            'who qualifies', 'patient selection', 'criteria', 'restrict',
            'age range', 'comorbidity', 'prior treatment', 'washout',
            'biomarker requirement', 'performance status'
        ],
        QueryIntent.SITE_SELECTION: [
            'site', 'country', 'countries', 'region', 'enroll faster', 'enrollment rate',
            'fastest enrollment', 'site activation', 'investigator', 'geography',
            'where to run', 'international'
        ],
        QueryIntent.DROPOUT_RETENTION: [
            'dropout', 'attrition', 'retention', 'withdrawal', 'lost to follow',
            'discontinuation', 'completion rate', 'adherence', 'compliance'
        ],
        QueryIntent.TRIAL_FAILURE_ANALYSIS: [
            'fail', 'failure', 'terminated', 'why stopped', 'early termination',
            'did not meet', 'negative trial', 'amendment', 'protocol change',
            'futility', 'safety concern', 'why trials fail'
        ],
        QueryIntent.PROTOCOL_REVIEW: [
            'review my protocol', 'assess my design', 'evaluate my protocol',
            'is this design sound', 'protocol feedback', 'design critique',
            'protocol optimization', 'my trial design'
        ]
    }

    def classify_intent(self, query: str) -> QueryIntent:
        """
        Rule-based intent classification with keyword matching.
        Falls back to GENERAL_BENCHMARK if no specific intent detected.
        """
        query_lower = query.lower()

        # Score each intent
        scores = {}
        for intent, keywords in self.INTENT_PATTERNS.items():
            scores[intent] = sum(1 for kw in keywords if kw in query_lower)

        best_intent = max(scores, key=scores.get)

        if scores[best_intent] == 0:
            return QueryIntent.GENERAL_BENCHMARK

        return best_intent

    def extract_filters(self, query: str) -> dict:
        """
        Extract metadata filters from the query text.
        These filters narrow ChromaDB retrieval before semantic search.
        """
        filters = {}
        query_lower = query.lower()

        # Phase detection
        phase_map = {
            'phase 1': 'phase_1', 'phase i': 'phase_1', 'phase i/ii': 'phase_1_2',
            'phase 1/2': 'phase_1_2', 'phase 2': 'phase_2', 'phase ii': 'phase_2',
            'phase 2/3': 'phase_2_3', 'phase ii/iii': 'phase_2_3',
            'phase 3': 'phase_3', 'phase iii': 'phase_3',
            'phase 4': 'phase_4', 'phase iv': 'phase_4'
        }
        for pattern, normalized in phase_map.items():
            if pattern in query_lower:
                filters['phase'] = normalized
                break

        # Recency filter — last N years
        year_match = re.search(r'(last|past)\s+(\d+)\s+years?', query_lower)
        if year_match:
            n_years = int(year_match.group(2))
            filters['start_year'] = {"$gte": datetime.now().year - n_years}

        return filters

    def build_strategy(self, query: str) -> RetrievalStrategy:
        """
        Build the complete retrieval strategy for a query.
        """
        intent = self.classify_intent(query)
        filters = self.extract_filters(query)

        strategies = {
            QueryIntent.SAMPLE_SIZE_BENCHMARKING: RetrievalStrategy(
                primary_collections=[COLLECTION_TRIAL_PROFILES],
                secondary_collections=[COLLECTION_TRIAL_RESULTS, "pubmed_trial_results"],
                metadata_filters=filters,
                top_k=20,
                include_pubmed=True,
                synthesis_mode="quantitative_benchmark"
            ),
            QueryIntent.ENDPOINT_SELECTION: RetrievalStrategy(
                primary_collections=[COLLECTION_ENDPOINTS, COLLECTION_FDA_REVIEWS],
                secondary_collections=["pubmed_trial_results"],
                metadata_filters=filters,
                top_k=20,
                include_pubmed=True,
                synthesis_mode="endpoint_analysis"
            ),
            QueryIntent.ELIGIBILITY_OPTIMIZATION: RetrievalStrategy(
                primary_collections=[COLLECTION_ELIGIBILITY],
                secondary_collections=[COLLECTION_TRIAL_PROFILES, "pubmed_trial_results"],
                metadata_filters=filters,
                top_k=20,
                include_pubmed=False,
                synthesis_mode="eligibility_analysis"
            ),
            QueryIntent.SITE_SELECTION: RetrievalStrategy(
                primary_collections=[COLLECTION_SITE_DATA],
                secondary_collections=[COLLECTION_TRIAL_PROFILES],
                metadata_filters=filters,
                top_k=20,
                include_pubmed=False,
                synthesis_mode="site_analysis"
            ),
            QueryIntent.DROPOUT_RETENTION: RetrievalStrategy(
                primary_collections=[COLLECTION_TRIAL_RESULTS],
                secondary_collections=["pubmed_trial_results"],
                metadata_filters=filters,
                top_k=20,
                include_pubmed=True,
                synthesis_mode="dropout_analysis"
            ),
            QueryIntent.TRIAL_FAILURE_ANALYSIS: RetrievalStrategy(
                primary_collections=[COLLECTION_TRIAL_RESULTS, COLLECTION_TRIAL_PROFILES],
                secondary_collections=["pubmed_trial_results"],
                metadata_filters=filters,
                top_k=25,
                include_pubmed=True,
                synthesis_mode="failure_analysis"
            ),
            QueryIntent.PROTOCOL_REVIEW: RetrievalStrategy(
                primary_collections=[
                    COLLECTION_TRIAL_PROFILES, COLLECTION_ENDPOINTS,
                    COLLECTION_ELIGIBILITY, COLLECTION_SITE_DATA
                ],
                secondary_collections=[COLLECTION_TRIAL_RESULTS, "pubmed_trial_results"],
                metadata_filters=filters,
                top_k=15,  # Per collection — comprehensive retrieval
                include_pubmed=True,
                synthesis_mode="comprehensive_protocol_review"
            ),
            QueryIntent.GENERAL_BENCHMARK: RetrievalStrategy(
                primary_collections=[
                    COLLECTION_TRIAL_PROFILES, COLLECTION_TRIAL_RESULTS
                ],
                secondary_collections=["pubmed_trial_results"],
                metadata_filters=filters,
                top_k=15,
                include_pubmed=True,
                synthesis_mode="general"
            )
        }

        return strategies[intent]
