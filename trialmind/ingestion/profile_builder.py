"""
Unified Trial Profile Builder.

Merges data from multiple sources (AACT, PubMed, FDA, WHO ICTRP) into
a single standardized TrialProfile record for ChromaDB storage.

The Profile Builder handles:
1. Data deduplication (same trial across multiple registries)
2. Field normalization (date formats, phase names, enrollment figures)
3. Quality scoring (completeness, data recency)
4. Cross-source enrichment (AACT + PubMed result matching)
"""

import re
from datetime import datetime
from loguru import logger
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import FLAG_OLD_EVIDENCE_YEARS


class TrialProfileBuilder:
    """
    Builds unified TrialProfile records from multi-source data.
    """

    CURRENT_YEAR = datetime.now().year

    def __init__(self):
        # Track NCT IDs seen to prevent duplicates
        self.seen_nct_ids = set()
        self.seen_pmids = set()

    def build_from_aact(self, aact_profile: dict) -> Optional[dict]:
        """
        Convert an AACT profile dict to a standardized TrialProfile.
        AACT profiles are already well-structured from AACTIngestion.
        """
        nct_id = aact_profile.get('nct_id', '')
        if not nct_id or nct_id in self.seen_nct_ids:
            return None

        self.seen_nct_ids.add(nct_id)

        # Compute data quality score
        quality_score = self._compute_quality_score(aact_profile)

        # Flag old evidence
        start_year = aact_profile.get('start_year', self.CURRENT_YEAR)
        age_years = self.CURRENT_YEAR - (start_year or self.CURRENT_YEAR)
        is_old_evidence = age_years > FLAG_OLD_EVIDENCE_YEARS

        enriched = {
            **aact_profile,
            "quality_score": quality_score,
            "age_years": age_years,
            "is_old_evidence": is_old_evidence,
            "profile_type": "aact_trial",
            "profile_version": "1.0",
            "indexed_at": datetime.now().isoformat(),
        }

        return enriched

    def build_from_pubmed(self, pubmed_record: dict) -> Optional[dict]:
        """
        Convert a PubMed record to a TrialProfile-like structure.
        PubMed records are lighter — primarily text with metadata.
        """
        pmid = pubmed_record.get('pmid', '')
        if not pmid or pmid in self.seen_pmids:
            return None

        self.seen_pmids.add(pmid)

        year_str = pubmed_record.get('year', str(self.CURRENT_YEAR))
        try:
            pub_year = int(str(year_str)[:4])
        except (ValueError, TypeError):
            pub_year = self.CURRENT_YEAR

        age_years = self.CURRENT_YEAR - pub_year
        is_old_evidence = age_years > FLAG_OLD_EVIDENCE_YEARS

        # Extract phase from text
        text = pubmed_record.get('text', '')
        phase = self._extract_phase_from_text(text)

        # Extract conditions from MeSH terms
        mesh_terms = pubmed_record.get('mesh_terms', [])
        conditions = self._extract_conditions_from_mesh(mesh_terms)

        profile = {
            "id": pubmed_record.get('id'),
            "pmid": pmid,
            "source": "pubmed",
            "title": pubmed_record.get('title', ''),
            "abstract": pubmed_record.get('abstract', ''),
            "text": pubmed_record.get('text', ''),
            "phase": phase,
            "conditions_str": "; ".join(conditions[:3]),
            "conditions": conditions,
            "mesh_terms": mesh_terms,
            "pub_year": pub_year,
            "journal": pubmed_record.get('journal', ''),
            "nct_references": pubmed_record.get('nct_references', []),
            "age_years": age_years,
            "is_old_evidence": is_old_evidence,
            "profile_type": "pubmed_abstract",
            "indexed_at": datetime.now().isoformat(),
        }

        return profile

    def enrich_aact_with_pubmed(
        self, aact_profile: dict, pubmed_records: list
    ) -> dict:
        """
        Enrich an AACT trial profile with matching PubMed result data.
        Matches by NCT ID reference in PubMed abstracts.
        """
        nct_id = aact_profile.get('nct_id', '')
        if not nct_id:
            return aact_profile

        matching_pubs = [
            rec for rec in pubmed_records
            if nct_id in rec.get('nct_references', [])
        ]

        if not matching_pubs:
            return aact_profile

        # Extract result snippets from matching abstracts
        result_snippets = []
        for pub in matching_pubs[:3]:  # Use top 3 matching publications
            abstract = pub.get('abstract', '')
            # Extract results section if structured
            results_match = re.search(
                r'(?i)results?[:\s]+(.*?)(?=conclusions?|$)',
                abstract,
                re.DOTALL
            )
            if results_match:
                result_snippets.append(results_match.group(1).strip()[:300])

        enriched = {
            **aact_profile,
            "published_results": result_snippets,
            "publication_count": len(matching_pubs),
            "has_published_results": len(matching_pubs) > 0,
            "result_journals": [p.get('journal', '') for p in matching_pubs[:3]],
        }

        return enriched

    def _compute_quality_score(self, profile: dict) -> float:
        """
        Compute a data completeness/quality score (0.0 to 1.0).
        Used for retrieval ranking — higher quality profiles get slight boost.
        """
        score = 0.0
        weights = {
            'primary_endpoint': 0.20,
            'planned_enrollment': 0.15,
            'actual_enrollment': 0.15,
            'conditions_str': 0.10,
            'drug_names_str': 0.10,
            'inclusion_criteria': 0.10,
            'exclusion_criteria': 0.10,
            'countries': 0.05,
            'duration_months': 0.05,
        }

        for field, weight in weights.items():
            val = profile.get(field)
            if val and (not isinstance(val, list) or len(val) > 0):
                score += weight

        return round(score, 2)

    def _extract_phase_from_text(self, text: str) -> str:
        """Extract phase designation from free text."""
        text_lower = text.lower()
        phase_patterns = [
            (r'phase\s+3', 'Phase 3'),
            (r'phase\s+iii', 'Phase 3'),
            (r'phase\s+2', 'Phase 2'),
            (r'phase\s+ii', 'Phase 2'),
            (r'phase\s+1', 'Phase 1'),
            (r'phase\s+i\b', 'Phase 1'),
        ]
        for pattern, label in phase_patterns:
            if re.search(pattern, text_lower):
                return label
        return ""

    def _extract_conditions_from_mesh(self, mesh_terms: list) -> list:
        """
        Extract disease/condition terms from MeSH vocabulary.
        Filters out non-disease MeSH terms (methods, drugs, etc.)
        """
        # Disease-related MeSH categories (simplified heuristic)
        disease_keywords = [
            'cancer', 'carcinoma', 'lymphoma', 'leukemia', 'tumor',
            'disease', 'disorder', 'syndrome', 'failure', 'infection',
            'diabetes', 'hypertension', 'arthritis', 'alzheimer',
            'parkinson', 'depression', 'asthma', 'fibrosis',
            'melanoma', 'sarcoma', 'glioblastoma', 'myeloma'
        ]

        conditions = []
        for term in mesh_terms:
            term_lower = term.lower()
            if any(kw in term_lower for kw in disease_keywords):
                conditions.append(term)

        return conditions[:5]

    def validate_profile(self, profile: dict) -> bool:
        """
        Validate that a profile has minimum required fields for useful retrieval.
        """
        required = ['id' if profile.get('source') == 'pubmed' else 'nct_id']
        for field in required:
            if not profile.get(field):
                return False

        # Must have some text content for embedding
        text_fields = ['text', 'brief_summary', 'abstract', 'title']
        has_text = any(
            profile.get(f) and len(str(profile.get(f, ''))) > 50
            for f in text_fields
        )

        return has_text

    def build_batch(self, aact_profiles: list, pubmed_records: list = None) -> list:
        """
        Build a batch of unified profiles from AACT and optionally PubMed data.

        Performs cross-source enrichment where PubMed records reference AACT trials.
        """
        pubmed_records = pubmed_records or []
        unified_profiles = []

        logger.info(f"Building unified profiles for {len(aact_profiles)} AACT trials")

        for aact_profile in aact_profiles:
            # Build base profile
            profile = self.build_from_aact(aact_profile)
            if not profile:
                continue

            # Enrich with PubMed if available
            if pubmed_records:
                profile = self.enrich_aact_with_pubmed(profile, pubmed_records)

            if self.validate_profile(profile):
                unified_profiles.append(profile)

        logger.info(f"Built {len(unified_profiles)} valid unified profiles")
        return unified_profiles
