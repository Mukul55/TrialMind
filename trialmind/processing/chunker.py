"""
Domain-aware chunking for clinical trial data.

A single trial is represented as FIVE different chunk types, each optimized
for a different retrieval use case:

1. DESIGN CHUNK      → answers "what design parameters were used?"
2. RESULTS CHUNK     → answers "what were the outcomes?"
3. ELIGIBILITY CHUNK → answers "what criteria were used / what caused recruitment issues?"
4. ENDPOINT CHUNK    → answers "what endpoints were used and accepted?"
5. SITE CHUNK        → answers "where was this trial conducted and how fast was enrollment?"

Each chunk type is stored in a separate ChromaDB collection with specialized metadata.
This avoids the common mistake of mixing all trial information into one dense chunk
that retrieves poorly for specific sub-questions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS


class TrialChunker:

    def create_all_chunks(self, profile: dict) -> dict:
        """
        Given a TrialProfile dict, return five specialized chunks.
        Each chunk is a dict with 'text' and 'metadata' keys.
        """
        chunks = {
            'design': self._design_chunk(profile),
            'results': self._results_chunk(profile),
            'eligibility': self._eligibility_chunk(profile),
            'endpoint': self._endpoint_chunk(profile),
            'site': self._site_chunk(profile)
        }
        # Filter out None chunks (when insufficient data exists)
        return {k: v for k, v in chunks.items() if v is not None}

    def _design_chunk(self, p: dict) -> dict:
        """
        Design chunk captures protocol structure.
        Optimized for queries like: "What sample sizes are typical for Phase 2 NSCLC trials?"
        """
        if not p.get('phase') or not p.get('conditions_str'):
            return None

        text = f"""TRIAL DESIGN PROFILE
NCT ID: {p['nct_id']}
Therapeutic Area / Condition: {p['conditions_str']}
Drug(s): {p['drug_names_str']}
Phase: {p['phase']}
Study Design: {p['intervention_model']} | Allocation: {p['allocation']} | Masking: {p['masking']}
Primary Purpose: {p['primary_purpose']}
Arms: {p['number_of_arms']}
Planned Enrollment: {p['planned_enrollment']}
Actual Enrollment: {p['actual_enrollment']}
Enrollment Ratio (Actual/Planned): {f"{p['enrollment_ratio']:.2f}" if p['enrollment_ratio'] else 'N/A'}
Recruitment Challenge: {"YES — enrolled below 80% of target" if p['recruitment_challenge_flag'] else "No"}
Trial Duration: {p['duration_months']} months
Number of Sites: {p['number_of_sites']}
Number of Countries: {p['num_countries']}
Start Year: {p['start_year']}
Study Status: {p['status']}
{"STOPPED EARLY — Reason: " + p['why_stopped'] if p['why_stopped'] else ""}
{"HIGH AMENDMENT FLAG — multiple protocol modifications detected" if p['high_amendment_flag'] else ""}
Summary: {p['brief_summary'][:500]}"""

        metadata = {
            "nct_id": p['nct_id'],
            "chunk_type": "design",
            "phase": self._normalize_phase(p['phase']),
            "conditions_str": p['conditions_str'][:100],
            "drug_names_str": p['drug_names_str'][:100],
            "planned_enrollment": p['planned_enrollment'] or 0,
            "actual_enrollment": p['actual_enrollment'] or 0,
            "recruitment_challenge": bool(p['recruitment_challenge_flag']),
            "status": p['status'],
            "start_year": p['start_year'] or 2000,
            "duration_months": p['duration_months'] or 0,
            "num_countries": p['num_countries'] or 0,
            "high_amendment": bool(p['high_amendment_flag']),
        }

        return {"text": text, "metadata": metadata,
                "id": f"{p['nct_id']}_design"}

    def _results_chunk(self, p: dict) -> dict:
        """
        Results chunk captures outcome data and success/failure.
        Optimized for: "What dropout rates are typical? What were outcome results?"
        """
        text = f"""TRIAL RESULTS SUMMARY
NCT ID: {p['nct_id']}
Condition: {p['conditions_str']}
Drug(s): {p['drug_names_str']}
Phase: {p['phase']}
Final Status: {p['status']}
{"TERMINATED — " + p['why_stopped'] if p['why_stopped'] else "Completed as planned"}
Primary Endpoint: {p['primary_endpoint']}
Endpoint Type: {p['endpoint_type']}
Endpoint Timeframe: {p['primary_endpoint_timeframe']}
Secondary Endpoints: {p['secondary_endpoints_str']}
Planned Enrollment: {p['planned_enrollment']}
Actual Enrollment: {p['actual_enrollment']}
Total Dropouts: {p['total_dropouts']}
Dropout Rate: {f"{p['dropout_rate']*100:.1f}%" if p['dropout_rate'] else 'N/A'}
Trial Duration: {p['duration_months']} months"""

        metadata = {
            "nct_id": p['nct_id'],
            "chunk_type": "results",
            "phase": self._normalize_phase(p['phase']),
            "conditions_str": p['conditions_str'][:100],
            "endpoint_type": p['endpoint_type'],
            "status": p['status'],
            "dropout_rate": p['dropout_rate'] or 0,
            "terminated": p['status'] == 'Terminated',
            "start_year": p['start_year'] or 2000,
        }

        return {"text": text, "metadata": metadata,
                "id": f"{p['nct_id']}_results"}

    def _eligibility_chunk(self, p: dict) -> dict:
        """
        Eligibility chunk captures patient selection criteria.
        Optimized for: "What eligibility criteria caused recruitment problems?"
        """
        if not p['inclusion_criteria'] and not p['exclusion_criteria']:
            return None

        inclusion_text = "\n  - ".join(p['inclusion_criteria'][:10]) if p['inclusion_criteria'] else "Not specified"
        exclusion_text = "\n  - ".join(p['exclusion_criteria'][:10]) if p['exclusion_criteria'] else "Not specified"

        text = f"""TRIAL ELIGIBILITY CRITERIA
NCT ID: {p['nct_id']}
Condition: {p['conditions_str']}
Phase: {p['phase']}
Age Range: {p['min_age']} to {p['max_age']}
Gender: {p['gender']}
Inclusion Criteria ({p['inclusion_count']} total):
  - {inclusion_text}
Exclusion Criteria ({p['exclusion_count']} total):
  - {exclusion_text}
Planned Enrollment: {p['planned_enrollment']}
Actual Enrollment: {p['actual_enrollment']}
Recruitment Outcome: {"UNDERENROLLED — criteria may have been too restrictive" if p['recruitment_challenge_flag'] else "Met enrollment target"}"""

        metadata = {
            "nct_id": p['nct_id'],
            "chunk_type": "eligibility",
            "phase": self._normalize_phase(p['phase']),
            "conditions_str": p['conditions_str'][:100],
            "status": p['status'],
            "recruitment_challenge": bool(p['recruitment_challenge_flag']),
            "inclusion_count": p['inclusion_count'],
            "exclusion_count": p['exclusion_count'],
            "start_year": p['start_year'] or 2000,
        }

        return {"text": text, "metadata": metadata,
                "id": f"{p['nct_id']}_eligibility"}

    def _endpoint_chunk(self, p: dict) -> dict:
        """
        Endpoint chunk for endpoint benchmarking queries.
        Optimized for: "What primary endpoints are accepted for [indication]?"
        """
        if not p['primary_endpoint']:
            return None

        text = f"""CLINICAL TRIAL ENDPOINT DATA
NCT ID: {p['nct_id']}
Therapeutic Area: {p['conditions_str']}
Drug: {p['drug_names_str']}
Phase: {p['phase']}
Primary Endpoint: {p['primary_endpoint']}
Endpoint Classification: {p['endpoint_type']}
Measurement Timeframe: {p['primary_endpoint_timeframe']}
Secondary Endpoints: {p['secondary_endpoints_str']}
Trial Outcome: {p['status']}
{"This endpoint was used in a terminated trial — may indicate endpoint was not achievable" if p['status'] == 'Terminated' else "Trial reached primary completion"}
FDA Regulated: {"Yes" if p['is_fda_regulated'] else "No"}"""

        metadata = {
            "nct_id": p['nct_id'],
            "chunk_type": "endpoint",
            "phase": self._normalize_phase(p['phase']),
            "conditions_str": p['conditions_str'][:100],
            "endpoint_type": p['endpoint_type'],
            "status": p['status'],
            "is_fda_regulated": bool(p['is_fda_regulated']),
            "start_year": p['start_year'] or 2000,
        }

        return {"text": text, "metadata": metadata,
                "id": f"{p['nct_id']}_endpoint"}

    def _site_chunk(self, p: dict) -> dict:
        """
        Site and geography chunk for operational planning.
        Optimized for: "Which countries enroll fastest for [indication]?"
        """
        if not p['countries']:
            return None

        text = f"""TRIAL SITE AND ENROLLMENT DATA
NCT ID: {p['nct_id']}
Condition: {p['conditions_str']}
Phase: {p['phase']}
Countries Conducting Trial: {p['countries_str']}
Number of Countries: {p['num_countries']}
Number of Sites: {p['number_of_sites']}
Planned Enrollment: {p['planned_enrollment']}
Actual Enrollment: {p['actual_enrollment']}
Enrollment Rate vs Target: {f"{p['enrollment_ratio']*100:.0f}%" if p['enrollment_ratio'] else "N/A"}
Trial Duration: {p['duration_months']} months
Enrollment Rate per Month: {
    f"{p['actual_enrollment'] / p['duration_months']:.1f} patients/month"
    if (p['actual_enrollment'] and p['duration_months'] and p['duration_months'] > 0)
    else "N/A"
}
Recruitment Outcome: {"Below target — possible site selection issue" if p['recruitment_challenge_flag'] else "Met enrollment target"}"""

        metadata = {
            "nct_id": p['nct_id'],
            "chunk_type": "site",
            "phase": self._normalize_phase(p['phase']),
            "conditions_str": p['conditions_str'][:100],
            "countries_str": p['countries_str'][:200],
            "num_countries": p['num_countries'],
            "enrollment_ratio": p['enrollment_ratio'] or 0,
            "recruitment_challenge": bool(p['recruitment_challenge_flag']),
            "duration_months": p['duration_months'] or 0,
            "start_year": p['start_year'] or 2000,
        }

        return {"text": text, "metadata": metadata,
                "id": f"{p['nct_id']}_site"}

    def _normalize_phase(self, phase_str: str) -> str:
        """Normalize phase strings for consistent metadata filtering."""
        if not phase_str:
            return "unknown"
        phase_lower = phase_str.lower()
        if 'phase 1/phase 2' in phase_lower or 'phase 1/2' in phase_lower:
            return "phase_1_2"
        if 'phase 2/phase 3' in phase_lower or 'phase 2/3' in phase_lower:
            return "phase_2_3"
        if 'phase 1' in phase_lower:
            return "phase_1"
        if 'phase 2' in phase_lower:
            return "phase_2"
        if 'phase 3' in phase_lower:
            return "phase_3"
        if 'phase 4' in phase_lower:
            return "phase_4"
        return "unknown"
