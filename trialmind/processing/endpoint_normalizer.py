"""
Endpoint Terminology Normalization.

Clinical trials use highly variable terminology for the same endpoints:
- "Overall Survival" = "OS" = "Time to Death" = "Survival Time"
- "Progression-Free Survival" = "PFS" = "Progression-Free Interval" = "Time to Progression"
- "Objective Response Rate" = "ORR" = "Response Rate" = "Tumor Response Rate"

This normalization is critical for accurate retrieval — without it,
a query about "PFS" would miss trials that used "time to progression".

Also provides:
- FDA regulatory precedent by endpoint type
- Typical measurement timeframes
- Cross-indication comparability flags
"""

from loguru import logger


class EndpointNormalizer:
    """
    Normalizes clinical trial endpoint terminology to standard categories.
    """

    # Canonical endpoint names and their aliases
    ENDPOINT_ALIASES = {
        "overall_survival": {
            "canonical": "Overall Survival (OS)",
            "abbreviations": ["OS", "OAS"],
            "terms": [
                "overall survival", "all-cause mortality", "all cause mortality",
                "time to death", "survival time", "survival duration",
                "time from randomization to death", "death from any cause"
            ],
            "typical_timeframes": ["12 months", "24 months", "36 months", "5 years"],
            "fda_precedent": "Gold standard endpoint; generally accepted for full approval",
            "regulatory_notes": "Required for oncology indication claims; median OS typically reported",
            "indication_types": ["oncology", "cardiovascular", "rare disease"]
        },
        "progression_free_survival": {
            "canonical": "Progression-Free Survival (PFS)",
            "abbreviations": ["PFS", "PFI", "TTP", "TTF"],
            "terms": [
                "progression-free survival", "progression free survival",
                "time to progression", "time to tumor progression",
                "disease-free interval", "relapse-free survival",
                "event-free survival", "failure-free survival",
                "time to treatment failure", "time to disease progression"
            ],
            "typical_timeframes": ["6 months", "12 months", "18 months", "24 months"],
            "fda_precedent": "Accepted for accelerated approval in oncology; full approval requires OS benefit or strong clinical meaningfulness",
            "regulatory_notes": "FDA guidance available; RECIST criteria standard for solid tumors",
            "indication_types": ["oncology"]
        },
        "objective_response_rate": {
            "canonical": "Objective Response Rate (ORR)",
            "abbreviations": ["ORR", "RR", "TRR", "CR+PR"],
            "terms": [
                "objective response rate", "overall response rate", "tumor response rate",
                "response rate", "complete and partial response", "CR+PR",
                "clinical response rate", "confirmed response rate",
                "best overall response rate", "BORR"
            ],
            "typical_timeframes": ["at 8 weeks", "at 12 weeks", "at 6 months"],
            "fda_precedent": "Accepted for accelerated approval in oncology single-arm trials; requires durability data",
            "regulatory_notes": "RECIST v1.1 standard; requires IRC confirmation for registration trials",
            "indication_types": ["oncology"]
        },
        "complete_response_rate": {
            "canonical": "Complete Response Rate (CRR)",
            "abbreviations": ["CRR", "CR rate"],
            "terms": [
                "complete response rate", "complete remission rate",
                "proportion achieving complete response"
            ],
            "fda_precedent": "Accepted endpoint in hematology (lymphoma, leukemia)",
            "regulatory_notes": "Often used with MRD negativity in hematologic malignancies",
            "indication_types": ["hematology", "oncology"]
        },
        "disease_free_survival": {
            "canonical": "Disease-Free Survival (DFS)",
            "abbreviations": ["DFS", "RFS", "EFS"],
            "terms": [
                "disease-free survival", "recurrence-free survival",
                "relapse-free survival", "event-free survival",
                "invasive disease-free survival", "iDFS"
            ],
            "typical_timeframes": ["3 years", "5 years", "10 years"],
            "fda_precedent": "Accepted for adjuvant settings in oncology",
            "regulatory_notes": "FDA considers DFS reasonably likely to predict OS in adjuvant settings",
            "indication_types": ["oncology_adjuvant"]
        },
        "pathological_complete_response": {
            "canonical": "Pathological Complete Response (pCR)",
            "abbreviations": ["pCR", "ypCR"],
            "terms": [
                "pathological complete response", "pathologic complete response",
                "pathological complete remission", "complete pathological response"
            ],
            "fda_precedent": "Accepted for accelerated approval in neoadjuvant breast cancer; confirmatory EFS required",
            "regulatory_notes": "FDA guidance: pCR defined as ypT0/Tis ypN0",
            "indication_types": ["breast_cancer_neoadjuvant"]
        },
        "glycemic_control": {
            "canonical": "Glycemic Control (HbA1c)",
            "abbreviations": ["HbA1c", "A1C", "A1c"],
            "terms": [
                "hba1c", "glycated hemoglobin", "hemoglobin a1c",
                "glycemic control", "blood glucose control",
                "change in hba1c", "reduction in hba1c"
            ],
            "typical_timeframes": ["12 weeks", "24 weeks", "52 weeks"],
            "fda_precedent": "Standard surrogate for T2DM; CVOT required for new agents",
            "regulatory_notes": "FDA requires cardiovascular outcomes trial for most new T2DM drugs",
            "indication_types": ["diabetes", "metabolic"]
        },
        "cardiovascular_composite": {
            "canonical": "Major Adverse Cardiovascular Events (MACE)",
            "abbreviations": ["MACE", "3P-MACE", "4P-MACE", "CVOT"],
            "terms": [
                "major adverse cardiovascular events", "MACE", "cardiovascular death",
                "non-fatal MI", "non-fatal stroke", "cardiovascular composite",
                "time to first MACE", "cardiovascular outcomes"
            ],
            "typical_timeframes": ["2 years", "3 years", "4 years", "5 years"],
            "fda_precedent": "Required for most cardiovascular and diabetes drugs",
            "regulatory_notes": "FDA 2008 guidance requires ruling out 1.8x CV risk increase",
            "indication_types": ["cardiovascular", "diabetes"]
        },
        "patient_reported_outcome": {
            "canonical": "Patient-Reported Outcome (PRO)",
            "abbreviations": ["PRO", "QoL", "HRQOL", "PRO-CTCAE"],
            "terms": [
                "patient-reported outcome", "quality of life", "health-related quality of life",
                "symptom burden", "pain score", "fatigue score", "PROMIS",
                "EQ-5D", "SF-36", "FACT-G", "PRO measure"
            ],
            "typical_timeframes": ["12 weeks", "24 weeks", "52 weeks"],
            "fda_precedent": "Accepted as primary endpoint in supportive care; secondary in most indications",
            "regulatory_notes": "FDA PRO Guidance 2009; requires validated instrument and minimal clinically important difference",
            "indication_types": ["all"]
        },
        "pharmacokinetics": {
            "canonical": "Pharmacokinetics (PK)",
            "abbreviations": ["PK", "PD", "PK/PD"],
            "terms": [
                "pharmacokinetics", "area under the curve", "auc", "cmax",
                "tmax", "half-life", "clearance", "volume of distribution",
                "bioavailability", "pk parameter"
            ],
            "fda_precedent": "Standard Phase 1 endpoint; not accepted for efficacy claims",
            "regulatory_notes": "Often co-primary with safety in Phase 1",
            "indication_types": ["all_phase1"]
        },
        "safety_tolerability": {
            "canonical": "Safety and Tolerability",
            "abbreviations": ["MTD", "RP2D", "DLT", "MAD"],
            "terms": [
                "safety", "tolerability", "adverse events", "dose-limiting toxicity",
                "maximum tolerated dose", "recommended phase 2 dose",
                "incidence of adverse events", "treatment-emergent adverse events"
            ],
            "fda_precedent": "Standard Phase 1 endpoint; acceptable co-primary in dose-finding studies",
            "regulatory_notes": "CTCAE grading standard; serious adverse events require expedited reporting",
            "indication_types": ["all_phase1"]
        }
    }

    def normalize(self, endpoint_text: str) -> dict:
        """
        Normalize an endpoint description to a standard category.

        Returns dict with:
        - normalized_type: the standard endpoint category
        - canonical_name: the canonical display name
        - confidence: high/medium/low
        - fda_precedent: regulatory acceptability note
        """
        if not endpoint_text:
            return {
                "normalized_type": "unknown",
                "canonical_name": "Unknown",
                "confidence": "low",
                "fda_precedent": None
            }

        endpoint_lower = endpoint_text.lower()

        best_match = None
        best_score = 0

        for endpoint_type, info in self.ENDPOINT_ALIASES.items():
            score = 0

            # Check abbreviations (high weight)
            for abbr in info.get('abbreviations', []):
                if abbr.lower() in endpoint_lower:
                    score += 10

            # Check term matches (medium weight)
            for term in info.get('terms', []):
                if term in endpoint_lower:
                    score += 5

            if score > best_score:
                best_score = score
                best_match = endpoint_type

        if best_score >= 10:
            confidence = "high"
        elif best_score >= 5:
            confidence = "medium"
        else:
            # Fallback to classification
            best_match = self._fallback_classify(endpoint_lower)
            confidence = "low"

        if best_match and best_match in self.ENDPOINT_ALIASES:
            info = self.ENDPOINT_ALIASES[best_match]
            return {
                "normalized_type": best_match,
                "canonical_name": info['canonical'],
                "confidence": confidence,
                "fda_precedent": info.get('fda_precedent'),
                "regulatory_notes": info.get('regulatory_notes'),
                "typical_timeframes": info.get('typical_timeframes', []),
                "indication_types": info.get('indication_types', [])
            }

        return {
            "normalized_type": "composite_other",
            "canonical_name": endpoint_text[:100],
            "confidence": "low",
            "fda_precedent": None
        }

    def _fallback_classify(self, endpoint_lower: str) -> str:
        """Fallback classification using simple keyword matching."""
        if any(t in endpoint_lower for t in ['surviv', 'death', 'mortalit']):
            if 'progression' in endpoint_lower or 'disease-free' in endpoint_lower:
                return 'progression_free_survival'
            return 'overall_survival'
        if 'response' in endpoint_lower:
            return 'objective_response_rate'
        if any(t in endpoint_lower for t in ['hba1c', 'glucose', 'glycem']):
            return 'glycemic_control'
        if any(t in endpoint_lower for t in ['safety', 'adverse', 'tolerab']):
            return 'safety_tolerability'
        if any(t in endpoint_lower for t in ['quality of life', 'qol', 'symptom']):
            return 'patient_reported_outcome'
        if any(t in endpoint_lower for t in ['cardiac', 'cardiovascular', 'mace']):
            return 'cardiovascular_composite'
        return 'composite_other'

    def get_fda_precedent(self, endpoint_type: str, indication: str = None) -> str:
        """
        Get FDA regulatory precedent for a given endpoint type.
        Optionally considers indication-specific context.
        """
        if endpoint_type in self.ENDPOINT_ALIASES:
            info = self.ENDPOINT_ALIASES[endpoint_type]

            # Check indication-specific notes
            indication_types = info.get('indication_types', [])
            precedent = info.get('fda_precedent', 'No specific FDA precedent found')

            if indication and not any(
                ind in (indication or '').lower()
                for ind in indication_types
            ) and 'all' not in indication_types:
                precedent += f" Note: This endpoint is primarily validated in {', '.join(indication_types)} — verify appropriateness for your indication."

            return precedent

        return "Endpoint type not in standard database — consult regulatory affairs team"

    def suggest_timeframe(self, endpoint_type: str, indication: str = None) -> list:
        """Suggest appropriate measurement timeframes for an endpoint."""
        if endpoint_type in self.ENDPOINT_ALIASES:
            return self.ENDPOINT_ALIASES[endpoint_type].get('typical_timeframes', [])
        return []

    def batch_normalize(self, endpoints: list) -> list:
        """Normalize a batch of endpoint strings."""
        return [self.normalize(ep) for ep in endpoints]
