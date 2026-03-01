"""
Eligibility Criteria Extraction and Analysis.

Uses NLP to:
1. Parse unstructured eligibility text into structured criteria items
2. Classify criteria types (age, biomarker, performance status, comorbidity, etc.)
3. Identify criteria that commonly cause recruitment challenges
4. Extract age ranges, gender requirements, and washout periods

This enables quantitative analysis of eligibility restrictiveness —
a key predictor of recruitment success.
"""

import re
from typing import Optional
from loguru import logger


class EligibilityProcessor:
    """
    Processes and analyzes clinical trial eligibility criteria.
    """

    # Criteria type classification patterns
    CRITERIA_TYPES = {
        "age": [
            r'\b(?:age|years old|years of age)\b',
            r'\b(?:adult|pediatric|geriatric|elderly)\b',
            r'\b\d{1,2}\s*(?:to|-)\s*\d{2,3}\s*years?\b'
        ],
        "performance_status": [
            r'\b(?:ECOG|Karnofsky|WHO performance|PS)\s*(?:score|status)?\s*[0-4]',
            r'\b(?:performance status|functional status)\b'
        ],
        "biomarker": [
            r'\b(?:HER2|PD-L1|EGFR|ALK|ROS1|BRAF|KRAS|MSI|TMB|ctDNA)\b',
            r'\b(?:biomarker|mutation|expression|amplification|fusion)\b',
            r'\b(?:positive|negative|high|low)\s+(?:expression|status)\b'
        ],
        "prior_treatment": [
            r'\b(?:prior|previous|prior line|treatment-naive|treatment-experienced)\b',
            r'\b(?:chemotherapy|immunotherapy|targeted therapy|radiation)\b',
            r'\b(?:washout|last dose|prior treatment)\b'
        ],
        "comorbidity": [
            r'\b(?:diabetes|hypertension|cardiac|hepatic|renal|liver|kidney)\b',
            r'\b(?:autoimmune|HIV|hepatitis|tuberculosis)\b',
            r'\b(?:history of|active|concurrent)\b'
        ],
        "lab_values": [
            r'\b(?:hemoglobin|creatinine|ALT|AST|bilirubin|platelets|ANC|WBC)\b',
            r'\b(?:eGFR|CrCl|INR|PT|aPTT)\b',
            r'\b(?:adequate|normal|within normal limits)\b'
        ],
        "organ_function": [
            r'\b(?:adequate organ function|adequate bone marrow|adequate hepatic)\b',
            r'\b(?:adequate renal function|adequate cardiac function)\b'
        ],
        "brain_metastases": [
            r'\b(?:brain metastases|CNS metastases|leptomeningeal)\b',
            r'\b(?:brain involvement|cerebral)\b'
        ],
        "pregnancy": [
            r'\b(?:pregnant|pregnancy|breastfeeding|lactating|contraception)\b',
            r'\b(?:women of childbearing|WOCBP|reproductive potential)\b'
        ],
        "consent": [
            r'\b(?:informed consent|willing to sign|ability to comply)\b'
        ]
    }

    # Criteria that most commonly cause recruitment challenges
    HIGH_RISK_CRITERIA = {
        "performance_status": {
            "risk_level": "high",
            "reason": "Restricting ECOG PS to 0-1 excludes ~30-40% of oncology patients",
            "mitigation": "Consider widening to PS 0-2 if scientifically justified"
        },
        "biomarker": {
            "risk_level": "high",
            "reason": "Biomarker requirements can reduce eligible population by 50-90%",
            "mitigation": "Ensure biomarker prevalence justifies sample size; consider all-comers with biomarker stratification"
        },
        "prior_treatment": {
            "risk_level": "medium",
            "reason": "Requiring specific prior treatment lines can narrow eligible pool significantly",
            "mitigation": "Validate prior treatment requirement against real-world treatment patterns"
        },
        "brain_metastases": {
            "risk_level": "medium",
            "reason": "Excluding brain metastases in lung cancer trials excludes 20-40% of patients",
            "mitigation": "Consider allowing stable, treated brain metastases"
        },
        "lab_values": {
            "risk_level": "low",
            "reason": "Strict lab cutoffs may exclude patients with minor abnormalities",
            "mitigation": "Use standard lab cutoffs unless stricter thresholds are safety-required"
        }
    }

    def classify_criterion(self, criterion_text: str) -> str:
        """
        Classify a single eligibility criterion into a type category.
        Returns the most specific matching type, or 'other'.
        """
        criterion_lower = criterion_text.lower()

        for criterion_type, patterns in self.CRITERIA_TYPES.items():
            for pattern in patterns:
                if re.search(pattern, criterion_lower, re.IGNORECASE):
                    return criterion_type

        return "other"

    def extract_age_range(self, criteria_text: str) -> tuple:
        """
        Extract minimum and maximum age from eligibility criteria text.
        Returns (min_age, max_age) or (None, None) if not found.
        """
        min_age, max_age = None, None

        # Patterns for minimum age
        min_patterns = [
            r'(?:age|aged?)\s*(?:≥|>=|>|at least|minimum)\s*(\d+)',
            r'(\d+)\s*years?\s*(?:of age|old)?\s*(?:or older|and older|or above)',
            r'(?:minimum|min)(?:imum)?\s*age[:\s]+(\d+)',
            r'\b(\d{2})\s*(?:to|-)\s*\d{2,3}\s*years?'  # Range pattern, capture min
        ]

        # Patterns for maximum age
        max_patterns = [
            r'(?:age|aged?)\s*(?:≤|<=|<|no more than|maximum|up to)\s*(\d+)',
            r'(\d+)\s*years?\s*(?:of age|old)?\s*(?:or younger|or below|or less)',
            r'(?:maximum|max)(?:imum)?\s*age[:\s]+(\d+)',
            r'\b\d{2}\s*(?:to|-)\s*(\d{2,3})\s*years?'  # Range pattern, capture max
        ]

        criteria_lower = criteria_text.lower() if criteria_text else ""

        for pattern in min_patterns:
            match = re.search(pattern, criteria_lower)
            if match:
                try:
                    min_age = int(match.group(1))
                    break
                except (ValueError, IndexError):
                    pass

        for pattern in max_patterns:
            match = re.search(pattern, criteria_lower)
            if match:
                try:
                    max_age = int(match.group(1))
                    break
                except (ValueError, IndexError):
                    pass

        return (min_age, max_age)

    def score_restrictiveness(self, inclusion_criteria: list, exclusion_criteria: list) -> dict:
        """
        Score the overall restrictiveness of eligibility criteria.
        Higher scores indicate more restrictive criteria and higher recruitment risk.

        Returns a dict with:
        - overall_score: 0-100 (higher = more restrictive)
        - risk_factors: list of identified high-risk criteria
        - recommendations: list of suggested modifications
        """
        risk_score = 0
        risk_factors = []
        recommendations = []

        # Analyze all criteria
        all_criteria = inclusion_criteria + exclusion_criteria

        # Count criteria types
        type_counts = {}
        for criterion in all_criteria:
            ctype = self.classify_criterion(criterion)
            type_counts[ctype] = type_counts.get(ctype, 0) + 1

        # Score based on presence of high-risk criteria types
        for ctype, info in self.HIGH_RISK_CRITERIA.items():
            if type_counts.get(ctype, 0) > 0:
                if info['risk_level'] == 'high':
                    risk_score += 25
                elif info['risk_level'] == 'medium':
                    risk_score += 15
                else:
                    risk_score += 5

                risk_factors.append({
                    "type": ctype,
                    "count": type_counts[ctype],
                    "risk_level": info['risk_level'],
                    "reason": info['reason']
                })
                recommendations.append(info['mitigation'])

        # Penalize for excessive number of exclusion criteria
        excl_count = len(exclusion_criteria)
        if excl_count > 20:
            risk_score += 20
            risk_factors.append({
                "type": "excessive_exclusions",
                "count": excl_count,
                "risk_level": "high",
                "reason": f"{excl_count} exclusion criteria — very restrictive"
            })
            recommendations.append("Review and consolidate exclusion criteria — >20 is unusually restrictive")
        elif excl_count > 15:
            risk_score += 10
            risk_factors.append({
                "type": "many_exclusions",
                "count": excl_count,
                "risk_level": "medium",
                "reason": f"{excl_count} exclusion criteria — moderately restrictive"
            })

        return {
            "overall_score": min(100, risk_score),
            "risk_level": "high" if risk_score >= 50 else "medium" if risk_score >= 25 else "low",
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "criteria_type_distribution": type_counts
        }

    def compare_to_benchmark(
        self,
        user_criteria: list,
        benchmark_criteria_list: list
    ) -> dict:
        """
        Compare user's eligibility criteria against benchmark criteria from
        historical trials. Identifies criteria that are more/less restrictive
        than typical practice.

        benchmark_criteria_list: list of criteria lists from comparable trials
        """
        if not benchmark_criteria_list:
            return {"error": "No benchmark criteria provided"}

        # Build frequency map of criteria types across benchmarks
        benchmark_type_freq = {}
        for criteria_list in benchmark_criteria_list:
            types_in_this_trial = set()
            for criterion in criteria_list:
                ctype = self.classify_criterion(criterion)
                types_in_this_trial.add(ctype)

            for ctype in types_in_this_trial:
                benchmark_type_freq[ctype] = benchmark_type_freq.get(ctype, 0) + 1

        n_benchmarks = len(benchmark_criteria_list)

        # Get user criteria types
        user_types = set()
        for criterion in user_criteria:
            ctype = self.classify_criterion(criterion)
            user_types.add(ctype)

        # Compare
        comparison = []
        for ctype in set(list(user_types) + list(benchmark_type_freq.keys())):
            freq_in_benchmarks = benchmark_type_freq.get(ctype, 0) / n_benchmarks
            in_user = ctype in user_types

            if in_user and freq_in_benchmarks < 0.3:
                status = "UNUSUAL — less than 30% of comparable trials include this"
                flag = "review"
            elif not in_user and freq_in_benchmarks > 0.7:
                status = "MISSING — over 70% of comparable trials include this"
                flag = "consider_adding"
            elif in_user and freq_in_benchmarks >= 0.7:
                status = "STANDARD — consistent with most comparable trials"
                flag = "ok"
            else:
                status = "VARIABLE — present in some comparable trials"
                flag = "optional"

            comparison.append({
                "criteria_type": ctype,
                "in_user_protocol": in_user,
                "benchmark_frequency": round(freq_in_benchmarks, 2),
                "status": status,
                "flag": flag
            })

        return {
            "comparison": sorted(comparison, key=lambda x: x['benchmark_frequency'], reverse=True),
            "benchmark_trial_count": n_benchmarks
        }
