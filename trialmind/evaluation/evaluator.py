"""
Automated evaluation pipeline for TrialMind.

Tests whether TrialMind retrieves evidence that would correctly answer
known questions about historical trial design.

Golden set: 50 questions with known ground-truth answers from public trial data.
"""

import re
import json
import requests
from datetime import datetime
from loguru import logger


GOLDEN_TEST_SET = [
    # ── Sample Size Benchmarking ──────────────────────────────────────────────
    {
        "query": "What was the typical sample size for Phase 3 trials in non-small cell lung cancer from 2015-2022?",
        "expected_keywords": ["KEYNOTE", "PACIFIC", "IMpower", "500", "700", "1000"],
        "expected_ncts": ["NCT02142738", "NCT02125461", "NCT02366143"],
        "category": "sample_size",
        "intent": "sample_size"
    },
    {
        "query": "What is the median sample size for Phase 2 breast cancer clinical trials?",
        "expected_keywords": ["breast", "phase 2", "median", "patients", "enrollment"],
        "category": "sample_size",
        "intent": "sample_size"
    },
    {
        "query": "How many patients are typically enrolled in Phase 3 cardiovascular outcome trials?",
        "expected_keywords": ["cardiovascular", "phase 3", "patients", "MACE", "outcomes"],
        "category": "sample_size",
        "intent": "sample_size"
    },
    {
        "query": "What sample sizes are used in Phase 1/2 rare disease trials?",
        "expected_keywords": ["rare disease", "phase 1", "phase 2", "orphan", "small"],
        "category": "sample_size",
        "intent": "sample_size"
    },
    {
        "query": "What enrollment rates (patients per month) are typical for Phase 3 rheumatoid arthritis trials?",
        "expected_keywords": ["rheumatoid", "arthritis", "enrollment", "patients", "month"],
        "category": "sample_size",
        "intent": "sample_size"
    },

    # ── Endpoint Selection ────────────────────────────────────────────────────
    {
        "query": "What primary endpoints were used in FDA-approved Phase 3 PD-1 inhibitor trials?",
        "expected_keywords": ["progression-free survival", "overall survival", "PFS", "OS"],
        "category": "endpoint",
        "intent": "endpoint"
    },
    {
        "query": "What primary endpoints has FDA accepted for Phase 3 heart failure trials?",
        "expected_keywords": ["hospitalization", "cardiovascular death", "MACE", "mortality"],
        "category": "endpoint",
        "intent": "endpoint"
    },
    {
        "query": "What endpoints are used as primary in Phase 2 solid tumor oncology trials?",
        "expected_keywords": ["response rate", "ORR", "progression", "PFS", "tumor"],
        "category": "endpoint",
        "intent": "endpoint"
    },
    {
        "query": "Has pathological complete response been accepted as a primary endpoint for neoadjuvant breast cancer trials?",
        "expected_keywords": ["pathological", "complete response", "pCR", "neoadjuvant", "breast"],
        "category": "endpoint",
        "intent": "endpoint"
    },
    {
        "query": "What are typical primary endpoints in Phase 3 diabetes clinical trials?",
        "expected_keywords": ["HbA1c", "glycemic", "A1c", "glucose", "cardiovascular"],
        "category": "endpoint",
        "intent": "endpoint"
    },

    # ── Eligibility Criteria ──────────────────────────────────────────────────
    {
        "query": "What are common exclusion criteria in Phase 2 Alzheimer's disease trials?",
        "expected_keywords": ["MMSE", "CDR", "cognitive", "dementia", "brain", "MRI"],
        "category": "eligibility",
        "intent": "eligibility"
    },
    {
        "query": "What ECOG performance status requirements are used in Phase 3 lung cancer trials?",
        "expected_keywords": ["ECOG", "performance status", "PS", "0-1", "lung", "cancer"],
        "category": "eligibility",
        "intent": "eligibility"
    },
    {
        "query": "What eligibility criteria in Phase 2 oncology trials most commonly cause recruitment failure?",
        "expected_keywords": ["eligibility", "restrictive", "recruitment", "criteria", "exclusion"],
        "category": "eligibility",
        "intent": "eligibility"
    },
    {
        "query": "Are brain metastases typically excluded from Phase 3 NSCLC trials?",
        "expected_keywords": ["brain metastases", "CNS", "exclusion", "lung", "criteria"],
        "category": "eligibility",
        "intent": "eligibility"
    },
    {
        "query": "What biomarker requirements are commonly used as inclusion criteria in Phase 2 oncology trials?",
        "expected_keywords": ["biomarker", "PD-L1", "HER2", "mutation", "expression", "inclusion"],
        "category": "eligibility",
        "intent": "eligibility"
    },

    # ── Site Selection ────────────────────────────────────────────────────────
    {
        "query": "Which countries show the fastest enrollment rates for Phase 3 cardiovascular trials?",
        "expected_keywords": ["United States", "Europe", "enrollment", "rate", "cardiovascular"],
        "category": "site",
        "intent": "site"
    },
    {
        "query": "How many sites are typically used in a Phase 3 oncology trial enrolling 500 patients?",
        "expected_keywords": ["sites", "centers", "enrollment", "phase 3", "oncology"],
        "category": "site",
        "intent": "site"
    },
    {
        "query": "What are typical enrollment rates per site per month in Phase 2 clinical trials?",
        "expected_keywords": ["site", "enrollment", "patients", "month", "rate"],
        "category": "site",
        "intent": "site"
    },
    {
        "query": "How many countries are used in a typical Phase 3 immunology trial?",
        "expected_keywords": ["countries", "international", "immunology", "sites", "global"],
        "category": "site",
        "intent": "site"
    },

    # ── Dropout / Retention ───────────────────────────────────────────────────
    {
        "query": "What dropout rates are typical in 24-month Phase 3 Alzheimer's trials?",
        "expected_keywords": ["dropout", "withdrawal", "discontinuation", "%", "Alzheimer"],
        "category": "dropout",
        "intent": "dropout"
    },
    {
        "query": "What attrition rates should I expect in a Phase 3 oncology trial with monthly IV dosing?",
        "expected_keywords": ["attrition", "dropout", "intravenous", "dosing", "monthly"],
        "category": "dropout",
        "intent": "dropout"
    },
    {
        "query": "What is the typical completion rate in Phase 3 rheumatoid arthritis trials?",
        "expected_keywords": ["completion", "dropout", "withdrawal", "rheumatoid", "arthritis"],
        "category": "dropout",
        "intent": "dropout"
    },

    # ── Trial Failure Analysis ────────────────────────────────────────────────
    {
        "query": "What are the most common reasons Phase 3 oncology trials are terminated early?",
        "expected_keywords": ["terminated", "enrollment", "safety", "efficacy", "futility"],
        "category": "failure",
        "intent": "failure"
    },
    {
        "query": "Why are Phase 3 heart failure trials frequently terminated early?",
        "expected_keywords": ["enrollment", "safety", "efficacy", "futility"],
        "category": "failure",
        "intent": "failure"
    },
    {
        "query": "What percentage of Phase 3 trials fail to meet their primary endpoint?",
        "expected_keywords": ["fail", "endpoint", "primary", "phase 3", "miss"],
        "category": "failure",
        "intent": "failure"
    },
    {
        "query": "What design factors predict clinical trial failure in neurodegenerative disease?",
        "expected_keywords": ["neurodegenerative", "design", "failure", "endpoint", "enrollment"],
        "category": "failure",
        "intent": "failure"
    },
    {
        "query": "What are common reasons for clinical trial amendments in Phase 2-3 oncology trials?",
        "expected_keywords": ["amendment", "protocol", "modification", "change", "oncology"],
        "category": "failure",
        "intent": "failure"
    },

    # ── General / Cross-cutting ───────────────────────────────────────────────
    {
        "query": "How long do Phase 3 cardiovascular trials typically take from start to primary completion?",
        "expected_keywords": ["duration", "months", "years", "cardiovascular", "phase 3"],
        "category": "general",
        "intent": "general"
    },
    {
        "query": "What is the typical trial duration for a Phase 2 immuno-oncology trial?",
        "expected_keywords": ["duration", "months", "immuno-oncology", "checkpoint", "phase 2"],
        "category": "general",
        "intent": "general"
    },
    {
        "query": "What percentage of Phase 3 oncology trials achieve their planned enrollment target?",
        "expected_keywords": ["enrollment", "target", "achieved", "oncology", "planned"],
        "category": "general",
        "intent": "general"
    },
]


class TrialMindEvaluator:
    """
    Automated evaluation pipeline for TrialMind.
    Runs golden test set queries and scores results.
    """

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_url = api_base_url
        self.session = requests.Session()

    def run_single_query(self, test_case: dict) -> dict:
        """Run a single test case and score the result."""
        query = test_case['query']

        try:
            response = self.session.post(
                f"{self.api_url}/query",
                json={"query": query},
                timeout=120
            )

            if response.status_code != 200:
                return {
                    "query": query,
                    "category": test_case.get('category'),
                    "score": 0.0,
                    "error": f"HTTP {response.status_code}: {response.text[:200]}"
                }

            result = response.json()
            analysis = result['analysis']

        except Exception as e:
            return {
                "query": query,
                "category": test_case.get('category'),
                "score": 0.0,
                "error": str(e)
            }

        # Score the result
        analysis_lower = analysis.lower()

        # Keyword recall
        expected_keywords = test_case.get('expected_keywords', [])
        if expected_keywords:
            keywords_found = sum(
                1 for kw in expected_keywords
                if kw.lower() in analysis_lower
            )
            keyword_recall = keywords_found / len(expected_keywords)
        else:
            keyword_recall = 1.0

        # NCT ID recall (if specified)
        nct_recall = 0.0
        expected_ncts = test_case.get('expected_ncts', [])
        if expected_ncts:
            ncts_found = sum(1 for nct in expected_ncts if nct in analysis.upper())
            nct_recall = ncts_found / len(expected_ncts)

        # Quantitativeness check — does the response contain numbers?
        has_numbers = bool(re.search(r'\d+', analysis))
        quantitative_score = 1.0 if has_numbers else 0.5

        # NCT citation check — does response cite any NCT IDs?
        has_nct_citations = bool(re.search(r'NCT\d{8}', analysis))
        citation_score = 1.0 if has_nct_citations else 0.7

        # Overall score
        if expected_ncts:
            overall = (keyword_recall * 0.4 + nct_recall * 0.3 +
                      quantitative_score * 0.15 + citation_score * 0.15)
        else:
            overall = (keyword_recall * 0.5 + quantitative_score * 0.25 +
                      citation_score * 0.25)

        return {
            "query": query[:80] + ("..." if len(query) > 80 else ""),
            "category": test_case.get('category'),
            "keyword_recall": round(keyword_recall, 2),
            "nct_recall": round(nct_recall, 2) if expected_ncts else None,
            "quantitative_score": quantitative_score,
            "citation_score": citation_score,
            "overall_score": round(overall, 2),
            "trials_retrieved": result.get('trial_count_retrieved', 0),
            "intent_detected": result.get('intent'),
            "expected_intent": test_case.get('intent')
        }

    def run_evaluation(
        self,
        test_cases: list = None,
        verbose: bool = True
    ) -> dict:
        """
        Run the full evaluation suite.

        Returns aggregate scores and per-test results.
        """
        test_cases = test_cases or GOLDEN_TEST_SET

        logger.info(f"Running evaluation with {len(test_cases)} test cases...")
        results = []

        for i, test in enumerate(test_cases, 1):
            logger.info(f"Test {i}/{len(test_cases)}: {test['query'][:60]}...")
            result = self.run_single_query(test)
            results.append(result)

            if verbose:
                status = (
                    "✅" if result.get('overall_score', 0) > 0.6
                    else "⚠️" if result.get('overall_score', 0) > 0.3
                    else "❌"
                )
                print(
                    f"{status} [{result.get('category', '?'):12}] "
                    f"Score: {result.get('overall_score', 0):.2f} | "
                    f"Trials: {result.get('trials_retrieved', 0)} | "
                    f"Intent: {result.get('intent_detected', '?')} | "
                    f"{result.get('query', '')[:50]}"
                )

        # Aggregate scores
        valid_results = [r for r in results if 'error' not in r]
        if not valid_results:
            return {"error": "All tests failed", "results": results}

        avg_score = sum(r['overall_score'] for r in valid_results) / len(valid_results)
        avg_keyword_recall = sum(r['keyword_recall'] for r in valid_results) / len(valid_results)

        # By category
        by_category = {}
        for result in valid_results:
            cat = result.get('category', 'unknown')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(result['overall_score'])

        category_scores = {
            cat: round(sum(scores) / len(scores), 2)
            for cat, scores in by_category.items()
        }

        # Intent accuracy
        intent_correct = sum(
            1 for r in valid_results
            if r.get('intent_detected') == r.get('expected_intent')
        )
        intent_accuracy = intent_correct / len(valid_results) if valid_results else 0

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(test_cases),
            "valid_tests": len(valid_results),
            "failed_tests": len(test_cases) - len(valid_results),
            "average_score": round(avg_score, 3),
            "average_keyword_recall": round(avg_keyword_recall, 3),
            "intent_accuracy": round(intent_accuracy, 3),
            "target_score": 0.65,
            "passes_target": avg_score >= 0.65,
            "by_category": category_scores,
            "results": results
        }

        # Print summary
        print(f"\n{'='*70}")
        print(f"TRIALMIND EVALUATION RESULTS")
        print(f"{'='*70}")
        print(f"Average Score:      {avg_score:.3f} / 1.000 (Target: 0.650)")
        print(f"Keyword Recall:     {avg_keyword_recall:.3f}")
        print(f"Intent Accuracy:    {intent_accuracy:.3f}")
        print(f"Tests Passed:       {len(valid_results)}/{len(test_cases)}")
        print(f"\nBy Category:")
        for cat, score in sorted(category_scores.items()):
            bar = "█" * int(score * 20)
            print(f"  {cat:<15} {score:.2f} {bar}")
        print(f"\nResult: {'✅ PASSES' if avg_score >= 0.65 else '❌ BELOW TARGET'}")
        print(f"{'='*70}")

        return summary


def save_evaluation_results(results: dict, output_path: str = "evaluation/results.json"):
    """Save evaluation results to a JSON file."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    evaluator = TrialMindEvaluator()
    results = evaluator.run_evaluation(verbose=True)
    save_evaluation_results(results)
