"""
Core synthesis engine that orchestrates retrieval and LLM generation.
"""

import re
import anthropic
from loguru import logger

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from retrieval.query_router import QueryRouter, QueryIntent
from retrieval.retrieval_engine import RetrievalEngine
from synthesis.prompts import (
    SAMPLE_SIZE_PROMPT, ENDPOINT_SELECTION_PROMPT, ELIGIBILITY_OPTIMIZATION_PROMPT,
    SITE_SELECTION_PROMPT, DROPOUT_ANALYSIS_PROMPT, FAILURE_ANALYSIS_PROMPT,
    COMPREHENSIVE_PROTOCOL_REVIEW_PROMPT
)
from config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE


class ProtocolAnalyzer:

    def __init__(self, vector_store):
        self.router = QueryRouter()
        self.retrieval_engine = RetrievalEngine(vector_store)
        self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def analyze(self, query: str, protocol_context: dict = None) -> dict:
        """
        Main analysis entry point.

        query: User's question or protocol description
        protocol_context: Optional structured protocol data
                          (phase, indication, N, endpoint, criteria)

        Returns: dict with 'analysis', 'retrieved_trials', 'intent', 'metadata'
        """
        # Step 1: Classify intent
        intent = self.router.classify_intent(query)
        strategy = self.router.build_strategy(query)
        logger.info(f"Query intent: {intent.value} | Strategy: {strategy.synthesis_mode}")

        # Step 2: Retrieve relevant evidence
        retrieved = self.retrieval_engine.retrieve(
            query=query,
            strategy=strategy,
            protocol_context=protocol_context
        )
        logger.info(f"Retrieved {len(retrieved['candidates'])} candidates after reranking")

        # Step 3: Format context for LLM
        context_block = self._format_retrieved_context(retrieved['candidates'])

        # Step 4: Build the full prompt
        system_prompt = self._get_system_prompt(intent)

        user_message = f"""
RETRIEVED EVIDENCE FROM TRIALMIND DATABASE:
{context_block}

────────────────────────────────────────────────
USER QUERY:
{query}

{f"PROTOCOL UNDER REVIEW:{chr(10)}{self._format_protocol_context(protocol_context)}" if protocol_context else ""}

Please analyze the retrieved evidence and provide your assessment following the
report structure in your instructions.
"""

        # Step 5: Generate synthesis
        logger.info("Generating LLM synthesis...")
        response = self.anthropic_client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )

        analysis_text = response.content[0].text

        # Post-process: validate NCT IDs in output are from retrieved context
        analysis_text = self._validate_nct_citations(
            analysis_text, retrieved['candidates']
        )

        return {
            "analysis": analysis_text,
            "intent": intent.value,
            "retrieved_trials": retrieved['candidates'],
            "retrieval_stats": retrieved['stats'],
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens
        }

    def _get_system_prompt(self, intent: QueryIntent) -> str:
        """Map intent to the appropriate detailed system prompt."""
        intent_to_prompt = {
            QueryIntent.SAMPLE_SIZE_BENCHMARKING: SAMPLE_SIZE_PROMPT,
            QueryIntent.ENDPOINT_SELECTION: ENDPOINT_SELECTION_PROMPT,
            QueryIntent.ELIGIBILITY_OPTIMIZATION: ELIGIBILITY_OPTIMIZATION_PROMPT,
            QueryIntent.SITE_SELECTION: SITE_SELECTION_PROMPT,
            QueryIntent.DROPOUT_RETENTION: DROPOUT_ANALYSIS_PROMPT,
            QueryIntent.TRIAL_FAILURE_ANALYSIS: FAILURE_ANALYSIS_PROMPT,
            QueryIntent.PROTOCOL_REVIEW: COMPREHENSIVE_PROTOCOL_REVIEW_PROMPT,
            QueryIntent.GENERAL_BENCHMARK: SAMPLE_SIZE_PROMPT,  # Default to benchmark
        }
        return intent_to_prompt.get(intent, SAMPLE_SIZE_PROMPT)

    def _format_retrieved_context(self, candidates: list) -> str:
        """
        Format retrieved candidates into a structured context block for the LLM.
        Organizes by chunk type so the LLM can easily parse the information.
        """
        if not candidates:
            return "No highly relevant trials retrieved for this query."

        sections = []
        for i, candidate in enumerate(candidates, 1):
            meta = candidate.get('metadata', {})
            text = candidate.get('text', '')
            chunk_type = meta.get('chunk_type', 'unknown')
            nct_id = meta.get('nct_id', 'unknown')
            score = candidate.get('rerank_score', 0)

            sections.append(
                f"[Evidence {i} | NCT: {nct_id} | Type: {chunk_type} | "
                f"Relevance: {score:.2f}]\n{text}\n"
            )

        return "\n".join(sections)

    def _format_protocol_context(self, context: dict) -> str:
        """Format user's protocol into readable text."""
        if not context:
            return ""
        lines = []
        field_labels = {
            'indication': 'Indication/Condition',
            'phase': 'Trial Phase',
            'drug_name': 'Drug/Intervention',
            'design': 'Study Design',
            'planned_enrollment': 'Planned Sample Size',
            'primary_endpoint': 'Primary Endpoint',
            'inclusion_criteria': 'Inclusion Criteria',
            'exclusion_criteria': 'Exclusion Criteria',
            'countries': 'Planned Countries',
            'duration_months': 'Planned Duration (months)',
            'dropout_assumption': 'Assumed Dropout Rate'
        }
        for field, label in field_labels.items():
            if field in context and context[field]:
                val = context[field]
                if isinstance(val, list):
                    val = "; ".join(str(v) for v in val)
                lines.append(f"{label}: {val}")
        return "\n".join(lines)

    def _validate_nct_citations(self, analysis_text: str, candidates: list) -> str:
        """
        Post-processing: ensure NCT IDs cited in the output were actually retrieved.
        Appends a warning if hallucinated NCT IDs are detected.
        """
        # Get NCT IDs from retrieved context
        retrieved_ncts = set()
        for candidate in candidates:
            meta = candidate.get('metadata', {})
            if 'nct_id' in meta:
                retrieved_ncts.add(meta['nct_id'])

        # Find NCT IDs in generated text
        cited_ncts = set(re.findall(r'NCT\d{8}', analysis_text))

        # Check for uncited NCTs
        hallucinated = cited_ncts - retrieved_ncts

        if hallucinated:
            warning = (
                f"\n\n---\n**DATA QUALITY NOTE**: The following NCT IDs were cited "
                f"but not found in the retrieved evidence: {', '.join(hallucinated)}. "
                f"These citations should be independently verified on ClinicalTrials.gov."
            )
            analysis_text += warning
            logger.warning(f"Potential hallucinated NCT IDs detected: {hallucinated}")

        return analysis_text
