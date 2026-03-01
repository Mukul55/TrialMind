"""
API route definitions for TrialMind.
Routes are registered in main.py.
"""

import re
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.models import QueryRequest, QueryResponse, BenchmarkResponse, HealthResponse

router = APIRouter()


def get_analyzer():
    """Dependency injection for the analyzer."""
    from api.main import analyzer
    return analyzer


def get_vector_store():
    """Dependency injection for the vector store."""
    from api.main import vector_store
    return vector_store


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint. Accepts natural language questions about trial design
    with optional structured protocol context.

    Example queries:
    - "What sample sizes are used in Phase 2 NSCLC trials with PD-L1 biomarker selection?"
    - "What primary endpoints has FDA accepted for HER2+ breast cancer?"
    - "Which countries enroll fastest for Phase 3 heart failure trials?"
    """
    _analyzer = get_analyzer()
    if not _analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")

    try:
        result = _analyzer.analyze(
            query=request.query,
            protocol_context=request.protocol.dict() if request.protocol else None
        )

        # Extract NCT IDs from analysis text
        nct_ids = list(set(re.findall(r'NCT\d{8}', result['analysis'])))

        return QueryResponse(
            analysis=result['analysis'],
            intent=result['intent'],
            trial_count_retrieved=len(result['retrieved_trials']),
            nct_ids_referenced=nct_ids,
            tokens_used=result['tokens_used']
        )

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/protocol-review")
async def protocol_review(protocol: dict):
    """
    Comprehensive protocol review endpoint.
    Accepts a full protocol description and returns a complete optimization report.
    """
    _analyzer = get_analyzer()
    if not _analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")

    query = f"""
    Please conduct a comprehensive protocol optimization review for the following trial:
    Indication: {protocol.get('indication', 'Not specified')}
    Phase: {protocol.get('phase', 'Not specified')}
    Drug: {protocol.get('drug_name', 'Not specified')}
    Planned Enrollment: {protocol.get('planned_enrollment', 'Not specified')}
    Primary Endpoint: {protocol.get('primary_endpoint', 'Not specified')}
    """

    try:
        result = _analyzer.analyze(query=query, protocol_context=protocol)
        return result
    except Exception as e:
        logger.error(f"Protocol review error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmark/{indication}", response_model=BenchmarkResponse)
async def benchmark_indication(indication: str, phase: str = None):
    """
    Quick benchmark for a specific indication.
    Returns key metrics without a detailed query.
    """
    _analyzer = get_analyzer()
    if not _analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")

    query = (
        f"Provide benchmark data for {phase or 'all phases'} clinical trials in {indication}: "
        f"typical sample sizes, enrollment rates, common endpoints, and dropout rates."
    )

    try:
        result = _analyzer.analyze(query=query)
        return BenchmarkResponse(
            indication=indication,
            phase=phase,
            benchmark=result['analysis']
        )
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint — returns collection sizes."""
    _vs = get_vector_store()
    if not _vs:
        return HealthResponse(status="initializing", collections={})

    try:
        collections = {
            name: coll.count()
            for name, coll in _vs.collections.items()
        }
        return HealthResponse(status="healthy", collections=collections)
    except Exception as e:
        return HealthResponse(status=f"error: {str(e)}", collections={})
