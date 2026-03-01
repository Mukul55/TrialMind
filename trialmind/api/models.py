"""
Pydantic request/response models for the TrialMind API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ProtocolContext(BaseModel):
    """Structured protocol data submitted for review."""
    indication: Optional[str] = Field(None, description="Disease/condition being studied")
    phase: Optional[str] = Field(None, description="Trial phase (e.g., 'Phase 2')")
    drug_name: Optional[str] = Field(None, description="Drug or intervention name")
    design: Optional[str] = Field(None, description="e.g., 'randomized, double-blind, placebo-controlled'")
    planned_enrollment: Optional[int] = Field(None, description="Target sample size")
    primary_endpoint: Optional[str] = Field(None, description="Primary efficacy endpoint")
    inclusion_criteria: Optional[list] = Field(default_factory=list)
    exclusion_criteria: Optional[list] = Field(default_factory=list)
    countries: Optional[list] = Field(default_factory=list)
    duration_months: Optional[int] = Field(None, description="Planned trial duration in months")
    dropout_assumption: Optional[float] = Field(None, description="Expected dropout rate (0.0-1.0)")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question about trial design")
    protocol: Optional[ProtocolContext] = Field(
        None,
        description="Optional structured protocol for comprehensive review"
    )
    filters: Optional[dict] = Field(
        default_factory=dict,
        description="Optional metadata filters: phase, indication, year range"
    )


class QueryResponse(BaseModel):
    analysis: str
    intent: str
    trial_count_retrieved: int
    nct_ids_referenced: list
    tokens_used: int


class BenchmarkResponse(BaseModel):
    indication: str
    phase: Optional[str]
    benchmark: str


class HealthResponse(BaseModel):
    status: str
    collections: dict
