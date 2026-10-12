"""Agents package for AgentX."""

from agentx.agents.base import ExternalAPIError, LegalAgent
from agentx.agents.manager import AgentManager
from agentx.agents.specialized import (
    ComplianceReviewAgent,
    ContractAnalysisAgent,
    LegalResearchAgent,
)

__all__ = [
    "LegalAgent",
    "AgentManager",
    "ContractAnalysisAgent",
    "LegalResearchAgent",
    "ComplianceReviewAgent",
    "ExternalAPIError",
]
