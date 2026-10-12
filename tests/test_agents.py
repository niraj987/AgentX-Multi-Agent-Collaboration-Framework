"""Tests for LegalAgent, AgentManager, and specialized agents."""

import pytest
from agentx.agents.manager import AgentManager
from agentx.agents.specialized import (
    ComplianceReviewAgent,
    ContractAnalysisAgent,
    LegalResearchAgent,
)
from agentx.memory.in_memory import InMemoryStore


def test_agent_manager_registration():
    manager = AgentManager()
    agent = ContractAnalysisAgent()
    manager.register_agent(agent)

    assert manager.get_agent("contract_analyst") is not None
    assert "contract_analyst" in manager.list_agents()
    assert manager.get_agent("non_existent") is None


def test_contract_analysis_agent_fallback():
    memory = InMemoryStore()
    agent = ContractAnalysisAgent(memory=memory)

    context = {
        "task_id": "test-task-1",
        "document_text": "Party A agrees to indemnify Party B for all liabilities without limit. Termination on 30 days notice.",
    }
    response = agent.process_input(context)

    assert response.task_id == "test-task-1"
    assert "contract_analysis" in response.output_data
    clauses = response.output_data["contract_analysis"]["detected_clauses"]
    assert "Indemnification Clause" in clauses
    assert "Termination Clause" in clauses

    # Check memory persistence
    mem_records = memory.load_context("contract_analyst", "test-task-1")
    assert len(mem_records) == 1


def test_legal_research_agent():
    agent = LegalResearchAgent()
    context = {"task_id": "test-task-2", "jurisdiction": "New York"}
    response = agent.process_input(context)

    assert response.task_id == "test-task-2"
    assert "legal_research" in response.output_data
    assert "relevant_statutes" in response.output_data["legal_research"]


def test_compliance_review_agent():
    agent = ComplianceReviewAgent()
    context = {"task_id": "test-task-3"}
    response = agent.process_input(context)

    assert response.task_id == "test-task-3"
    assert "compliance_review" in response.output_data
    assert "compliance_score" in response.output_data["compliance_review"]
