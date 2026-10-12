"""Tests for AgentOrchestrator."""

import time
import pytest
from agentx.agents.manager import AgentManager
from agentx.agents.specialized import ContractAnalysisAgent
from agentx.memory.in_memory import InMemoryStore
from agentx.models.schemas import TaskRequest
from agentx.orchestrator.orchestrator import AgentOrchestrator


def test_orchestrator_multi_agent_workflow():
    memory = InMemoryStore()
    manager = AgentManager.create_default(memory_store=memory)
    orchestrator = AgentOrchestrator(agent_manager=manager, memory_store=memory)

    request = TaskRequest(
        task_id="legal-task-001",
        input_data={
            "document_text": "This agreement terminates in 10 days and includes an indemnification clause.",
        },
        metadata={"task_type": "full_review"},
    )

    response = orchestrator.run_workflow(request)

    assert response.task_id == "legal-task-001"
    assert response.status in ["completed", "completed_with_warnings"]
    assert "contract_analysis" in response.output_data
    assert "legal_research" in response.output_data
    assert "compliance_review" in response.output_data

    # Check orchestrator state
    assert orchestrator.state is not None
    assert orchestrator.state.step_count == 3
    assert len(orchestrator.state.history) == 3


def test_orchestrator_single_agent_pipeline():
    memory = InMemoryStore()
    manager = AgentManager.create_default(memory_store=memory)
    orchestrator = AgentOrchestrator(agent_manager=manager, memory_store=memory)

    request = TaskRequest(
        task_id="single-agent-task",
        input_data={"document_text": "Standard agreement."},
        metadata={"pipeline": ["contract_analyst"]},
    )

    response = orchestrator.run_workflow(request)
    assert response.task_id == "single-agent-task"
    assert "contract_analysis" in response.output_data
    assert "legal_research" not in response.output_data
    assert orchestrator.state.step_count == 1


def test_orchestrator_timeout_handling():
    """Verify LLD Section 5: Workflow timeout -> Abort workflow, return partial results."""
    import asyncio
    from agentx.agents.base import LegalAgent
    from agentx.models.schemas import AgentResponse

    class SlowAgent(LegalAgent):
        def __init__(self):
            super().__init__("slow_agent", "Slow Agent", "Simulates slow execution")

        def _execute_subtask(self, context):
            time.sleep(1.0)
            return {"slow_done": True}

        def _generate_fallback(self, context, reason):
            return {}

    manager = AgentManager()
    manager.register_agent(SlowAgent())
    orchestrator = AgentOrchestrator(agent_manager=manager, timeout_seconds=0.2)

    request = TaskRequest(
        task_id="timeout-task",
        input_data={},
        metadata={"pipeline": ["slow_agent"]},
    )

    response = asyncio.run(orchestrator.arun_workflow(request, timeout_seconds=0.2))
    assert response.status == "aborted"
    assert any("timeout" in err.lower() for err in response.errors)

