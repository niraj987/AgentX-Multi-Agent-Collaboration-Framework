"""AgentX Multi-Agent Collaboration Workflow Demonstration Script.

Demonstrates:
1. Orchestrator initializing workflow from TaskRequest
2. Dynamic agent routing via LangGraph
3. Specialized agents executing tasks (Contract Analysis, Legal Research, Compliance Review)
4. State progression & context accumulation
5. MemoryStore context persistence
6. Fallback and resilience handling
"""

import json
from agentx.agents.manager import AgentManager
from agentx.memory.in_memory import InMemoryStore
from agentx.models.schemas import TaskRequest
from agentx.orchestrator.orchestrator import AgentOrchestrator


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    print_section("AgentX - Multi-Agent Collaboration Framework Demo")

    # 1. Initialize MemoryStore and AgentManager
    memory_store = InMemoryStore()
    agent_manager = AgentManager.create_default(memory_store=memory_store)
    print(f"Registered agents: {agent_manager.list_agents()}")

    # 2. Instantiate Orchestrator
    orchestrator = AgentOrchestrator(
        agent_manager=agent_manager, memory_store=memory_store, timeout_seconds=30.0
    )

    # 3. Create Sample Legal Task Request
    task_payload = TaskRequest(
        task_id="legal-contract-review-2026",
        input_data={
            "contract_title": "Enterprise Cloud Services & Data Processing Agreement",
            "document_text": (
                "Vendor provides cloud hosting services. Vendor liability is capped at $50,000, "
                "except Customer shall indemnify Vendor without limitation for any intellectual "
                "property infringement or data claims. Either party may terminate with 30 days notice. "
                "Agreement is governed by Delaware law. Personal customer records will be processed."
            ),
            "jurisdiction": "Delaware / United States",
        },
        metadata={
            "task_type": "full_review",
            "priority": "high",
            "client_name": "Acme Global Legal Ops",
        },
    )

    print(f"\nSubmitting Task: {task_payload.task_id}")
    print(f"Metadata: {task_payload.metadata}")

    # 4. Run Multi-Agent Collaborative Workflow
    print_section("Running Collaborative Workflow via LangGraph")
    response = orchestrator.run_workflow(task_payload)

    print(f"Workflow Finished Status: {response.status}")
    print(f"Errors / Warnings: {response.errors}")

    print_section("Execution Audit History (LangGraph Trajectory)")
    for step in orchestrator.state.history:
        print(f"Step {step['step']}: Agent='{step['agent_id']}' | Status='{step['status']}' | Outputs={step['output_keys']}")

    print_section("Collaborative Multi-Agent Output Data")
    print(json.dumps(response.output_data, indent=2))

    print_section("MemoryStore Persistence Verification")
    for agent_id in agent_manager.list_agents():
        records = memory_store.load_context(agent_id, task_payload.task_id)
        print(f"Agent '{agent_id}' has {len(records)} stored memory record(s) in conversation '{task_payload.task_id}'")

    print_section("Demo Completed Successfully")


if __name__ == "__main__":
    main()
