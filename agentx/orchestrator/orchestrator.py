"""Agent Orchestrator leveraging LangGraph for multi-agent workflows."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import END, StateGraph

from agentx.agents.manager import AgentManager
from agentx.config import settings
from agentx.memory.base import MemoryStore
from agentx.memory.in_memory import InMemoryStore, MemoryStoreUnavailableError
from agentx.models.schemas import AgentResponse, AgentState, TaskRequest

logger = logging.getLogger(__name__)


class WorkflowGraphState(TypedDict):
    """Internal dictionary state passed through LangGraph nodes."""

    task_id: str
    agent_id: Optional[str]
    context: Dict[str, Any]
    history: List[Dict[str, Any]]
    status: str
    errors: List[str]
    planned_agents: List[str]
    current_agent_index: int
    step_count: int
    max_steps: int
    is_complete: bool


class AgentOrchestrator:
    """Manages agent workflow, task routing, and execution state using LangGraph.

    Methods/Attributes per LLD Section 2:
    - run_workflow()
    - route_task()
    - state
    """

    def __init__(
        self,
        agent_manager: Optional[AgentManager] = None,
        memory_store: Optional[MemoryStore] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.memory_store = memory_store or InMemoryStore()
        self.agent_manager = agent_manager or AgentManager.create_default(memory_store=self.memory_store)
        self.timeout_seconds = timeout_seconds or settings.WORKFLOW_TIMEOUT_SECONDS
        self._state: Optional[AgentState] = None
        self._graph = self._build_langgraph_workflow()

    @property
    def state(self) -> Optional[AgentState]:
        """Current execution state."""
        return self._state

    @state.setter
    def state(self, value: Optional[AgentState]) -> None:
        self._state = value

    def route_task(self, state: AgentState) -> Optional[str]:
        """Routes task to the next appropriate agent based on current state and context.

        Returns:
            agent_id (str) or None if workflow is complete.
        """
        if state.is_complete():
            return None

        # 1. If explicit pipeline is planned in state
        if state.planned_agents and state.current_agent_index < len(state.planned_agents):
            next_agent = state.planned_agents[state.current_agent_index]
            if self.agent_manager.get_agent(next_agent):
                return next_agent

        # 2. Dynamic content-based routing if no explicit plan or completed
        context = state.context
        registered = self.agent_manager.list_agents()

        # Contract Analysis needed?
        if "contract_analysis" not in context and "contract_analyst" in registered:
            return "contract_analyst"

        # Legal Research needed?
        if "legal_research" not in context and "legal_researcher" in registered:
            return "legal_researcher"

        # Compliance Review needed?
        if "compliance_review" not in context and "compliance_officer" in registered:
            return "compliance_officer"

        # No more agents needed
        return None

    def initialize_state(self, task_request: TaskRequest) -> AgentState:
        """Initializes AgentState from incoming TaskRequest."""
        # Check if pipeline was specified in metadata or input_data
        planned_agents = task_request.metadata.get("pipeline") or task_request.input_data.get("pipeline")
        if not planned_agents:
            # Default collaborative multi-agent pipeline for legal workflow
            task_type = task_request.metadata.get("task_type", "full_review")
            if task_type == "contract_only":
                planned_agents = ["contract_analyst"]
            elif task_type == "research_only":
                planned_agents = ["legal_researcher"]
            elif task_type == "compliance_only":
                planned_agents = ["compliance_officer"]
            else:
                planned_agents = ["contract_analyst", "legal_researcher", "compliance_officer"]

        initial_context = {
            "task_id": task_request.task_id,
            **task_request.input_data,
            "metadata": task_request.metadata,
        }

        self._state = AgentState(
            agent_id=planned_agents[0] if planned_agents else None,
            context=initial_context,
            history=[],
            status="running",
            task_id=task_request.task_id,
            step_count=0,
            max_steps=len(planned_agents) + 2,
            planned_agents=list(planned_agents),
            current_agent_index=0,
            errors=[],
        )
        return self._state

    def _build_langgraph_workflow(self):
        """Constructs and compiles the LangGraph StateGraph."""
        workflow = StateGraph(WorkflowGraphState)

        # Node 1: Router Node
        def router_node(state_dict: WorkflowGraphState) -> WorkflowGraphState:
            current_state = self._dict_to_state(state_dict)
            next_agent = self.route_task(current_state)
            state_dict["agent_id"] = next_agent
            state_dict["is_complete"] = next_agent is None or current_state.is_complete()
            return state_dict

        # Node 2: Agent Execution Node
        def agent_node(state_dict: WorkflowGraphState) -> WorkflowGraphState:
            agent_id = state_dict.get("agent_id")
            if not agent_id:
                state_dict["is_complete"] = True
                return state_dict

            agent = self.agent_manager.get_agent(agent_id)
            if not agent:
                err = f"Agent '{agent_id}' not found in registry"
                logger.error(err)
                state_dict["errors"].append(err)
                state_dict["current_agent_index"] += 1
                return state_dict

            # Process input with agent
            try:
                response = agent.process_input(state_dict["context"])
                current_state = self._dict_to_state(state_dict)
                current_state.update(response)

                # Persist in MemoryStore per LLD Pseudocode:
                # MemoryStore.save_context(agent.id, state.context)
                if self.memory_store:
                    try:
                        self.memory_store.save_context(
                            agent_id=agent.agent_id,
                            context=current_state.context,
                            conversation_id=state_dict["task_id"],
                        )
                    except MemoryStoreUnavailableError as mem_err:
                        logger.error("Memory store error: %s", mem_err)
                        current_state.errors.append(f"Memory store error: {mem_err}")

                updated_dict = self._state_to_dict(current_state)
                updated_dict["agent_id"] = agent_id
                return updated_dict
            except Exception as e:
                logger.error("Unexpected error executing agent %s: %s", agent_id, e)
                state_dict["errors"].append(f"Agent execution error: {e}")
                state_dict["current_agent_index"] += 1
                return state_dict

        # Node 3: Finalizer Node
        def finalizer_node(state_dict: WorkflowGraphState) -> WorkflowGraphState:
            if not state_dict.get("errors"):
                state_dict["status"] = "completed"
            elif any("error" in err.lower() for err in state_dict.get("errors", [])):
                state_dict["status"] = "completed_with_warnings"
            state_dict["is_complete"] = True
            return state_dict

        workflow.add_node("router", router_node)
        workflow.add_node("executor", agent_node)
        workflow.add_node("finalizer", finalizer_node)

        workflow.set_entry_point("router")

        def route_condition(state_dict: WorkflowGraphState) -> str:
            if state_dict.get("is_complete") or not state_dict.get("agent_id"):
                return "finalizer"
            return "executor"

        workflow.add_conditional_edges(
            "router",
            route_condition,
            {"executor": "executor", "finalizer": "finalizer"},
        )
        workflow.add_edge("executor", "router")
        workflow.add_edge("finalizer", END)

        return workflow.compile()

    def _state_to_dict(self, state: AgentState) -> WorkflowGraphState:
        return {
            "task_id": state.task_id or "default",
            "agent_id": state.agent_id,
            "context": dict(state.context),
            "history": list(state.history),
            "status": state.status,
            "errors": list(state.errors),
            "planned_agents": list(state.planned_agents),
            "current_agent_index": state.current_agent_index,
            "step_count": state.step_count,
            "max_steps": state.max_steps,
            "is_complete": state.is_complete(),
        }

    def _dict_to_state(self, d: WorkflowGraphState) -> AgentState:
        return AgentState(
            task_id=d["task_id"],
            agent_id=d["agent_id"],
            context=d["context"],
            history=d["history"],
            status=d["status"],
            errors=d["errors"],
            planned_agents=d["planned_agents"],
            current_agent_index=d["current_agent_index"],
            step_count=d["step_count"],
            max_steps=d["max_steps"],
        )

    async def arun_workflow(
        self, task_request: TaskRequest, timeout_seconds: Optional[float] = None
    ) -> AgentResponse:
        """Asynchronous execution of the agent workflow with timeout enforcement.

        Handles:
        - Workflow timeout: Abort workflow, return partial results
        """
        timeout = timeout_seconds or self.timeout_seconds
        state = self.initialize_state(task_request)

        async def _execute():
            initial_graph_state = self._state_to_dict(state)
            final_graph_state = await asyncio.to_thread(self._graph.invoke, initial_graph_state)
            self._state = self._dict_to_state(final_graph_state)
            return self._state.final_output()

        try:
            return await asyncio.wait_for(_execute(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Workflow timeout for task '%s' after %s seconds. Returning partial results.",
                task_request.task_id,
                timeout,
            )
            state.status = "aborted"
            state.errors.append(f"Workflow timeout after {timeout} seconds")
            self._state = state
            return AgentResponse(
                task_id=task_request.task_id,
                output_data=state.context,
                status="aborted",
                errors=state.errors,
            )
        except Exception as e:
            logger.error("Workflow failed for task '%s': %s", task_request.task_id, e)
            state.status = "failed"
            state.errors.append(str(e))
            self._state = state
            return AgentResponse(
                task_id=task_request.task_id,
                output_data=state.context,
                status="failed",
                errors=state.errors,
            )

    def run_workflow(
        self, task_request: TaskRequest, timeout_seconds: Optional[float] = None
    ) -> AgentResponse:
        """Synchronous execution of the agent orchestration flow.

        Follows Section 4 Pseudocode:
        def run_workflow(task_request):
            state = initialize_state(task_request)
            while not state.is_complete():
                agent = select_next_agent(state)
                response = agent.process_input(state.context)
                state.update(response)
                MemoryStore.save_context(agent.id, state.context)
            return state.final_output()
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self.arun_workflow(task_request, timeout_seconds)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.arun_workflow(task_request, timeout_seconds))
        except RuntimeError:
            return asyncio.run(self.arun_workflow(task_request, timeout_seconds))
