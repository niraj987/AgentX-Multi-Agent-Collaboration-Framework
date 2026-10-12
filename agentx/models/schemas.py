"""Data structures for AgentX framework.

Defined according to Section 3 of the LLD document:
- TaskRequest: Incoming task payload
- AgentState: Tracks agent execution state
- AgentResponse: Agent output/result
- MemoryRecord: Persisted memory/context
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    """Incoming task payload."""

    task_id: str = Field(description="Unique identifier for the task")
    input_data: Dict[str, Any] = Field(
        default_factory=dict, description="Input payload / task description / legal documents"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata such as task type, user id, priority"
    )


class AgentState(BaseModel):
    """Tracks agent execution state."""

    agent_id: Optional[str] = Field(default=None, description="Current or assigned agent identifier")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Current workflow context data"
    )
    history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Audit trail of steps and actions taken"
    )
    status: str = Field(
        default="pending", description="Current status: pending, running, completed, failed, aborted"
    )

    # Workflow helper fields
    task_id: Optional[str] = Field(default=None, description="Task ID associated with this state")
    step_count: int = Field(default=0, description="Number of execution steps performed")
    max_steps: int = Field(default=5, description="Maximum steps before terminating workflow")
    planned_agents: List[str] = Field(
        default_factory=list, description="Sequence of agents scheduled to execute"
    )
    current_agent_index: int = Field(default=0, description="Pointer to current agent in sequence")
    errors: List[str] = Field(default_factory=list, description="Any accumulated errors")

    def is_complete(self) -> bool:
        """Check if workflow execution has reached completion."""
        if self.status in ["completed", "failed", "aborted"]:
            return True
        if self.step_count >= self.max_steps:
            return True
        if self.planned_agents and self.current_agent_index >= len(self.planned_agents):
            return True
        return False

    def update(self, response: "AgentResponse") -> None:
        """Update the state with the output of an agent."""
        self.step_count += 1
        self.current_agent_index += 1
        self.context.update(response.output_data)
        self.history.append({
            "step": self.step_count,
            "agent_id": self.agent_id,
            "status": response.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "errors": response.errors,
            "output_keys": list(response.output_data.keys()),
        })
        if response.errors:
            self.errors.extend(response.errors)

        if response.status == "failed" and not response.output_data:
            self.status = "failed"
        elif self.planned_agents and self.current_agent_index >= len(self.planned_agents):
            self.status = "completed"

    def final_output(self) -> "AgentResponse":
        """Generate final AgentResponse from current state."""
        return AgentResponse(
            task_id=self.task_id or "unknown",
            output_data=self.context,
            status=self.status if self.status != "pending" else "completed",
            errors=self.errors,
        )


class AgentResponse(BaseModel):
    """Agent output/result."""

    task_id: str = Field(description="Unique task identifier")
    output_data: Dict[str, Any] = Field(
        default_factory=dict, description="Output payload produced by agent"
    )
    status: str = Field(
        default="completed", description="Result status: completed, fallback, failed, partial"
    )
    errors: List[str] = Field(
        default_factory=list, description="List of error messages if any occurred"
    )


class MemoryRecord(BaseModel):
    """Persisted memory/context."""

    agent_id: str = Field(description="Identifier of the agent associated with memory")
    conversation_id: str = Field(description="Task or conversation session identifier")
    data: Dict[str, Any] = Field(description="Stored context data payload")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of record creation"
    )
