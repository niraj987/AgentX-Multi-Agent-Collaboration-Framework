"""Memory Store interface for AgentX."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from agentx.models.schemas import MemoryRecord


class MemoryStore(ABC):
    """Abstract Base Class for agent memory storage."""

    @abstractmethod
    def save_context(
        self,
        agent_id: str,
        context: Dict[str, Any],
        conversation_id: Optional[str] = None,
    ) -> MemoryRecord:
        """Stores agent context. Retries on transient failure."""
        pass

    @abstractmethod
    def load_context(
        self, agent_id: str, conversation_id: Optional[str] = None
    ) -> List[MemoryRecord]:
        """Retrieves stored agent context for given agent and conversation."""
        pass
