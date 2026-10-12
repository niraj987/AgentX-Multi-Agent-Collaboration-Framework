"""In-memory implementation of MemoryStore with retry capability."""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from agentx.memory.base import MemoryStore
from agentx.models.schemas import MemoryRecord

logger = logging.getLogger(__name__)


class MemoryStoreUnavailableError(Exception):
    """Raised when memory store cannot be accessed after retries."""

    pass


class InMemoryStore(MemoryStore):
    """In-memory storage for agent memory and context persistence."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 0.1):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Key: agent_id -> list of MemoryRecord
        self._records: Dict[str, List[MemoryRecord]] = {}
        # Fault injection flag for testing resilience
        self.is_available: bool = True

    def save_context(
        self,
        agent_id: str,
        context: Dict[str, Any],
        conversation_id: Optional[str] = None,
    ) -> MemoryRecord:
        """Stores agent context. Retries on transient failure, raising error to orchestrator if persistent."""
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if not self.is_available:
                    raise ConnectionError("Memory store backend is unavailable")

                conv_id = conversation_id or "default"
                record = MemoryRecord(
                    agent_id=agent_id,
                    conversation_id=conv_id,
                    data=dict(context),
                    timestamp=datetime.now(timezone.utc),
                )
                if agent_id not in self._records:
                    self._records[agent_id] = []
                self._records[agent_id].append(record)
                logger.debug("Saved context for agent %s in conversation %s", agent_id, conv_id)
                return record
            except Exception as e:
                last_error = e
                logger.warning(
                    "Memory store save_context attempt %d failed: %s. Retrying...",
                    attempt,
                    e,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        logger.error("Memory store unavailable after %d attempts: %s", self.max_retries, last_error)
        raise MemoryStoreUnavailableError(
            f"Failed to save context after {self.max_retries} attempts: {last_error}"
        )

    def load_context(
        self, agent_id: str, conversation_id: Optional[str] = None
    ) -> List[MemoryRecord]:
        """Loads context for an agent, optionally filtered by conversation_id."""
        if not self.is_available:
            raise MemoryStoreUnavailableError("Memory store backend is unavailable")

        records = self._records.get(agent_id, [])
        if conversation_id:
            return [r for r in records if r.conversation_id == conversation_id]
        return list(records)

    def clear(self) -> None:
        """Clears all stored records."""
        self._records.clear()
