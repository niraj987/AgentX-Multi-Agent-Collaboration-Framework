"""Base class for specialized legal agents."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agentx.config import settings
from agentx.memory.base import MemoryStore
from agentx.models.schemas import AgentResponse

logger = logging.getLogger(__name__)


class ExternalAPIError(Exception):
    """Exception for external OpenAI API failures."""

    pass


class LegalAgent(ABC):
    """Base class for specialized legal agents.

    Attributes:
        agent_id (str): Unique agent identifier.
        name (str): Human-readable agent name.
        description (str): Agent domain specialty description.
        memory (MemoryStore): Memory store for context persistence.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        memory: Optional[MemoryStore] = None,
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.memory = memory
        self._llm = None
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize LangChain ChatOpenAI if API key is provided."""
        if settings.OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI

                self._llm = ChatOpenAI(
                    model=settings.OPENAI_MODEL,
                    api_key=settings.OPENAI_API_KEY,
                    temperature=0.2,
                )
                logger.info("Agent '%s' initialized with OpenAI model '%s'", self.agent_id, settings.OPENAI_MODEL)
            except Exception as e:
                logger.warning(
                    "Agent '%s' failed to initialize ChatOpenAI: %s. Using rule-based fallback.",
                    self.agent_id,
                    e,
                )
                self._llm = None
        else:
            logger.info("Agent '%s' initialized in offline fallback mode (no OPENAI_API_KEY).", self.agent_id)
            self._llm = None

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(ExternalAPIError),
    )
    def _call_llm_with_backoff(self, prompt: str) -> str:
        """Invokes external LLM with exponential backoff on transient failure."""
        if self._llm is None:
            raise ExternalAPIError("No LLM client configured")
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            system_instruction = (
                f"You are an expert legal AI agent specializing in: {self.description}. "
                "Provide rigorous, professional, structured legal assessment."
            )
            response = self._llm.invoke([
                SystemMessage(content=system_instruction),
                HumanMessage(content=prompt),
            ])
            return response.content
        except Exception as e:
            logger.error("LLM call failed for agent '%s': %s", self.agent_id, e)
            raise ExternalAPIError(str(e)) from e

    def process_input(self, context: Dict[str, Any]) -> AgentResponse:
        """Process incoming task context and return an AgentResponse.

        Handles:
        - Agent processing failure: Log error, update status, return fallback
        - External API (OpenAI) failure: Exponential backoff, fallback response
        - Memory updates: Store agent output in MemoryStore
        """
        task_id = context.get("task_id", "task-default")
        logger.info("Agent '%s' processing task '%s'", self.agent_id, task_id)

        try:
            # Domain-specific subtask execution
            output_data = self._execute_subtask(context)
            status = "completed"
            errors = []
        except ExternalAPIError as e:
            logger.warning(
                "External API failure for agent '%s': %s. Returning fallback response.",
                self.agent_id,
                e,
            )
            output_data = self._generate_fallback(context, reason=f"External API failure: {e}")
            status = "fallback"
            errors = [f"External API error: {e}"]
        except Exception as e:
            logger.error(
                "Agent processing failure for agent '%s': %s. Returning fallback response.",
                self.agent_id,
                e,
                exc_info=True,
            )
            output_data = self._generate_fallback(context, reason=f"Internal agent error: {e}")
            status = "fallback"
            errors = [f"Agent processing error: {e}"]

        response = AgentResponse(
            task_id=task_id,
            output_data=output_data,
            status=status,
            errors=errors,
        )

        # Update local memory store if attached
        if self.memory:
            try:
                self.memory.save_context(
                    agent_id=self.agent_id,
                    context=output_data,
                    conversation_id=task_id,
                )
            except Exception as mem_err:
                logger.error("Failed to persist context in memory store: %s", mem_err)
                response.errors.append(f"Memory store warning: {mem_err}")

        return response

    @abstractmethod
    def _execute_subtask(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute domain-specific subtask logic."""
        pass

    @abstractmethod
    def _generate_fallback(self, context: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Provide fallback output when LLM or subtask execution fails."""
        pass
