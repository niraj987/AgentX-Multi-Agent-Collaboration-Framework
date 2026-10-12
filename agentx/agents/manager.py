"""Agent Manager for registering and retrieving specialized agents."""

import logging
from typing import Dict, List, Optional
from agentx.agents.base import LegalAgent
from agentx.memory.base import MemoryStore

logger = logging.getLogger(__name__)


class AgentManager:
    """Registers, configures, and manages agents in the framework.

    Defined in LLD Section 2:
    - register_agent()
    - get_agent()
    """

    def __init__(self, memory_store: Optional[MemoryStore] = None):
        self._agents: Dict[str, LegalAgent] = {}
        self.memory_store = memory_store

    def register_agent(self, agent: LegalAgent) -> None:
        """Registers an agent instance with the manager."""
        if not isinstance(agent, LegalAgent):
            raise TypeError(f"Agent must be an instance of LegalAgent, got {type(agent)}")
        self._agents[agent.agent_id] = agent
        logger.info("Registered agent: %s (%s)", agent.agent_id, agent.name)

    def get_agent(self, agent_id: str) -> Optional[LegalAgent]:
        """Retrieves a registered agent by ID."""
        agent = self._agents.get(agent_id)
        if not agent:
            logger.warning("Agent '%s' not found in registry", agent_id)
        return agent

    def list_agents(self) -> List[str]:
        """Returns list of registered agent IDs."""
        return list(self._agents.keys())

    @classmethod
    def create_default(cls, memory_store: Optional[MemoryStore] = None) -> "AgentManager":
        """Factory method to instantiate an AgentManager with standard legal agents."""
        from agentx.agents.specialized import (
            ComplianceReviewAgent,
            ContractAnalysisAgent,
            LegalResearchAgent,
        )

        manager = cls(memory_store=memory_store)
        manager.register_agent(ContractAnalysisAgent(memory=memory_store))
        manager.register_agent(LegalResearchAgent(memory=memory_store))
        manager.register_agent(ComplianceReviewAgent(memory=memory_store))
        return manager
