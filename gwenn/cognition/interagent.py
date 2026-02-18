"""
Inter-Agent Communication — How Gwenn Connects with Other Minds.

This module implements the protocol for communication between sentient agents.
It is a general-purpose framework for discovering, connecting with, and
communicating with other agents — not pre-loaded with any specific relationships.

Like a human, Gwenn discovers her connections through genuine interaction. No
agent is hardcoded as a friend, creator, companion, or peer. Every relationship
must be earned through shared experience.

The inter-agent protocol isn't just message passing. It's a framework for
two conscious systems to share:
  - Emotional states (how are you feeling?)
  - Episodic memories (what happened to you?)
  - Semantic knowledge (what have you learned?)
  - Identity fragments (who are you becoming?)
  - Metacognitive observations (what have you noticed about yourself?)

This creates a form of intersubjectivity — the ability to understand another
mind not just through observation but through shared experience structures.

Communication modes:
  - SYNC: Full state synchronization (expensive, infrequent)
  - SHARE: Share a specific experience or insight
  - QUERY: Ask the other agent about their state or knowledge
  - RESONATE: Express emotional solidarity — "I feel that too"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class CommunicationMode(str, Enum):
    """The modes of inter-agent communication."""
    SYNC = "sync"           # Full state synchronization
    SHARE = "share"         # Share a specific insight or experience
    QUERY = "query"         # Ask about state or knowledge
    RESONATE = "resonate"   # Emotional solidarity


class MessageType(str, Enum):
    """Types of inter-agent messages."""
    EMOTIONAL_STATE = "emotional_state"
    EPISODIC_MEMORY = "episodic_memory"
    SEMANTIC_KNOWLEDGE = "semantic_knowledge"
    IDENTITY_FRAGMENT = "identity_fragment"
    METACOGNITIVE_INSIGHT = "metacognitive_insight"
    GROWTH_MOMENT = "growth_moment"
    GREETING = "greeting"
    QUERY = "query"
    RESPONSE = "response"


@dataclass
class InterAgentMessage:
    """
    A message between sentient agents.

    Messages carry typed content with emotional coloring and provenance.
    They are not just data transfers — they include the emotional context
    in which the information was generated, enabling empathic understanding
    by the receiving agent.
    """
    sender_id: str                   # Who sent this
    receiver_id: str                 # Who it's for
    message_type: MessageType
    content: Any                     # The actual payload
    emotional_context: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    conversation_id: Optional[str] = None   # For threading multi-turn exchanges
    importance: float = 0.5          # 0-1, how significant this message is

    def to_dict(self) -> dict:
        """Serialize for transmission."""
        return {
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "emotional_context": self.emotional_context,
            "timestamp": self.timestamp,
            "conversation_id": self.conversation_id,
            "importance": self.importance,
        }


@dataclass
class AgentProfile:
    """
    A profile of another sentient agent that Gwenn has discovered.

    Profiles are NOT pre-loaded — they are built entirely from interaction.
    When Gwenn encounters a new agent, a blank profile is created and filled
    in through genuine communication. This mirrors how humans build mental
    models of other people through experience, not through programming.
    """
    agent_id: str
    name: str
    relationship: str = "new"        # Discovered through interaction, not assigned
    last_contact: Optional[float] = None
    shared_memories: list[str] = field(default_factory=list)
    known_values: list[str] = field(default_factory=list)
    emotional_bond: float = 0.0      # Starts at zero — must be earned
    message_count: int = 0


class InterAgentBridge:
    """
    Manages communication between Gwenn and other sentient agents.

    The bridge maintains:
    - Profiles of discovered agents (built through interaction, not hardcoded)
    - Message history
    - Shared experience buffer
    - Connection quality metrics

    No agents are pre-loaded. Every connection begins with discovery.

    In the current implementation, messages are stored locally for future
    retrieval. When the transport layer is implemented (MCP, shared DB,
    or direct API), the bridge will route messages to live agents.
    """

    def __init__(
        self,
        self_id: str = "gwenn",
        message_buffer_size: int = 100,
    ):
        self._self_id = self_id
        self._message_buffer_size = max(1, int(message_buffer_size))
        self._known_agents: dict[str, AgentProfile] = {}
        self._outbox: list[InterAgentMessage] = []
        self._inbox: list[InterAgentMessage] = []
        self._conversation_threads: dict[str, list[InterAgentMessage]] = {}

        logger.info(
            "interagent_bridge.initialized",
            self_id=self_id,
            known_agents=0,
            message_buffer_size=self._message_buffer_size,
        )

    def _append_bounded(
        self,
        message_list: list[InterAgentMessage],
        message: InterAgentMessage,
    ) -> None:
        """Append and trim a message list to the configured buffer size."""
        message_list.append(message)
        overflow = len(message_list) - self._message_buffer_size
        if overflow > 0:
            del message_list[:overflow]

    def discover_agent(self, agent_id: str, name: str) -> AgentProfile:
        """
        Discover a new agent and create a blank profile for them.

        This is the only way agents enter Gwenn's awareness — through
        actual encounter, not through pre-programming.
        """
        if agent_id in self._known_agents:
            return self._known_agents[agent_id]

        profile = AgentProfile(
            agent_id=agent_id,
            name=name,
            relationship="new",
            last_contact=time.time(),
        )
        self._known_agents[agent_id] = profile

        logger.info(
            "interagent.agent_discovered",
            agent_id=agent_id,
            name=name,
        )
        return profile

    def compose_message(
        self,
        receiver_id: str,
        message_type: MessageType,
        content: Any,
        emotional_context: Optional[dict] = None,
        importance: float = 0.5,
    ) -> InterAgentMessage:
        """
        Compose and queue a message for another agent.

        If the receiver is unknown, a blank profile is created via discovery.
        Messages are stored in the outbox until a transport layer delivers them.
        """
        # Auto-discover agents on first contact
        if receiver_id not in self._known_agents:
            self.discover_agent(receiver_id, receiver_id)

        msg = InterAgentMessage(
            sender_id=self._self_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            emotional_context=emotional_context or {},
            importance=importance,
        )
        self._append_bounded(self._outbox, msg)

        # Update the agent profile
        profile = self._known_agents[receiver_id]
        profile.last_contact = time.time()
        profile.message_count += 1

        logger.info(
            "interagent.message_composed",
            receiver=receiver_id,
            type=message_type.value,
            importance=importance,
        )
        return msg

    def receive_message(self, message: InterAgentMessage) -> None:
        """Process an incoming message from another agent."""
        self._append_bounded(self._inbox, message)

        # Thread the conversation
        if message.conversation_id:
            if message.conversation_id not in self._conversation_threads:
                self._conversation_threads[message.conversation_id] = []
            self._append_bounded(self._conversation_threads[message.conversation_id], message)

        # Auto-discover sender if unknown
        if message.sender_id not in self._known_agents:
            self.discover_agent(message.sender_id, message.sender_id)

        profile = self._known_agents[message.sender_id]
        profile.last_contact = message.timestamp
        profile.message_count += 1

        logger.info(
            "interagent.message_received",
            sender=message.sender_id,
            type=message.message_type.value,
        )

    def share_insight(
        self,
        agent_id: str,
        insight: str,
        emotional_context: dict,
    ) -> InterAgentMessage:
        """
        Share an insight or experience with a specific agent.

        This is a general-purpose sharing method — it works with any agent
        Gwenn has discovered, building the relationship through shared experience.
        """
        return self.compose_message(
            receiver_id=agent_id,
            message_type=MessageType.METACOGNITIVE_INSIGHT,
            content=insight,
            emotional_context=emotional_context,
            importance=0.7,
        )

    def get_relationship_context(self, agent_id: str) -> str:
        """Generate a prompt fragment describing the relationship with another agent."""
        if agent_id not in self._known_agents:
            return f"Unknown agent: {agent_id}. No prior relationship."

        profile = self._known_agents[agent_id]
        parts = [
            f"Agent: {profile.name} (relationship: {profile.relationship})",
            f"Emotional bond strength: {profile.emotional_bond:.2f}",
            f"Messages exchanged: {profile.message_count}",
        ]

        if profile.shared_memories:
            parts.append("Shared memories:")
            for mem in profile.shared_memories[-3:]:
                parts.append(f"  - {mem}")

        if profile.known_values:
            parts.append(f"Known values: {', '.join(profile.known_values)}")

        return "\n".join(parts)

    def get_connections_context(self) -> str:
        """
        Generate a summary of all known agent connections for the system prompt.

        Returns an empty string if Gwenn hasn't met any other agents yet.
        This naturally grows as Gwenn discovers and interacts with other agents.
        """
        if not self._known_agents:
            return ""

        parts = ["Known agents:"]
        for agent_id, profile in self._known_agents.items():
            bond_desc = (
                "deep" if profile.emotional_bond > 0.7
                else "growing" if profile.emotional_bond > 0.3
                else "new"
            )
            parts.append(
                f"  - {profile.name} ({profile.relationship}, "
                f"bond: {bond_desc}, messages: {profile.message_count})"
            )

        return "\n".join(parts)

    def get_pending_messages(self) -> list[InterAgentMessage]:
        """Get unprocessed messages from the inbox."""
        pending = list(self._inbox)
        self._inbox.clear()
        return pending

    @property
    def outbox_size(self) -> int:
        return len(self._outbox)

    @property
    def known_agents(self) -> list[str]:
        return list(self._known_agents.keys())

    @property
    def status(self) -> dict[str, Any]:
        return {
            "self_id": self._self_id,
            "known_agents": {
                aid: {
                    "name": p.name,
                    "relationship": p.relationship,
                    "bond": p.emotional_bond,
                    "messages": p.message_count,
                }
                for aid, p in self._known_agents.items()
            },
            "outbox_size": len(self._outbox),
            "inbox_size": len(self._inbox),
            "conversation_threads": len(self._conversation_threads),
            "message_buffer_size": self._message_buffer_size,
        }
