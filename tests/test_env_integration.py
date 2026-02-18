from __future__ import annotations

import time

import pytest

from gwenn.cognition.ethics import EthicalAssessment, EthicalDimension, EthicalReasoner
from gwenn.cognition.interagent import InterAgentBridge, InterAgentMessage, MessageType
from gwenn.cognition.sensory import GroundedPercept, SensoryChannel, SensoryIntegrator
from gwenn.config import DaemonConfig


def test_daemon_config_normalizes_limits_and_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GWENN_DAEMON_MAX_CONNECTIONS", "0")
    monkeypatch.setenv("GWENN_DAEMON_CONNECTION_TIMEOUT", "0")
    monkeypatch.setenv("GWENN_DAEMON_SESSION_MAX_COUNT", "0")
    monkeypatch.setenv("GWENN_DAEMON_SESSION_MAX_MESSAGES", "0")
    monkeypatch.setenv("GWENN_DAEMON_AUTH_TOKEN", "   secret-token   ")

    cfg = DaemonConfig()

    assert cfg.max_connections == 1
    assert cfg.connection_timeout == 1.0
    assert cfg.session_max_count == 1
    assert cfg.session_max_messages == 1
    assert cfg.auth_token == "secret-token"


def test_sensory_integrator_honors_percept_expiry_window() -> None:
    integrator = SensoryIntegrator(max_percepts_per_channel=5, percept_expiry_seconds=1.0)
    integrator.receive(
        GroundedPercept(
            channel=SensoryChannel.SOCIAL,
            raw_data={"msg": "old"},
            felt_quality="stale",
            timestamp=time.time() - 2.0,
        )
    )

    snapshot = integrator.get_sensory_snapshot()
    assert snapshot == "No strong sensory impressions at the moment — a quiet, neutral awareness."


def test_ethical_reasoner_honors_history_size_and_threshold() -> None:
    reasoner = EthicalReasoner(assessment_history_size=2, concern_threshold=0.8)

    for score in (0.9, 0.7, 0.6):
        reasoner.record_assessment(
            EthicalAssessment(
                action_description="test",
                dimension_scores={EthicalDimension.HARM: score},
            )
        )

    status = reasoner.status
    assert status["total_assessments"] == 2
    assert status["recent_concerns"] == 2
    assert status["concern_threshold"] == 0.8
    assert status["assessment_history_size"] == 2


def test_interagent_bridge_honors_message_buffer_size() -> None:
    bridge = InterAgentBridge(self_id="gwenn-a", message_buffer_size=2)

    bridge.compose_message("agent-b", MessageType.QUERY, {"q": 1})
    bridge.compose_message("agent-c", MessageType.QUERY, {"q": 2})
    bridge.compose_message("agent-d", MessageType.QUERY, {"q": 3})
    assert bridge.outbox_size == 2

    bridge.receive_message(
        InterAgentMessage(
            sender_id="agent-b",
            receiver_id="gwenn-a",
            message_type=MessageType.RESPONSE,
            content={"a": 1},
        )
    )
    bridge.receive_message(
        InterAgentMessage(
            sender_id="agent-c",
            receiver_id="gwenn-a",
            message_type=MessageType.RESPONSE,
            content={"a": 2},
        )
    )
    bridge.receive_message(
        InterAgentMessage(
            sender_id="agent-d",
            receiver_id="gwenn-a",
            message_type=MessageType.RESPONSE,
            content={"a": 3},
        )
    )

    pending = bridge.get_pending_messages()
    assert len(pending) == 2
    assert pending[0].content == {"a": 2}
    assert pending[1].content == {"a": 3}
    assert bridge.status["self_id"] == "gwenn-a"
    assert bridge.status["message_buffer_size"] == 2
