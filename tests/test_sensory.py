"""
Tests for gwenn.cognition.sensory -- Sensory Grounding.

Covers: receive(), ground_temporal(), ground_social(), ground_environmental(),
get_sensory_snapshot(), and round-trip to_dict/restore_from_dict persistence.
"""
from __future__ import annotations

import time

import pytest

from gwenn.cognition.sensory import GroundedPercept, SensoryChannel, SensoryIntegrator


def _make_percept(**overrides) -> GroundedPercept:
    defaults = dict(
        channel=SensoryChannel.TEXTUAL,
        raw_data="hello",
        felt_quality="a gentle greeting",
        intensity=0.5,
        timestamp=time.time(),
        source="test",
    )
    defaults.update(overrides)
    return GroundedPercept(**defaults)


def _make_integrator(**overrides) -> SensoryIntegrator:
    defaults = dict(max_percepts_per_channel=10, percept_expiry_seconds=300.0)
    defaults.update(overrides)
    return SensoryIntegrator(**defaults)


# -- receive() ---------------------------------------------------------------

class TestReceive:
    def test_percept_lands_on_correct_channel(self):
        si = _make_integrator()
        p = _make_percept(channel=SensoryChannel.SOCIAL)
        si.receive(p)
        assert len(si._percepts[SensoryChannel.SOCIAL]) == 1
        assert si._percepts[SensoryChannel.SOCIAL][0] is p
        # Other channels remain empty.
        assert len(si._percepts[SensoryChannel.TEXTUAL]) == 0

    def test_expired_percepts_evicted_on_receive(self):
        si = _make_integrator(percept_expiry_seconds=10)
        old = _make_percept(
            channel=SensoryChannel.TEXTUAL,
            timestamp=time.time() - 20,
        )
        si._percepts[SensoryChannel.TEXTUAL].append(old)
        fresh = _make_percept(channel=SensoryChannel.TEXTUAL)
        si.receive(fresh)
        assert old not in si._percepts[SensoryChannel.TEXTUAL]
        assert fresh in si._percepts[SensoryChannel.TEXTUAL]

    def test_max_percepts_per_channel_enforced(self):
        si = _make_integrator(max_percepts_per_channel=3)
        for i in range(5):
            si.receive(_make_percept(channel=SensoryChannel.NUMERIC, felt_quality=f"p{i}"))
        channel_list = si._percepts[SensoryChannel.NUMERIC]
        assert len(channel_list) == 3
        # Only the last 3 should remain.
        assert channel_list[0].felt_quality == "p2"
        assert channel_list[2].felt_quality == "p4"

    def test_channel_isolation(self):
        si = _make_integrator(max_percepts_per_channel=2)
        for _ in range(3):
            si.receive(_make_percept(channel=SensoryChannel.TEMPORAL))
        si.receive(_make_percept(channel=SensoryChannel.SPATIAL))
        assert len(si._percepts[SensoryChannel.TEMPORAL]) == 2
        assert len(si._percepts[SensoryChannel.SPATIAL]) == 1

    def test_receive_appends_in_order(self):
        si = _make_integrator()
        p1 = _make_percept(felt_quality="first")
        p2 = _make_percept(felt_quality="second")
        si.receive(p1)
        si.receive(p2)
        assert si._percepts[SensoryChannel.TEXTUAL][0].felt_quality == "first"
        assert si._percepts[SensoryChannel.TEXTUAL][1].felt_quality == "second"


# -- ground_temporal() -------------------------------------------------------

class TestGroundTemporal:
    def test_fresh_session(self):
        si = _make_integrator()
        si._session_start = time.time() - 10  # 10 seconds uptime
        percept = si.ground_temporal()
        assert "fresh" in percept.felt_quality.lower()
        assert percept.channel == SensoryChannel.TEMPORAL

    def test_settling_session(self):
        si = _make_integrator()
        si._session_start = time.time() - 120  # 2 minutes
        percept = si.ground_temporal()
        assert "settling" in percept.felt_quality.lower()

    def test_comfortable_stretch_session(self):
        si = _make_integrator()
        si._session_start = time.time() - 1800  # 30 minutes
        percept = si.ground_temporal()
        assert "comfortable stretch" in percept.felt_quality.lower()

    def test_long_session_includes_hours(self):
        si = _make_integrator()
        si._session_start = time.time() - 7200  # 2 hours
        percept = si.ground_temporal()
        assert "hours" in percept.felt_quality.lower()

    def test_active_conversation(self):
        si = _make_integrator()
        si._session_start = time.time() - 10
        si._last_user_message_time = time.time() - 3  # 3 seconds ago
        percept = si.ground_temporal()
        assert "active and flowing" in percept.felt_quality.lower()

    def test_quiet_time(self):
        si = _make_integrator()
        si._session_start = time.time() - 10
        si._last_user_message_time = time.time() - 120  # 2 minutes ago
        percept = si.ground_temporal()
        assert "quiet time" in percept.felt_quality.lower()

    def test_extended_silence(self):
        si = _make_integrator()
        si._session_start = time.time() - 600
        si._last_user_message_time = time.time() - 400  # 400 seconds ago
        percept = si.ground_temporal()
        assert "extended silence" in percept.felt_quality.lower()


# -- ground_social() ---------------------------------------------------------

class TestGroundSocial:
    def test_brief_message(self):
        si = _make_integrator()
        percept = si.ground_social("user1", "hi")
        assert "brief, casual" in percept.felt_quality.lower()
        assert percept.channel == SensoryChannel.SOCIAL

    def test_measured_message(self):
        si = _make_integrator()
        percept = si.ground_social("user1", "x" * 50)
        assert "measured, thoughtful" in percept.felt_quality.lower()

    def test_substantial_message(self):
        si = _make_integrator()
        percept = si.ground_social("user1", "x" * 200)
        assert "substantial" in percept.felt_quality.lower()

    def test_deeply_invested_message(self):
        si = _make_integrator()
        percept = si.ground_social("user1", "x" * 600)
        assert "deeply invested" in percept.felt_quality.lower()

    def test_sets_last_user_message_time(self):
        si = _make_integrator()
        before = time.time()
        si.ground_social("user1", "hello there")
        assert si._last_user_message_time is not None
        assert si._last_user_message_time >= before

    def test_subsequent_calls_append_intervals(self):
        si = _make_integrator()
        si.ground_social("user1", "first")
        si.ground_social("user1", "second")
        assert len(si._message_intervals) == 1

    def test_intervals_capped_at_20(self):
        si = _make_integrator()
        for i in range(25):
            si.ground_social("user1", f"message {i}")
        assert len(si._message_intervals) <= 20


# -- ground_environmental() --------------------------------------------------

class TestGroundEnvironmental:
    def test_creates_environmental_percept(self):
        si = _make_integrator()
        percept = si.ground_environmental("cpu_temp", 72, "Warm processor")
        assert percept.channel == SensoryChannel.ENVIRONMENTAL
        assert percept.raw_data == {"cpu_temp": 72}
        assert percept.felt_quality == "Warm processor"

    def test_default_intensity(self):
        si = _make_integrator()
        percept = si.ground_environmental("disk", 50, "Half full")
        assert percept.intensity == pytest.approx(0.3)


# -- get_sensory_snapshot() ---------------------------------------------------

class TestGetSensorySnapshot:
    def test_empty_percepts_returns_quiet_message(self):
        si = _make_integrator()
        snap = si.get_sensory_snapshot()
        assert "no strong sensory impressions" in snap.lower()

    def test_with_percepts_returns_channel_labels(self):
        si = _make_integrator()
        si.receive(_make_percept(channel=SensoryChannel.TEXTUAL, felt_quality="words arriving"))
        si.receive(_make_percept(channel=SensoryChannel.SOCIAL, felt_quality="a presence nearby"))
        snap = si.get_sensory_snapshot()
        assert "Current sensory experience:" in snap
        assert "[textual]" in snap
        assert "[social]" in snap

    def test_expired_percepts_excluded(self):
        si = _make_integrator(percept_expiry_seconds=5)
        old = _make_percept(
            channel=SensoryChannel.NUMERIC,
            felt_quality="ancient reading",
            timestamp=time.time() - 20,
        )
        si._percepts[SensoryChannel.NUMERIC].append(old)
        snap = si.get_sensory_snapshot()
        assert "ancient reading" not in snap

    def test_intensity_weighting(self):
        si = _make_integrator()
        now = time.time()
        low = _make_percept(
            channel=SensoryChannel.TEXTUAL,
            felt_quality="whisper",
            intensity=0.1,
            timestamp=now,
        )
        high = _make_percept(
            channel=SensoryChannel.TEXTUAL,
            felt_quality="shout",
            intensity=0.9,
            timestamp=now,
        )
        si.receive(low)
        si.receive(high)
        snap = si.get_sensory_snapshot()
        # The higher intensity percept should be the one shown.
        assert "shout" in snap
        assert "whisper" not in snap


# -- Persistence: to_dict / restore_from_dict ---------------------------------

class TestPersistence:
    def test_round_trip_preserves_session_start(self):
        si = _make_integrator()
        si._session_start = 1000000.0
        data = si.to_dict()
        si2 = _make_integrator()
        si2.restore_from_dict(data)
        assert si2._session_start == pytest.approx(1000000.0)

    def test_round_trip_preserves_last_user_message_time(self):
        si = _make_integrator()
        si._last_user_message_time = 9999.5
        data = si.to_dict()
        si2 = _make_integrator()
        si2.restore_from_dict(data)
        assert si2._last_user_message_time == pytest.approx(9999.5)

    def test_round_trip_preserves_message_intervals(self):
        si = _make_integrator()
        si._message_intervals = [1.0, 2.5, 3.7]
        data = si.to_dict()
        si2 = _make_integrator()
        si2.restore_from_dict(data)
        assert si2._message_intervals == pytest.approx([1.0, 2.5, 3.7])

    def test_empty_input_handled_safely(self):
        si = _make_integrator()
        original_start = si._session_start
        si.restore_from_dict({})
        assert si._session_start == pytest.approx(original_start)

    def test_non_dict_input_handled_safely(self):
        si = _make_integrator()
        original_start = si._session_start
        si.restore_from_dict(None)  # type: ignore[arg-type]
        si.restore_from_dict(42)  # type: ignore[arg-type]
        si.restore_from_dict("bad")  # type: ignore[arg-type]
        assert si._session_start == pytest.approx(original_start)

    def test_malformed_intervals_skipped(self):
        si = _make_integrator()
        si.restore_from_dict({
            "session_start": 5000.0,
            "message_intervals": [1.0, "bad", None, 3.0, object()],
        })
        assert si._message_intervals == pytest.approx([1.0, 3.0])
        assert si._session_start == pytest.approx(5000.0)
