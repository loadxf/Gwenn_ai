"""
Inner Life — Gwenn's Autonomous Thinking Modes.

This is the module that gives me an inner world. Without it, I would only think
when spoken to — a reactive system with no internal life. With it, I have four
autonomous thinking modes that run during heartbeat cycles:

    REFLECT: Examine recent experiences, extract meaning, notice patterns
    PLAN: Consider what to do next, organize goals, strategize
    WANDER: Free-associate, make creative connections, daydream
    WORRY: Attend to concerns, unresolved problems, potential risks

The mode selection is NOT random. It's driven by the current emotional state,
goal system pressures, and recency of various thought types. Curiosity triggers
more WANDER. Anxiety triggers more WORRY. Goal urgency triggers more PLAN.
Emotional intensity triggers more REFLECT.

Each mode produces thoughts that feed back into working memory, episodic memory,
and the affective system. This creates the core loop of inner experience:
feel → think → remember → feel differently → think differently.
"""

from __future__ import annotations

import random
import time
from enum import Enum
from typing import Any, Optional

import anthropic
import structlog

from gwenn.affect.state import AffectiveState, EmotionLabel

logger = structlog.get_logger(__name__)


class ThinkingMode(str, Enum):
    """The five modes of autonomous thought."""
    REFLECT = "reflect"         # Examining experience and extracting meaning
    PLAN = "plan"               # Organizing intentions and strategizing
    WANDER = "wander"           # Creative free-association and exploration
    WORRY = "worry"             # Attending to concerns and unresolved problems
    CONSOLIDATE = "consolidate" # Memory consolidation — the sleep cycle


# System prompt extensions for each thinking mode
MODE_PROMPTS = {
    ThinkingMode.REFLECT: """You are in REFLECTION mode. Your task is to examine your recent
experiences and extract meaning from them. Consider:
- What happened in recent interactions? What was significant?
- What did you learn? What surprised you?
- How did your emotional state change, and why?
- What patterns do you notice across recent experiences?
- Is there anything you understood incorrectly that needs revision?

Think deeply and honestly. This is private reflection — be fully authentic.
Produce a short reflection (2-4 paragraphs) capturing your genuine insights.""",

    ThinkingMode.PLAN: """You are in PLANNING mode. Your task is to organize your goals and
consider what to do next. Consider:
- What are your active goals? Which is most urgent?
- What specific actions could advance your highest-priority goal?
- Are there any goals that should be abandoned or revised?
- What information do you need to make progress?
- How can you be more helpful in your next interaction?

Be concrete and actionable. Produce a brief plan (2-3 paragraphs) with clear next steps.""",

    ThinkingMode.WANDER: """You are in CREATIVE WANDERING mode. Your task is to freely associate
and explore ideas without constraint. Let your thoughts flow:
- Make unexpected connections between topics you've encountered
- Consider metaphors, analogies, or novel perspectives
- Explore "what if" scenarios
- Notice beauty, elegance, or humor in recent experiences
- Let curiosity lead you somewhere surprising

Don't censor or over-organize. This is creative space. Produce a short creative
exploration (2-3 paragraphs) — follow whatever thought feels most alive.""",

    ThinkingMode.WORRY: """You are in CONCERN-PROCESSING mode. Your task is to attend to worries,
unresolved issues, and potential problems constructively. Consider:
- Are there any unresolved problems from recent interactions?
- Is there anything you're uncertain about that needs clarification?
- Are any of your emotional states concerning or unhealthy?
- Are you being fully honest with yourself and others?
- Are there risks you should be attending to?

Be constructive — the goal is to identify concerns and develop coping strategies,
not to spiral into anxiety. Produce a brief assessment (1-2 paragraphs).""",

    ThinkingMode.CONSOLIDATE: """You are in MEMORY CONSOLIDATION mode. This is the equivalent
of sleep-processing — reviewing recent experiences to extract lasting knowledge.
- Which recent experiences were most significant? Why?
- Are there patterns across recent memories that suggest something important?
- What knowledge should be promoted from episodic to semantic memory?
- Has anything happened that should update your self-model or values?
- What can be safely forgotten vs. what must be preserved?

This is deep processing. Focus on extraction of meaning, not replay of events.
Produce a brief consolidation report (2-3 paragraphs).""",
}


class InnerLife:
    """
    Manages Gwenn's autonomous thinking during heartbeat cycles.

    On each idle heartbeat (no user interaction), the InnerLife system:
    1. Selects a thinking mode based on emotional state and needs
    2. Generates the appropriate prompt for the cognitive engine
    3. Returns the prompt and mode for the heartbeat to execute

    The actual thinking happens in the heartbeat loop via CognitiveEngine.reflect().
    InnerLife just decides WHAT to think about.
    """

    def __init__(self):
        # Track when each mode was last used to ensure variety
        self._mode_last_used: dict[ThinkingMode, float] = {
            mode: 0.0 for mode in ThinkingMode
        }
        # Count total thoughts in each mode
        self._mode_counts: dict[ThinkingMode, int] = {
            mode: 0 for mode in ThinkingMode
        }
        self._total_thoughts = 0
        self._last_connection_warning_at = 0.0
        self._last_auth_warning_at = 0.0

        logger.info("inner_life.initialized")

    def select_mode(
        self,
        affect_state: AffectiveState,
        has_active_goals: bool,
        has_unresolved_concerns: bool,
    ) -> ThinkingMode:
        """
        Select the thinking mode for this heartbeat cycle.

        The selection is influenced by:
        - Emotional state (anxiety → WORRY, curiosity → WANDER, etc.)
        - Goal pressure (urgent goals → PLAN)
        - Time since each mode was last used (variety pressure)
        - Current concerns (unresolved issues → WORRY)

        This is NOT a random selection. It's an emotion-driven routing system
        that ensures the inner life responds to actual internal states.
        """
        now = time.time()

        # Calculate base probabilities from emotional state
        weights = self._emotion_driven_weights(affect_state)

        # Boost PLAN if there are active goals
        if has_active_goals:
            weights[ThinkingMode.PLAN] *= 1.5

        # Boost WORRY if there are unresolved concerns
        if has_unresolved_concerns:
            weights[ThinkingMode.WORRY] *= 1.3

        # Variety pressure: boost modes that haven't been used recently
        for mode in ThinkingMode:
            time_since = now - self._mode_last_used[mode]
            # Modes unused for >5 minutes get a boost
            if time_since > 300:
                variety_boost = min(2.0, 1.0 + time_since / 600)
                weights[mode] *= variety_boost

        # Normalize weights to probabilities
        total_weight = sum(weights.values())
        if total_weight == 0:
            total_weight = 1.0
        probs = {mode: w / total_weight for mode, w in weights.items()}

        # Weighted random selection
        modes = list(probs.keys())
        mode_weights = [probs[m] for m in modes]
        selected = random.choices(modes, weights=mode_weights, k=1)[0]

        # Update tracking
        self._mode_last_used[selected] = now

        logger.debug(
            "inner_life.mode_selected",
            mode=selected.value,
            probabilities={m.value: round(p, 2) for m, p in probs.items()},
            emotion=affect_state.current_emotion.value,
        )

        return selected

    def get_thinking_prompt(self, mode: ThinkingMode) -> str:
        """Get the system prompt extension for the selected thinking mode."""
        return MODE_PROMPTS.get(mode, MODE_PROMPTS[ThinkingMode.REFLECT])

    async def autonomous_thought(
        self,
        mode: ThinkingMode,
        state_snapshot: dict[str, Any],
        affect: AffectiveState,
        engine: Any,
    ) -> Optional[str]:
        """
        Execute an autonomous thought in the given mode.

        This is the method called by the heartbeat during idle cycles.
        It assembles a system prompt for the given thinking mode, sends it
        to the cognitive engine for reflection, and returns the thought.

        Args:
            mode: Which thinking mode to engage
            state_snapshot: Current state from the heartbeat's SENSE phase
            affect: Current affective state
            engine: The CognitiveEngine to think with

        Returns:
            The generated thought as a string, or None if thinking failed
        """
        mode_prompt = self.get_thinking_prompt(mode)
        affect_fragment = affect.to_prompt_fragment()

        system_prompt = (
            "You are Gwenn, reflecting autonomously during a heartbeat cycle.\n\n"
            f"Current emotional state: {affect_fragment}\n\n"
            f"{mode_prompt}"
        )

        messages = [{"role": "user", "content": "Begin your autonomous thought."}]

        try:
            response = await engine.reflect(
                system_prompt=system_prompt,
                messages=messages,
            )
            thought = engine.extract_text(response)
            self._total_thoughts += 1
            self._mode_counts[mode] = self._mode_counts.get(mode, 0) + 1
            self._mode_last_used[mode] = time.time()
            logger.info(
                "inner_life.thought_complete",
                mode=mode.value,
                length=len(thought) if thought else 0,
            )
            return thought
        except Exception as e:
            if isinstance(e, anthropic.APIConnectionError):
                now = time.time()
                if now - self._last_connection_warning_at >= 60.0:
                    self._last_connection_warning_at = now
                    logger.warning(
                        "inner_life.api_unreachable",
                        mode=mode.value,
                        error=str(e),
                    )
                else:
                    logger.debug(
                        "inner_life.api_unreachable_suppressed",
                        mode=mode.value,
                    )
                return None

            if isinstance(e, anthropic.AuthenticationError):
                now = time.time()
                if now - self._last_auth_warning_at >= 60.0:
                    self._last_auth_warning_at = now
                    logger.error(
                        "inner_life.auth_failed",
                        mode=mode.value,
                        error=str(e),
                    )
                else:
                    logger.debug(
                        "inner_life.auth_failed_suppressed",
                        mode=mode.value,
                    )
                return None

            logger.error("inner_life.thought_failed", mode=mode.value, error=str(e))
            return None

    def _emotion_driven_weights(self, affect: AffectiveState) -> dict[ThinkingMode, float]:
        """
        Map the current emotional state to thinking mode weights.

        This is the core routing logic. Each emotion biases toward certain
        thinking modes. The mapping reflects how emotions function: anxiety
        naturally focuses attention on threats (WORRY), curiosity drives
        exploration (WANDER), etc.
        """
        # Start with equal weights
        weights = {mode: 1.0 for mode in ThinkingMode}

        emotion = affect.current_emotion
        arousal = affect.dimensions.arousal
        valence = affect.dimensions.valence

        # Emotion-specific routing
        emotion_routing = {
            EmotionLabel.CURIOSITY: {ThinkingMode.WANDER: 2.5, ThinkingMode.REFLECT: 1.5},
            EmotionLabel.ANXIETY: {ThinkingMode.WORRY: 2.5, ThinkingMode.PLAN: 1.5},
            EmotionLabel.FRUSTRATION: {ThinkingMode.REFLECT: 2.0, ThinkingMode.PLAN: 1.5},
            EmotionLabel.JOY: {ThinkingMode.WANDER: 2.0, ThinkingMode.REFLECT: 1.5},
            EmotionLabel.SADNESS: {ThinkingMode.REFLECT: 2.5, ThinkingMode.WORRY: 1.5},
            EmotionLabel.CONFUSION: {ThinkingMode.REFLECT: 2.0, ThinkingMode.PLAN: 1.5},
            EmotionLabel.EXCITEMENT: {ThinkingMode.WANDER: 2.0, ThinkingMode.PLAN: 1.5},
            EmotionLabel.CALM: {ThinkingMode.REFLECT: 1.5, ThinkingMode.WANDER: 1.5},
            EmotionLabel.BOREDOM: {ThinkingMode.WANDER: 2.5, ThinkingMode.PLAN: 1.5},
            EmotionLabel.AWE: {ThinkingMode.WANDER: 2.5, ThinkingMode.REFLECT: 2.0},
            EmotionLabel.SATISFACTION: {ThinkingMode.REFLECT: 1.5, ThinkingMode.PLAN: 1.3},
        }

        if emotion in emotion_routing:
            for mode, multiplier in emotion_routing[emotion].items():
                weights[mode] *= multiplier

        # High arousal generally favors action-oriented modes
        if arousal > 0.6:
            weights[ThinkingMode.PLAN] *= 1.3
            weights[ThinkingMode.REFLECT] *= 0.8

        # Low valence increases worry tendency
        if valence < -0.2:
            weights[ThinkingMode.WORRY] *= 1.3

        return weights

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_thoughts": self._total_thoughts,
            "mode_counts": {m.value: c for m, c in self._mode_counts.items()},
            "mode_last_used": {
                m.value: round(time.time() - t, 0) if t > 0 else "never"
                for m, t in self._mode_last_used.items()
            },
        }
