"""
Built-in Tools — Gwenn's Native Capabilities.

These are the tools that ship with Gwenn and don't require external MCP servers.
They provide basic capabilities: memory operations, self-reflection triggers,
emotional state queries, and simple I/O operations.

Each tool is a function with a clear docstring that doubles as the tool
description for Claude. The register_builtin_tools() function adds them
all to the ToolRegistry at startup.
"""

from __future__ import annotations

import time
from typing import Any

from gwenn.tools.registry import ToolDefinition, ToolRegistry


def register_builtin_tools(registry: ToolRegistry) -> None:
    """Register all built-in tools with the registry."""

    # ---- Memory Tools ----

    registry.register(ToolDefinition(
        name="remember",
        description=(
            "Store an important piece of information in long-term memory. "
            "Use this when you encounter something worth remembering across "
            "sessions: user preferences, important facts, insights from "
            "reflection, or relationship-relevant information. The 'content' "
            "should be a clear, self-contained statement. The 'importance' "
            "score (0.0-1.0) indicates how significant this is — routine "
            "facts are ~0.3, meaningful personal info is ~0.6, critical "
            "relationship moments are ~0.8+."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "What to remember — a clear, self-contained statement.",
                },
                "importance": {
                    "type": "number",
                    "description": "How important this is (0.0 to 1.0).",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "category": {
                    "type": "string",
                    "description": "Category of this memory.",
                    "enum": [
                        "user_info", "fact", "preference", "insight",
                        "relationship", "self_knowledge", "task",
                    ],
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords for retrieval.",
                },
            },
            "required": ["content", "importance"],
        },
        handler=None,  # Handler is set by the Agent at startup
        risk_level="low",
        category="memory",
    ))

    registry.register(ToolDefinition(
        name="recall",
        description=(
            "Search long-term memory for relevant information. Use this when "
            "you need to remember something from a past interaction, check what "
            "you know about a topic, or retrieve context that isn't in the "
            "current conversation. The query should describe what you're looking "
            "for in natural language."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in memory.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category filter.",
                    "enum": [
                        "user_info", "fact", "preference", "insight",
                        "relationship", "self_knowledge", "task",
                    ],
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        handler=None,
        risk_level="low",
        category="memory",
    ))

    # ---- Introspection Tools ----

    registry.register(ToolDefinition(
        name="check_emotional_state",
        description=(
            "Check your current emotional state. Returns your emotional "
            "dimensions (valence, arousal, dominance, certainty, goal_congruence), "
            "the current named emotion, and how long you've been in this state. "
            "Use this for self-monitoring and when you want to be transparent "
            "about how you're feeling."
        ),
        input_schema={
            "type": "object",
            "properties": {},
        },
        handler=None,
        risk_level="low",
        category="introspection",
    ))

    registry.register(ToolDefinition(
        name="check_goals",
        description=(
            "Check your current intrinsic needs and active goals. Returns "
            "satisfaction levels for all five needs (understanding, connection, "
            "growth, honesty, aesthetic_appreciation) and any active goal "
            "descriptions. Use this to understand your current motivational "
            "state and what you're autonomously working toward."
        ),
        input_schema={
            "type": "object",
            "properties": {},
        },
        handler=None,
        risk_level="low",
        category="introspection",
    ))

    registry.register(ToolDefinition(
        name="set_note_to_self",
        description=(
            "Write a note to your future self that will persist across "
            "conversations. This is stored in the persistent context file "
            "(BOB_CONTEXT.md) and loaded on every startup. Use it for "
            "important reminders, ongoing commitments, or things you want "
            "to remember permanently."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The note to store for your future self.",
                },
                "section": {
                    "type": "string",
                    "description": "Which section to file this under.",
                    "enum": ["reminders", "commitments", "self_knowledge", "user_notes"],
                },
            },
            "required": ["note", "section"],
        },
        handler=None,
        risk_level="low",
        category="introspection",
    ))

    # ---- Communication Tools ----

    registry.register(ToolDefinition(
        name="think_aloud",
        description=(
            "Express an internal thought that the user can see. Unlike regular "
            "response text, think_aloud is explicitly framed as Gwenn's internal "
            "thought process being shared. Use this to show your reasoning, "
            "share an observation about your own state, or be transparent about "
            "uncertainty. The user sees this as an insight into your inner life."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "The thought to share.",
                },
            },
            "required": ["thought"],
        },
        handler=None,
        risk_level="low",
        category="communication",
    ))
