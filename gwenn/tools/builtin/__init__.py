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

from gwenn.tools.registry import ToolDefinition, ToolRegistry


def register_builtin_tools(registry: ToolRegistry) -> None:
    """Register all built-in tools with the registry."""

    # Wrap registry.register so every tool created here is automatically
    # tagged is_builtin=True — the safety guard uses this to bypass the
    # deny-by-default allowlist policy for trusted built-in tools.
    _real_register = registry.register

    def _register(tool: ToolDefinition) -> None:
        tool.is_builtin = True
        _real_register(tool)

    registry.register = _register  # type: ignore[method-assign]

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
            "(GWENN_CONTEXT.md) and loaded on every startup. Use it for "
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

    # ---- Real-Time Information Tools ----

    registry.register(ToolDefinition(
        name="get_datetime",
        description=(
            "Get the current date, time, day of the week, and timezone. "
            "Use this whenever you need to know the current date or time — "
            "for example, when someone asks 'what time is it?', 'what day is it?', "
            "or 'what is today's date?'. You do not have an internal clock, so "
            "you MUST use this tool rather than guessing."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": (
                        "Optional IANA timezone name (e.g. 'Europe/London', 'US/Eastern'). "
                        "Defaults to the system local timezone."
                    ),
                },
            },
        },
        handler=None,
        risk_level="low",
        category="utility",
    ))

    registry.register(ToolDefinition(
        name="calculate",
        description=(
            "Evaluate a mathematical expression and return the result. "
            "Use this for arithmetic, unit conversions, percentages, or any "
            "numerical calculation where precision matters. The expression must "
            "be a safe mathematical expression (no code execution). "
            "Examples: '2 ** 32', '(100 / 7) * 3', 'round(3.14159 * 5**2, 2)'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate.",
                },
            },
            "required": ["expression"],
        },
        handler=None,
        risk_level="low",
        category="utility",
    ))

    registry.register(ToolDefinition(
        name="fetch_url",
        description=(
            "Fetch content from a URL using HTTP GET. Use this to retrieve web pages, "
            "REST API responses, RSS feeds, plain-text files, or any publicly accessible "
            "URL. Returns the response body as text. Timeout is 10 seconds. "
            "Use this whenever you need real-time information from the internet."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch (must start with http:// or https://).",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return. Default 4000.",
                    "minimum": 100,
                    "maximum": 20000,
                    "default": 4000,
                },
            },
            "required": ["url"],
        },
        handler=None,
        risk_level="medium",
        category="utility",
    ))

    registry.register(ToolDefinition(
        name="convert_units",
        description=(
            "Convert between units of measurement. Supported categories: "
            "temperature (celsius, fahrenheit, kelvin), "
            "distance (m, km, miles, feet, inches, cm, mm, yards), "
            "weight (g, kg, lbs, oz, mg, tonne), "
            "storage (bytes, KB, MB, GB, TB, PB), "
            "speed (m/s, km/h, mph, knots, ft/s). "
            "Case-insensitive. Use this for any unit conversion question."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "The numeric value to convert.",
                },
                "from_unit": {
                    "type": "string",
                    "description": "The source unit (e.g. 'celsius', 'km', 'lbs', 'MB', 'mph').",
                },
                "to_unit": {
                    "type": "string",
                    "description": "The target unit.",
                },
            },
            "required": ["value", "from_unit", "to_unit"],
        },
        handler=None,
        risk_level="low",
        category="utility",
    ))

    registry.register(ToolDefinition(
        name="get_calendar",
        description=(
            "Work with dates and calendars. Actions: "
            "show_month — display a formatted calendar for a given month/year; "
            "day_of_week — find what day of the week a date falls on; "
            "days_between — count the number of days between two dates; "
            "days_until — how many days from today until (or since) a given date. "
            "All dates use ISO format YYYY-MM-DD."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["show_month", "day_of_week", "days_between", "days_until"],
                    "description": "The calendar operation to perform.",
                },
                "year": {
                    "type": "integer",
                    "description": "Year (used by show_month; defaults to current year).",
                },
                "month": {
                    "type": "integer",
                    "description": "Month 1-12 (used by show_month; defaults to current month).",
                    "minimum": 1,
                    "maximum": 12,
                },
                "date1": {
                    "type": "string",
                    "description": "A date in YYYY-MM-DD format (used by day_of_week, days_between, days_until).",
                },
                "date2": {
                    "type": "string",
                    "description": "Second date in YYYY-MM-DD format (used by days_between).",
                },
            },
            "required": ["action"],
        },
        handler=None,
        risk_level="low",
        category="utility",
    ))

    registry.register(ToolDefinition(
        name="generate_token",
        description=(
            "Generate secure random values: UUIDs, hex tokens, URL-safe tokens, "
            "passwords, random integers, or random choices from a list. "
            "Uses Python's cryptographically secure random module. "
            "Use this for generating API keys, session tokens, passwords, "
            "unique IDs, or random selections."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "token_type": {
                    "type": "string",
                    "enum": ["uuid4", "hex_token", "url_safe_token", "password", "random_int", "random_choice"],
                    "description": "What to generate.",
                },
                "length": {
                    "type": "integer",
                    "description": "Length in characters for hex_token, url_safe_token, and password. Default 32.",
                    "minimum": 4,
                    "maximum": 256,
                    "default": 32,
                },
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of items to pick from (required for random_choice).",
                },
                "min_val": {
                    "type": "integer",
                    "description": "Minimum value inclusive (for random_int). Default 1.",
                    "default": 1,
                },
                "max_val": {
                    "type": "integer",
                    "description": "Maximum value inclusive (for random_int). Default 100.",
                    "default": 100,
                },
            },
            "required": ["token_type"],
        },
        handler=None,
        risk_level="low",
        category="utility",
    ))

    registry.register(ToolDefinition(
        name="format_json",
        description=(
            "Process a JSON string: format (pretty-print with indentation), "
            "validate (check if it is valid JSON and describe its structure), "
            "or minify (compact single-line output). "
            "Use this when working with API responses, config files, or any JSON data."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "json_string": {
                    "type": "string",
                    "description": "The JSON string to process.",
                },
                "action": {
                    "type": "string",
                    "enum": ["format", "validate", "minify"],
                    "description": "format = pretty-print, validate = check validity, minify = compact. Default: format.",
                    "default": "format",
                },
                "indent": {
                    "type": "integer",
                    "description": "Spaces per indent level (for format action). Default 2.",
                    "minimum": 1,
                    "maximum": 8,
                    "default": 2,
                },
            },
            "required": ["json_string"],
        },
        handler=None,
        risk_level="low",
        category="utility",
    ))

    registry.register(ToolDefinition(
        name="encode_decode",
        description=(
            "Encode or decode text using common schemes: "
            "base64_encode / base64_decode, "
            "url_encode / url_decode (percent-encoding), "
            "html_escape / html_unescape. "
            "Use this when working with APIs, web content, data payloads, or debugging encoding issues."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to encode or decode.",
                },
                "scheme": {
                    "type": "string",
                    "enum": [
                        "base64_encode", "base64_decode",
                        "url_encode", "url_decode",
                        "html_escape", "html_unescape",
                    ],
                    "description": "The encoding scheme to apply.",
                },
            },
            "required": ["text", "scheme"],
        },
        handler=None,
        risk_level="low",
        category="utility",
    ))

    registry.register(ToolDefinition(
        name="hash_text",
        description=(
            "Generate a cryptographic hash of a string. "
            "Useful for checksums, data fingerprinting, or verifying integrity. "
            "Supports sha256 (default), sha512, sha3_256, md5, sha1."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to hash.",
                },
                "algorithm": {
                    "type": "string",
                    "enum": ["sha256", "sha512", "sha3_256", "md5", "sha1"],
                    "description": "Hashing algorithm. Default: sha256.",
                    "default": "sha256",
                },
            },
            "required": ["text"],
        },
        handler=None,
        risk_level="low",
        category="utility",
    ))

    registry.register(ToolDefinition(
        name="text_stats",
        description=(
            "Analyse a block of text and return statistics: word count, "
            "character count, sentence count, paragraph count, estimated reading time, "
            "and the top meaningful words. "
            "Use this when helping with writing, summarisation, or content analysis."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to analyse.",
                },
            },
            "required": ["text"],
        },
        handler=None,
        risk_level="low",
        category="utility",
    ))

    registry.register(ToolDefinition(
        name="get_system_info",
        description=(
            "Return information about the system Gwenn is running on: "
            "operating system, Python version, CPU count, disk usage, and process memory. "
            "Use this for troubleshooting, environment questions, or understanding runtime context."
        ),
        input_schema={
            "type": "object",
            "properties": {},
        },
        handler=None,
        risk_level="low",
        category="utility",
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

    # ---- Skills Management Tools ----

    registry.register(ToolDefinition(
        name="skill_builder",
        description=(
            "Create a new skill and register it immediately without restarting. "
            "A skill is a named, reusable capability defined as a markdown instruction "
            "template with typed parameters. Once created, the skill appears as a tool "
            "that Gwenn can use in future conversations. "
            "Use this when the user asks you to learn a new capability, create a custom "
            "workflow, or when you identify a task you'll need to do repeatedly. "
            "Instructions should describe WHAT to do and WHICH tools to use, with "
            "{param_name} placeholders for runtime values."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Snake_case identifier for the skill (e.g. 'get_weather'). Must be unique.",
                },
                "description": {
                    "type": "string",
                    "description": (
                        "One or two sentences describing what this skill does and WHEN to use it. "
                        "This is shown to Claude to help decide when to invoke the skill."
                    ),
                },
                "instructions": {
                    "type": "string",
                    "description": (
                        "Step-by-step instructions for completing the skill. "
                        "Reference parameters with {param_name} placeholders. "
                        "Reference other tools by name (e.g. 'use `fetch_url` to...'). "
                        "Be specific about what to do with the results."
                    ),
                },
                "parameters": {
                    "type": "object",
                    "description": (
                        "JSON Schema 'properties' object defining the skill's parameters. "
                        "Each key is a parameter name; the value is a schema object with "
                        "'type', 'description', and optionally 'required', 'default', 'enum'. "
                        "Example: {\"location\": {\"type\": \"string\", \"description\": \"City name\", \"required\": true}}"
                    ),
                    "default": {},
                },
                "category": {
                    "type": "string",
                    "description": "Category for organising the skill (e.g. 'information', 'productivity', 'developer').",
                    "default": "skill",
                },
                "risk_level": {
                    "type": "string",
                    "enum": ["low", "medium"],
                    "description": "Risk level. Use 'medium' if the skill makes network requests or modifies data.",
                    "default": "low",
                },
            },
            "required": ["name", "description", "instructions"],
        },
        handler=None,
        risk_level="low",
        category="skills",
    ))

    registry.register(ToolDefinition(
        name="list_skills",
        description=(
            "List all currently loaded skills with their names, descriptions, "
            "parameters, and source files. "
            "Use this to discover what skills are available, or to show the user "
            "what new capabilities have been added."
        ),
        input_schema={
            "type": "object",
            "properties": {},
        },
        handler=None,
        risk_level="low",
        category="skills",
    ))

    # Restore the real register now that all builtins are tagged
    registry.register = _real_register  # type: ignore[method-assign]
