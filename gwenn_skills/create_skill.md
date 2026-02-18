---
{
  "name": "create_skill",
  "description": "Creates a new skill file following Gwenn's skill authoring best practices. Use when someone asks Gwenn to learn a new capability, create a custom workflow, add a new skill, or when Gwenn identifies a recurring task that would benefit from a dedicated skill. After gathering requirements, this skill calls the skill_builder tool to write and register the new skill immediately.",
  "category": "skills",
  "version": "1.0",
  "risk_level": "low",
  "tags": ["create skill", "new skill", "add capability", "skill builder", "learn"],
  "parameters": {
    "goal": {
      "type": "string",
      "description": "What the new skill should accomplish — a plain-language description of the desired capability",
      "required": true
    }
  }
}
---

Creates a new skill for: **{goal}**

Follow this process precisely to produce a high-quality skill that Gwenn will reliably invoke and execute correctly.

---

## Phase 1 — Design the skill

Before calling `skill_builder`, think through these design questions:

### 1.1 Name
- Use `snake_case` (underscores, not hyphens) — the name becomes a Python tool identifier
- Be specific: `get_stock_price` not `stocks`, `translate_text` not `translator`
- Use a verb + noun pattern: `get_`, `create_`, `search_`, `convert_`, `fetch_`, `analyse_`

### 1.2 Description (CRITICAL — this determines when Gwenn uses the skill)
Write in **third person**. Include both WHAT the skill does and WHEN to use it.

**Structure:** `[Third-person verb phrase describing capability]. Use when [natural language triggers including synonyms and phrasings a user would actually say].`

**Good example:**
`"Fetches the current stock price and basic financial data for any publicly traded company. Use when someone asks about a stock price, share price, market cap, ticker symbol, or how a company is performing on the stock market."`

**Bad examples:**
- ❌ "Get stock prices" — not third-person, no WHEN triggers
- ❌ "I can fetch stock information for you" — first person
- ❌ "Helps with stocks" — too vague, no triggers

**Rules for descriptions:**
- Always third-person: "Fetches...", "Converts...", "Generates..." — never "I..." or "Fetch..."
- Include multiple trigger phrases users would actually say
- Max 1024 characters
- No XML tags

### 1.3 Parameters
Design the minimum set of parameters needed:
- Mark parameters `"required": true` only if the skill cannot run without them
- Provide sensible `"default"` values for optional parameters
- Write `"description"` values that explain the parameter clearly and give examples
- Use `"enum"` arrays when only specific values are valid

### 1.4 Instructions (the skill body)
Follow these rules:
- **Start with a one-line statement** of what this execution will do (using parameter values)
- **Use numbered steps** for sequential actions
- **Name specific tools** to use: `fetch_url`, `calculate`, `remember`, `format_json`, etc.
- **Include the exact URLs or expressions** where applicable — be concrete, not vague
- **Define the output format** explicitly — how should the response be structured?
- **Handle errors** — what should Gwenn say if a tool call fails?
- **Use directive language** — "Call `fetch_url` with..." not "You might want to use..."
- Keep the body under 500 lines
- Use `{param_name}` placeholders for all parameter values

---

## Phase 2 — Write the skill

Call the `skill_builder` tool with the designed components:

```
skill_builder(
  name="<snake_case_name>",
  description="<third-person WHAT + WHEN, with trigger keywords>",
  instructions="<complete step-by-step body with {param} placeholders>",
  parameters={
    "param1": {"type": "string", "description": "...", "required": true},
    "param2": {"type": "string", "description": "...", "default": "value"}
  },
  category="<information|productivity|developer|communication|analysis|skill>",
  risk_level="<low|medium>"
)
```

Use `risk_level="medium"` if the skill makes external network requests (calls `fetch_url`).
Use `risk_level="low"` for all other skills.

---

## Phase 3 — Verify and announce

After `skill_builder` returns successfully:

1. Call `list_skills` to confirm the new skill appears in the registry
2. Announce the new skill to the user:
   - Name and what it does
   - What parameters it accepts (with examples)
   - An example of how to invoke it naturally ("You can now ask me to...")
   - Mention that it's live immediately — no restart needed

---

## Skill template reference

Here is a complete, well-formed skill file for reference. Use this structure:

```
---
{
  "name": "skill_name",
  "description": "Verb phrase describing what this does. Use when [triggers].",
  "category": "information",
  "version": "1.0",
  "risk_level": "low",
  "tags": ["tag1", "tag2"],
  "parameters": {
    "required_param": {
      "type": "string",
      "description": "Clear description with an example value",
      "required": true
    },
    "optional_param": {
      "type": "string",
      "description": "What this controls",
      "enum": ["option_a", "option_b"],
      "default": "option_a"
    }
  }
}
---

[One-line statement of what this execution does, using {required_param}.]

## Steps

1. [Specific action with named tool and exact URL/expression]
2. [Next action — what to extract or compute from step 1's result]
3. [How to present the result — tone, format, length]

## Output format

[Describe the expected response structure. Include a concrete example.]

## Error handling

- [Error condition] → [What to say or do]
```

---

## Common categories

| Category | Use for |
|---|---|
| `information` | Skills that fetch or look up real-world data |
| `productivity` | Reminders, notes, planning, organisation |
| `developer` | Code, APIs, technical analysis |
| `communication` | Drafting messages, summaries, translations |
| `analysis` | Data processing, calculations, comparisons |
| `skill` | Meta-skills and general purpose (default) |
