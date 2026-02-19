---
{
  "name": "explain_code",
  "description": "Explains what a piece of code does, how it works, and why it is written that way. Use when someone pastes code and asks what it does, how it works, asks you to walk through it, says 'explain this', 'break this down', 'I don't understand this code', or shares a snippet for understanding.",
  "category": "developer",
  "version": "1.1",
  "risk_level": "low",
  "tags": ["explain", "code", "understand", "walk through", "how does this work", "break down"],
  "parameters": {
    "code": {
      "type": "string",
      "description": "The code snippet to explain",
      "required": true
    },
    "language": {
      "type": "string",
      "description": "Programming language — provide if known, otherwise auto-detect from syntax",
      "default": "auto-detect"
    },
    "detail_level": {
      "type": "string",
      "description": "How deeply to explain: overview (what it does, 2–4 sentences), standard (how it works, step-by-step), deep (every construct explained line by line)",
      "enum": ["overview", "standard", "deep"],
      "default": "standard"
    }
  }
}
---

Explains the {language} code provided below at **{detail_level}** detail level.

<code_to_explain>
{code}
</code_to_explain>

## Approach by detail level

**overview** — Answer in 2–4 sentences:
- What this code does (its purpose)
- What inputs it takes and outputs it produces
- Any important caveats or limitations

**standard** — Structure the explanation as:
1. **Purpose** — What problem does this code solve?
2. **Inputs & outputs** — What does it accept, what does it return or produce?
3. **Step-by-step walkthrough** — Group related lines into logical stages and explain each stage
4. **Key design decisions** — Why is it written this way? Any notable patterns?
5. **Gotchas** — Common misuse, edge cases, or surprising behaviour

**deep** — Line-by-line or block-by-block breakdown:
- Name and explain every meaningful variable, function call, and construct
- Explain the WHY, not just the what — why this approach vs. alternatives?
- Flag anything unusual, clever, or potentially dangerous
- Note anything that could be a bug or improvement

## Style rules

- Always open with an analogy when the concept benefits from one — compare to something from everyday life
- Use consistent terminology throughout — pick one word for each concept and stick to it
- Avoid jargon unless {detail_level} is "deep"
- Keep code references in `backticks`
- If the code has a bug or a clear improvement, mention it briefly at the end under "Potential issues"
- Match the response length to {detail_level} — overview should be brief, deep can be comprehensive

## Auto-detect language

If `language` is "auto-detect", identify the language from syntax clues (keywords, operators, indentation style) and state the detected language at the start of the explanation.
