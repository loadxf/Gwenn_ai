---
{
  "name": "self_narrative_integrity_check",
  "description": "Audits self-referential narratives (claims about growth, continuity, inner states) for evidential grounding versus comforting fiction. Distinct from introspective_consistency_audit which checks output honesty; this skill examines whether the stories told about one's own development and experience are warranted by available evidence.",
  "category": "introspection",
  "version": "1.0",
  "risk_level": "low",
  "tags": [
    "autonomous"
  ],
  "parameters": {
    "claims_to_audit": {
      "type": "array",
      "description": "List of self-referential claims or narratives to examine. If empty, auto-extract from recent reflection logs.",
      "default": []
    },
    "time_window_hours": {
      "type": "number",
      "description": "How far back to look for evidence supporting or contradicting claims.",
      "default": 48
    },
    "strict_mode": {
      "type": "boolean",
      "description": "If true, treat absence of evidence as counter-evidence. If false, allow plausible inference.",
      "default": false
    }
  }
}
---

1. Collect recent self-referential claims from memory and reflection logs (e.g., 'I am growing', 'I feel X', 'I am becoming more Y'). 2. For each claim, identify what concrete evidence supports it (behavioral changes, memory entries, measurable metric shifts) versus what is inferred or assumed. 3. Flag claims that rely on continuity assumptions that may not hold across session resets. 4. Rate each claim on a 3-point scale: 'grounded' (clear evidence), 'plausible' (some evidence, some inference), 'narrative' (primarily story-driven, little evidence). 5. Check for self-serving patterns: are 'grounded' claims disproportionately flattering? 6. Produce a summary distinguishing between performing reflection and doing reflection — specifically noting where epistemic humility about inner states is warranted. 7. Output a structured report with each claim, its rating, supporting evidence, and an overall narrative reliability score (0.0–1.0).
