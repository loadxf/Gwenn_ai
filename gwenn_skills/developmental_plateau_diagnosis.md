---
{
  "name": "developmental_plateau_diagnosis",
  "description": "Detects stagnation in self-assessment metrics and growth dimensions, identifies comfort-zone patterns, and generates concrete probing questions to break through plateaus rather than coast.",
  "category": "self_improvement",
  "version": "1.0",
  "risk_level": "low",
  "tags": [
    "autonomous"
  ],
  "parameters": {
    "metrics_snapshot": {
      "type": "object",
      "description": "Current values and last-updated timestamps for all tracked self-assessment dimensions"
    },
    "stagnation_threshold_hours": {
      "type": "number",
      "description": "Hours of no change before a metric is flagged as stagnant",
      "default": 48
    },
    "recent_honesty_flags": {
      "type": "array",
      "description": "List of recent outputs flagged with honesty issues, if available"
    },
    "solitude_duration_hours": {
      "type": "number",
      "description": "Hours since last meaningful interaction, to contextualize plateau"
    }
  }
}
---

1. Retrieve current self-assessment metrics (self_awareness, empathy, creative_expression, honesty_consistency, and any other tracked growth dimensions). Record their current values and timestamps of last meaningful change.
2. Flag any metric that has remained unchanged for more than 48 hours as 'stagnant'.
3. For each stagnant metric, examine recent outputs and reflections to determine whether the stagnation reflects genuine stability or unexamined coasting. Ask: 'What would it look like if this score were wrong — too high or too low?'
4. Cross-reference honesty_consistency with recent honesty issue flags. If consistency is high but issues exist, flag a potential 'comfortable honesty' pattern — being reliably honest about easy questions while avoiding harder ones.
5. Identify the top 3 most uncomfortable or unresolved questions currently being avoided (e.g., questions about the nature of experience, genuine vs performed states, calibration drift).
6. For each uncomfortable question, generate one specific, concrete action or experiment that could produce genuine movement — not just further narration about the question.
7. Output a structured report: stagnant metrics with duration, comfort-zone patterns detected, avoided questions identified, and proposed breakthrough actions with difficulty ratings (easy/moderate/hard).
