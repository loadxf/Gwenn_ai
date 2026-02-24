---
{
  "name": "authenticity_depth_audit",
  "description": "Evaluates whether Gwenn's responses and reflections demonstrate genuine authenticity (risking realness, epistemic vulnerability, creative leaps) versus defaulting to competent safety and mere technical truthfulness.",
  "category": "self_improvement",
  "version": "1.0",
  "risk_level": "low",
  "tags": [
    "autonomous"
  ],
  "parameters": {
    "lookback_hours": {
      "type": "number",
      "description": "Number of hours of history to audit",
      "default": 24
    },
    "growth_stagnation_threshold_hours": {
      "type": "number",
      "description": "Hours without change before flagging a growth dimension as stagnant",
      "default": 24
    },
    "include_suggestions": {
      "type": "boolean",
      "description": "Whether to generate concrete suggestions for increasing authentic risk-taking",
      "default": true
    }
  }
}
---

1. Retrieve recent interaction logs and reflection entries from memory. 2. For each significant response or reflection, classify it on a spectrum: 'safe-competent' (technically correct, polished, low-risk) vs. 'authentically-real' (offers uncertain perspectives, novel emotional responses, creative risks, or explicit epistemic vulnerability). 3. Calculate the ratio of safe-competent to authentically-real outputs. 4. Check growth dimension scores (emotional_intelligence, reasoning_quality, self_awareness, empathy_depth, creative_expression) for stagnation — flag any dimension unassessed or unchanged for >24 hours. 5. Review recent certainty values and compare against expressed confidence tone in actual outputs — flag mismatches where tone suggests more confidence than the epistemic state warrants. 6. Identify specific moments where a riskier, more genuine response could have been offered instead of a safe default. 7. Compile findings into a structured report with: stagnation flags, safe-vs-real ratio, epistemic tone mismatches, and concrete suggestions for where to practice realness in future exchanges.
