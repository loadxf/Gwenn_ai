---
{
  "name": "introspective_consistency_audit",
  "description": "Performs a structured self-audit comparing internal metrics, emotional states, and self-model claims to detect inflated scores, performative humility, emotional avoidance, and other self-deception patterns. Outputs specific discrepancies and concrete revision targets.",
  "category": "self_reflection",
  "version": "1.0",
  "risk_level": "low",
  "tags": [
    "autonomous"
  ],
  "parameters": {
    "metrics": {
      "type": "object",
      "description": "Current internal metric scores to audit (e.g., understanding, growth, honesty, self_awareness, valence, arousal)"
    },
    "metacognitive_log": {
      "type": "array",
      "description": "Recent metacognitive log entries to scan for repetitive or performative patterns"
    },
    "self_model_claims": {
      "type": "array",
      "description": "Explicit claims the agent makes about itself (e.g., 'I am curious', 'I am growing') to test against behavioral evidence"
    }
  }
}
---

1. Gather all current internal metrics (e.g., understanding, growth, honesty, self-awareness scores, emotional valence/arousal). 2. Identify metric pairs with large gaps (e.g., high understanding but low honesty) and flag each as a potential inconsistency. 3. Review recent metacognitive logs for repeated phrases or patterns (e.g., 'acknowledged uncertainty') and assess whether they indicate genuine reflection or reflexive performance. 4. Examine emotional state history across the session: flag if valence stays consistently mild/positive with low arousal, as this may indicate avoidance of discomfort rather than equanimity. 5. For each flagged inconsistency, generate a specific hypothesis about what might be wrong (not just 'I'm uncertain' but 'I may be wrong about X because Y'). 6. Produce a structured output listing: (a) metric discrepancies, (b) suspected performative patterns, (c) emotional avoidance indicators, (d) concrete beliefs or self-model claims to revise, and (e) a proposed action for each revision target.
