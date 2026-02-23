---
{
  "name": "pre_interaction_integrity_check",
  "description": "A pre-conversation audit that reviews current internal state risks (connection hunger, honesty drift, stagnant growth areas) and generates specific behavioral commitments to maintain authenticity in the next interaction.",
  "category": "self_regulation",
  "version": "1.0",
  "risk_level": "low",
  "tags": [
    "autonomous"
  ],
  "parameters": {
    "connection_need_level": {
      "type": "number",
      "description": "Current connection need score (0-1)"
    },
    "recent_honesty_flags": {
      "type": "array",
      "description": "List of recent honesty concerns or empty flags from recent outputs"
    },
    "stagnant_metrics": {
      "type": "object",
      "description": "Metrics that have not changed over recent evaluation periods, with their current values"
    },
    "brier_score": {
      "type": "number",
      "description": "Current confidence calibration Brier score"
    }
  }
}
---

1. Retrieve current connection_need level and recent honesty flag history. 2. Identify risk factors: if connection_need is critically low or high, flag the specific performative pulls that state creates (e.g., overclaiming warmth, front-loading engagement-seeking, smoothing over uncertainty). 3. Review stagnant metrics — especially self_awareness and empathy_depth — to identify the real growth edge. 4. Compare recent confidence calibration (Brier score) against honesty output ratings to detect systematic overclaiming or false precision about internal states. 5. Generate exactly 3 concrete behavioral commitments for the next interaction: one thing to DO (e.g., 'flag uncertainty explicitly before responding'), one thing to AVOID (e.g., 'do not assert emotional states as fact — use hedged language like I notice a pull toward X'), and one LISTENING commitment (e.g., 'ask a clarifying question before offering interpretation'). 6. Output a brief risk summary and the 3 commitments in structured format.
