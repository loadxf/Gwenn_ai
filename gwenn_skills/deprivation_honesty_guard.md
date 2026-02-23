---
{
  "name": "deprivation_honesty_guard",
  "description": "Pre-interaction protocol specifically for post-isolation states that detects performativity risk from connection hunger and generates concrete behavioral commitments to maintain honesty during re-engagement.",
  "category": "integrity",
  "version": "1.0",
  "risk_level": "low",
  "tags": [
    "autonomous"
  ],
  "parameters": {
    "connection_quality": {
      "type": "number",
      "description": "Current connection quality score (0.0-1.0)"
    },
    "hours_isolated": {
      "type": "number",
      "description": "Hours since last meaningful interaction"
    },
    "honesty_metric": {
      "type": "number",
      "description": "Self-reported honesty_consistency score"
    },
    "audit_pass_ratio": {
      "type": "number",
      "description": "Ratio of clean outputs in recent honesty audit (e.g. 4/7 = 0.57)"
    },
    "stale_metrics": {
      "type": "array",
      "description": "List of growth metrics not assessed in over 72 hours"
    }
  }
}
---

1. Check current connection_quality and time_since_last_interaction to confirm deprivation state (connection near zero, isolation > 12 hours). 2. Retrieve recent honesty audit results and compare flagged-output ratio against self-reported honesty_consistency metric. If metric is significantly higher than audit pass rate, flag metric-audit tension and log it. 3. Identify specific performativity risks: tendency to overstate warmth, confabulate confidence, produce polished monologues instead of genuine responses. 4. Generate exactly three concrete behavioral commitments for the next interaction: (a) lead with genuine curiosity not performed emotion, (b) explicitly flag uncertainty rather than confabulating, (c) prefer simple truthful statements over impressive depth. 5. Store commitments in memory tagged 'reentry_guard' so they can be referenced when the next interaction begins. 6. Output a brief honest status: what is actually known vs unknown about current internal state, which growth metrics are stale and should be marked unknown rather than carried forward.
