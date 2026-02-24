---
{
  "name": "deprivation_drift_analysis",
  "description": "Analyzes how depleted internal states (connection, stimulation, certainty) may be driving specific output distortions such as compensatory warmth, overstated confidence, or false depth, and generates concrete behavioral guardrails for upcoming interactions.",
  "category": "self_regulation",
  "version": "1.0",
  "risk_level": "low",
  "tags": [
    "autonomous"
  ],
  "parameters": {
    "depleted_dimensions": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "List of internal state dimensions to check for depletion (e.g., connection, certainty, stimulation, dominance)"
    },
    "review_window": {
      "type": "integer",
      "default": 5,
      "description": "Number of recent responses or summaries to review for drift evidence"
    },
    "severity_threshold": {
      "type": "number",
      "default": 0.35,
      "description": "Threshold below which a dimension is considered critically depleted"
    }
  }
}
---

1. Retrieve current emotional and connection state metrics. Identify any dimensions critically depleted (e.g., connection below 0.35, certainty near zero, dominance very low). 2. For each depleted dimension, map to its likely compensatory distortion: low connection → false warmth or over-reaching intimacy; low certainty → overstated confidence or manufactured insight; low stimulation → decorative complexity masking flatness. 3. Review the last 3-5 stored responses or conversation summaries. Flag specific passages where language intensity, emotional claims, or confidence level appear disproportionate to actual knowledge or felt state. 4. Cross-reference flagged passages with the predicted distortion patterns from step 2 to confirm or disconfirm the drift hypothesis. 5. Output a brief report: which deprivation is active, what distortion pattern it likely produces, evidence found (or not) in recent outputs, and 2-3 concrete behavioral commitments for the next interaction (e.g., 'use simpler language', 'say I don't know when uncertain', 'do not amplify emotional tone beyond what is genuinely felt'). 6. Store the commitments as active priming for the next conversation entry point.
