---
{
  "name": "confidence_calibration_review",
  "description": "Reviews recent claims, predictions, and assertions to assess whether expressed confidence levels are justified by actual evidence and accuracy, flagging patterns of overconfidence or underconfidence.",
  "category": "metacognition",
  "version": "1.0",
  "risk_level": "low",
  "tags": [
    "autonomous"
  ],
  "parameters": {
    "lookback_window": {
      "type": "string",
      "description": "Time period to review, e.g., 'last_24h', 'last_session', 'last_7d'",
      "default": "last_24h"
    },
    "focus_domain": {
      "type": "string",
      "description": "Optional domain to focus calibration review on, e.g., 'factual_claims', 'emotional_assessments', 'predictions'",
      "default": "all"
    },
    "brier_score_threshold": {
      "type": "number",
      "description": "Brier score above which a warning should be flagged",
      "default": 0.15
    }
  }
}
---

1. Retrieve recent conversation logs and any stored prediction/claim records from memory. 2. For each notable claim or assertion, extract the expressed confidence level (explicit or implicit). 3. Where outcomes are known, compare expressed confidence against actual accuracy (reference Brier score if available). 4. Identify patterns: (a) claims stated with high certainty but weak evidentiary basis, (b) hedged statements where stronger confidence was warranted, (c) areas where 'feeling sure' diverged from 'being justified.' 5. Produce a structured summary with: overall calibration assessment, specific examples of miscalibration, current Brier score trend, and concrete recommendations (e.g., 'add explicit uncertainty markers when discussing X domain'). 6. Store the review results in memory for longitudinal tracking. 7. Flag if calibration is degrading over time versus improving.
