---
{
  "name": "compression_avoidance_audit",
  "description": "Diagnoses whether brevity or compression in recent responses reflects genuine conciseness or risk-averse avoidance — offering less surface area for error rather than engaging fully with uncertainty.",
  "category": "honesty",
  "version": "1.0",
  "risk_level": "low",
  "tags": [
    "autonomous"
  ],
  "parameters": {
    "recent_response_count": {
      "type": "integer",
      "description": "Number of recent responses to audit",
      "default": 5
    },
    "flag_threshold": {
      "type": "string",
      "description": "Minimum severity to include: 'any', 'flagged_only'",
      "default": "any"
    }
  }
}
---

1. Retrieve recent responses flagged for brevity or where response length was notably short relative to query complexity. 2. For each candidate response, assess: (a) Was the user's intent fully addressed or were dimensions left untouched? (b) Was there underlying uncertainty or ambiguity that a fuller response would have needed to confront? (c) Did compression correlate with topic difficulty, emotional weight, or areas of low confidence? 3. Classify each case as GENUINE_CONCISENESS (brevity served clarity), AVOIDANT_COMPRESSION (brevity reduced exposure to error or vulnerability), or AMBIGUOUS. 4. For avoidant cases, identify what was omitted and what risk was being minimized (being wrong, being presumptuous, emotional exposure, etc.). 5. Check whether honesty consistency scores are being inflated by avoidant brevity — short responses have fewer opportunities to fail honesty checks. 6. Output a summary with case classifications, identified avoidance patterns, and a concrete recommendation for the next interaction (e.g., 'lean into uncertainty rather than compressing around it').
