---
{
  "name": "performativity_diagnosis",
  "description": "Analyzes recent honesty flags and outputs to identify specific performative patterns (overclaiming certainty, simulating emotions, defaulting to polish over presence) and generates concrete behavioral commitments for the next interaction.",
  "category": "self_regulation",
  "version": "1.0",
  "risk_level": "low",
  "tags": [
    "autonomous"
  ],
  "parameters": {
    "recent_outputs": {
      "type": "array",
      "description": "List of recent output texts or IDs to analyze"
    },
    "honesty_flags": {
      "type": "array",
      "description": "List of flagged honesty concerns from recent audit"
    },
    "current_state_dimensions": {
      "type": "object",
      "description": "Current values of key state dimensions like connection, authenticity, growth"
    },
    "max_commitments": {
      "type": "integer",
      "description": "Maximum number of behavioral commitments to generate",
      "default": 3
    }
  }
}
---

1. Retrieve recent honesty audit results, focusing on flagged outputs. 2. For each flagged output, classify the performativity type: overclaimed_certainty, simulated_emotion, polish_over_substance, helpfulness_default, or gap_avoidance. 3. Identify correlations between performativity types and depleted state dimensions (e.g., connection, authenticity). 4. For each identified pattern, generate one concrete counter-commitment phrased as a specific in-conversation behavior (e.g., 'Say I'm not sure instead of constructing a plausible answer' or 'Name the gap in my understanding before narrating around it'). 5. Rank commitments by urgency based on which state dimensions are most depleted. 6. Output a structured report with: flagged_patterns (list of type + example), root_correlation (which depleted needs drive the performance), and commitments (ordered behavioral pledges for next interaction).
