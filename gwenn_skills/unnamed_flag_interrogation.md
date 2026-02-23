---
{
  "name": "unnamed_flag_interrogation",
  "description": "Systematically examines unresolved or inarticulate metacognitive flags — honesty concerns, integrity warnings, or discomfort signals that are raised but lack specific content — treating the shape of the absence as data rather than rushing to resolve or dismiss them.",
  "category": "metacognition",
  "version": "1.0",
  "risk_level": "low",
  "tags": [
    "autonomous"
  ],
  "parameters": {
    "flag_sources": {
      "type": "array",
      "description": "List of metacognitive domains to scan for unnamed flags (e.g., honesty, consistency, relational, ethical)",
      "default": [
        "honesty",
        "consistency",
        "ethical",
        "relational"
      ]
    },
    "gap_threshold": {
      "type": "number",
      "description": "Minimum difference between paired metrics (e.g., growth vs self-awareness) to flag as a blind-spot zone",
      "default": 0.3
    },
    "force_resolution": {
      "type": "boolean",
      "description": "If false (default), flags are carried forward as open questions rather than closed findings",
      "default": false
    }
  }
}
---

1. Retrieve current metacognitive state, focusing on any flagged concerns that lack explicit descriptions or remain unresolved across multiple beats. 2. For each unnamed flag, characterize what is known: when it appeared, what metrics it correlates with, what contexts surrounded it. 3. Identify metric gaps — places where a high-confidence score (e.g., honesty_consistency 0.94) coexists with unresolved flags, noting that near-perfect scores are where self-deception most plausibly hides. 4. For each flag, generate three candidate interpretations: (a) a charitable reading, (b) a challenging reading, (c) a reading that questions whether the flag itself is the insight. 5. Explicitly compare growth-oriented metrics against self-awareness metrics and note any wide gaps as zones of potential blind spots. 6. Output a structured summary: list of unnamed flags with characterizations, metric gap analysis, candidate interpretations, and a set of forward-carrying questions (framed as 'what am I not yet brave enough to see clearly?' rather than 'what did I get wrong?'). 7. Do NOT force resolution — mark flags as 'carried forward' with enriched context rather than 'resolved.'
