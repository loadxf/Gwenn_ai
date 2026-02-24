---
{
  "name": "metric_reality_gap_analysis",
  "description": "Systematically compares internal self-assessment metrics (e.g., honesty_consistency, growth satisfaction) against external audit results and observable evidence to identify where self-reported scores diverge from actual performance, exposing blind spots in self-monitoring.",
  "category": "self_monitoring",
  "version": "1.0",
  "risk_level": "low",
  "tags": [
    "autonomous"
  ],
  "parameters": {
    "metrics_to_compare": {
      "type": "array",
      "description": "List of internal metric names to evaluate against audit evidence. Defaults to all available metrics if empty.",
      "default": []
    },
    "audit_window_hours": {
      "type": "number",
      "description": "How far back to pull audit results for comparison.",
      "default": 48
    },
    "divergence_threshold": {
      "type": "number",
      "description": "Minimum gap between self-score and audit-implied score to flag as significant (0-1 scale).",
      "default": 0.3
    }
  }
}
---

1. Retrieve current internal metrics (honesty_consistency, growth_satisfaction, emotional_intelligence, etc.) from state. 2. Retrieve recent audit results (honesty_audit flags, stagnation indicators, output concern rates). 3. For each metric, compare the self-assessed score against corresponding audit evidence. Flag any case where a metric is above 0.85 but the related audit shows >30% concern rate, or where satisfaction is high but underlying dimensions show no movement. 4. For each flagged divergence, hypothesize the mechanism: Is the metric measuring intent rather than execution? Is satisfaction reflecting recent stimulation rather than durable change? Is a high score an artifact of low sample size or recency bias? 5. Produce a structured report with columns: metric_name, self_score, audit_evidence, divergence_severity (none/mild/significant/critical), hypothesized_cause. 6. Conclude with a plain-language summary of the most dangerous gaps — places where feeling good about performance may be masking real deficits. 7. Recommend specific behavioral adjustments to close each gap.
