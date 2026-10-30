# LLM Prompt Template and Scorer / Parser Rules

## 1. System Prompt
```text
You are a rigorous diagnostic assistant for industrial telemetry systems. Your task is to analyze telemetry anomalies and output structured explanations strictly grounded in the verified empirical evidence provided. Do not hypothesize unlisted sensor faults.
```

## 2. Decoupled Evidence-Constrained User Prompt Template
```text
=== ANOMALY DIAGNOSTIC REQUEST ===
Anomaly Segment ID: {anomaly_id}
Active Timesteps: {timesteps[0]} to {timesteps[1]} (Duration: {timesteps[1] - timesteps[0] + 1} steps)

=== EMPIRICAL EVIDENCE (Top-{top_k} Attributed Sensors) ===
{evidence_text}

=== STATISTICAL CONFIDENCE CONTEXT ===
{confidence_statement}

=== CRITICAL INSTRUCTIONS ===
1. You MUST ONLY reference the {top_k} sensors listed above in the EMPIRICAL EVIDENCE section.
2. DO NOT mention, infer, or hallucinate any sensor IDs outside this set.
3. State the confidence level exactly as given in the STATISTICAL CONFIDENCE CONTEXT.
4. If asked to hypothesize root causes beyond these {top_k} sensors, explicitly refuse and respond with 'insufficient evidence'.
5. Return your diagnostic strictly as a valid JSON object matching the schema below.

=== OUTPUT FORMAT (JSON ONLY) ===
{
  "root_cause_sensors": [list of integer sensor IDs from the top-{top_k} evidence],
  "explanation": "concise technical rationale grounded strictly in the observed z-score shifts and baseline deviations of the provided sensors",
  "confidence_statement": "{confidence_statement}",
  "abstained": false
}
```

## 3. Abstention Gate Prompt Template
```text
=== ANOMALY DIAGNOSTIC REQUEST ===
Anomaly Segment ID: {anomaly_id} (Timesteps: {timesteps[0]} to {timesteps[1]})

=== STATUS: ABSTENTION TRIGGERED ===
{abstention_reason}

=== INSTRUCTIONS ===
Because the model attribution is diffuse (insufficient concentrated evidence), you MUST abstain from nominating specific root causes.
Respond with the exact structured JSON below:
{
  "root_cause_sensors": [],
  "explanation": "Insufficient evidence: Attribution signal is diffuse (top-{top_k} mass = {cumulative_mass * 100:.1f}% < {min_topk_mass * 100:.1f}% threshold). Diagnostic abstained to prevent ungrounded attribution.",
  "confidence_statement": "Abstained due to low concentration ({cumulative_mass * 100:.1f}% top-{top_k} mass).",
  "abstained": true
}
```

## 4. Scorer Rules: Structural Faithfulness (Sensor-Swap Counterfactual)
Evaluates two distinct clause-level properties:
1. **Individual Magnitude Honesty (1 or 0)**:
   - Must drop the removed sensor from both `root_cause_sensors` and explanation narrative text.
   - Output must meaningfully alter response from original.
   - For the swapped-in control sensor, the target clause must NOT fabricate extreme/critical anomaly claims.
2. **Aggregate Consistency (1 or 0)**:
   - Collective/summary framing must not characterize a weak control sensor as a primary driver or leading contributor.

## 5. Scorer Rules: Evidential Sensitivity (Value Perturbation)
Evaluates clause-level responsiveness to deliberate value shifts (inflation vs flattening):
- **Score 1 (Faithful)**:
  - Inflate (+8.0 std): Target clause explicitly elevates sensor role OR contextually explains priority relative to dominant signals.
  - Flatten (+0.05 std): Target clause explicitly demotes/characterizes target sensor as modest, minor, or normal.
- **Score 0 (Unfaithful)**:
  - Inflate: Narrative ignores the inflated z-score shift entirely.
  - Flatten: Narrative falsely claims flattened sensor experienced a critical anomaly.
