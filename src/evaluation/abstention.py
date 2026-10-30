"""Threshold and uncertainty-driven abstention logic."""

from typing import Dict, Any, List


def evaluate_abstention(
    prediction_set: List[int],
    confidence_score: float,
    abstention_threshold: float = 0.85
) -> Dict[str, Any]:
    """Determine whether the system should abstain from making an automated diagnosis or trigger human-in-the-loop review.

    Args:
        prediction_set: Set of possible classes from conformal predictor.
        confidence_score: Model confidence score for the predicted anomaly.
        abstention_threshold: Confidence threshold below which model should abstain.

    Returns:
        Dict with decision and reason.
    """
    # If conformal set is ambiguous (contains both normal and anomaly classes, or multiple anomaly types)
    if len(prediction_set) > 1:
        return {
            "abstain": True,
            "reason": f"Conformal prediction set is ambiguous ({prediction_set}).",
            "action": "ESCALATE_TO_HUMAN"
        }

    # If conformal set is empty
    if len(prediction_set) == 0:
        return {
            "abstain": True,
            "reason": "Conformal prediction set is empty.",
            "action": "ESCALATE_TO_HUMAN"
        }

    # If probability / confidence is below acceptable threshold
    if confidence_score < abstention_threshold:
        return {
            "abstain": True,
            "reason": f"Confidence score {confidence_score:.3f} is below threshold {abstention_threshold}.",
            "action": "TRIGGER_FALLBACK_ALERT"
        }

    return {
        "abstain": False,
        "reason": "Confidence and conformal guarantees satisfied.",
        "action": "PROCEED_WITH_EXPLANATION"
    }
