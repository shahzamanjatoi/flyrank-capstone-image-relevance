from dataclasses import dataclass


DEFAULT_SIMILARITY_THRESHOLD = 0.70
MIN_CONFIDENCE = 0.60


@dataclass
class MatchDecision:
    matched: bool
    reason: str
    similarity: float
    threshold: float


def evaluate_match(
    similarity: float,
    confidence: float | None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> MatchDecision:
    """
    Decide whether content is a safe semantic match for an image.

    The decision uses:
    1. Semantic similarity.
    2. Vision confidence.
    3. A conservative similarity threshold.
    """

    if confidence is None:
        return MatchDecision(
            matched=False,
            reason="Image analysis confidence is unavailable.",
            similarity=similarity,
            threshold=threshold,
        )

    if confidence < MIN_CONFIDENCE:
        return MatchDecision(
            matched=False,
            reason=(
                f"Image analysis confidence is too low "
                f"({confidence:.2f})."
            ),
            similarity=similarity,
            threshold=threshold,
        )

    if similarity < threshold:
        return MatchDecision(
            matched=False,
            reason=(
                f"Semantic similarity ({similarity:.4f}) "
                f"is below the required threshold ({threshold:.2f})."
            ),
            similarity=similarity,
            threshold=threshold,
        )

    return MatchDecision(
        matched=True,
        reason=(
            f"Semantic similarity ({similarity:.4f}) "
            f"meets the required threshold ({threshold:.2f})."
        ),
        similarity=similarity,
        threshold=threshold,
    )