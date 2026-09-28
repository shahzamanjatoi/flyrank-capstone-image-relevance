from app.ai.matching import (
    DEFAULT_SIMILARITY_THRESHOLD,
    MIN_CONFIDENCE,
    evaluate_match,
)


def test_high_similarity_and_high_confidence_matches():
    decision = evaluate_match(
        similarity=0.85,
        confidence=0.95,
    )

    assert decision.matched is True
    assert decision.threshold == DEFAULT_SIMILARITY_THRESHOLD


def test_low_similarity_rejects():
    decision = evaluate_match(
        similarity=0.40,
        confidence=0.95,
    )

    assert decision.matched is False
    assert "below the required threshold" in decision.reason


def test_low_confidence_rejects():
    decision = evaluate_match(
        similarity=0.90,
        confidence=0.50,
    )

    assert decision.matched is False
    assert "confidence is too low" in decision.reason


def test_missing_confidence_rejects():
    decision = evaluate_match(
        similarity=0.90,
        confidence=None,
    )

    assert decision.matched is False
    assert "confidence is unavailable" in decision.reason


def test_similarity_exactly_at_threshold_matches():
    decision = evaluate_match(
        similarity=DEFAULT_SIMILARITY_THRESHOLD,
        confidence=MIN_CONFIDENCE,
    )

    assert decision.matched is True


def test_similarity_just_below_threshold_rejects():
    decision = evaluate_match(
        similarity=DEFAULT_SIMILARITY_THRESHOLD - 0.0001,
        confidence=MIN_CONFIDENCE,
    )

    assert decision.matched is False


def test_confidence_exactly_at_minimum_matches_if_similarity_is_high():
    decision = evaluate_match(
        similarity=0.90,
        confidence=MIN_CONFIDENCE,
    )

    assert decision.matched is True


def test_custom_threshold_is_respected():
    decision = evaluate_match(
        similarity=0.60,
        confidence=0.95,
        threshold=0.50,
    )

    assert decision.matched is True


def test_custom_threshold_rejects_below_threshold():
    decision = evaluate_match(
        similarity=0.49,
        confidence=0.95,
        threshold=0.50,
    )

    assert decision.matched is False
