import pytest

from app.ai.similarity import (
    cosine_similarity,
    embedding_from_json,
)


def test_cosine_similarity_identical_vectors():
    result = cosine_similarity(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    )

    assert result == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    result = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert result == pytest.approx(0.0)


def test_cosine_similarity_empty_first_vector():
    with pytest.raises(
        ValueError,
        match="Vectors cannot be empty.",
    ):
        cosine_similarity(
            [],
            [1.0, 2.0],
        )


def test_cosine_similarity_empty_second_vector():
    with pytest.raises(
        ValueError,
        match="Vectors cannot be empty.",
    ):
        cosine_similarity(
            [1.0, 2.0],
            [],
        )


def test_cosine_similarity_dimension_mismatch():
    with pytest.raises(
        ValueError,
        match="Vectors must have the same dimensions.",
    ):
        cosine_similarity(
            [1.0, 2.0],
            [1.0, 2.0, 3.0],
        )


def test_cosine_similarity_zero_first_vector():
    with pytest.raises(
        ValueError,
        match="Cannot calculate similarity for a zero vector.",
    ):
        cosine_similarity(
            [0.0, 0.0],
            [1.0, 2.0],
        )


def test_cosine_similarity_zero_second_vector():
    with pytest.raises(
        ValueError,
        match="Cannot calculate similarity for a zero vector.",
    ):
        cosine_similarity(
            [1.0, 2.0],
            [0.0, 0.0],
        )


def test_embedding_from_valid_json():
    result = embedding_from_json(
        "[1.0, 2.5, -3.0]"
    )

    assert result == [1.0, 2.5, -3.0]


def test_embedding_from_invalid_json():
    with pytest.raises(
        ValueError,
        match="Stored embedding is not valid JSON.",
    ):
        embedding_from_json(
            "not valid json"
        )


def test_embedding_from_json_non_list():
    with pytest.raises(
        ValueError,
        match="Stored embedding must be a JSON list.",
    ):
        embedding_from_json(
            '{"value": 1}'
        )


def test_embedding_from_json_converts_numbers():
    result = embedding_from_json(
        '["1.5", "2.5", "3.5"]'
    )

    assert result == [1.5, 2.5, 3.5]
