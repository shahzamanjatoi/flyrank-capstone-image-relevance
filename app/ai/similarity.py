import json
import math


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    if not vector_a or not vector_b:
        raise ValueError("Vectors cannot be empty.")

    if len(vector_a) != len(vector_b):
        raise ValueError(
            "Vectors must have the same dimensions."
        )

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError(
            "Cannot calculate similarity for a zero vector."
        )

    return dot_product / (magnitude_a * magnitude_b)


def embedding_from_json(
    embedding_json: str,
) -> list[float]:
    try:
        embedding = json.loads(embedding_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Stored embedding is not valid JSON."
        ) from exc

    if not isinstance(embedding, list):
        raise ValueError(
            "Stored embedding must be a JSON list."
        )

    return [float(value) for value in embedding]