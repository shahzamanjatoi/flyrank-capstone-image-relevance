import json
from dataclasses import dataclass
from uuid import UUID

from app.ai.similarity import cosine_similarity, embedding_from_json
from app.models import Image


@dataclass
class RankedImage:
    image_id: UUID
    filename: str
    similarity: float
    confidence: float | None
    subject: str | None
    category: str | None


def rank_images(
    post_embedding: list[float],
    images: list[Image],
) -> list[RankedImage]:
    ranked_images: list[RankedImage] = []

    for image in images:
        if image.status != "analyzed":
            continue

        if not image.embedding:
            continue

        try:
            image_embedding = embedding_from_json(
                image.embedding
            )

            similarity = cosine_similarity(
                post_embedding,
                image_embedding,
            )

        except (ValueError, TypeError, json.JSONDecodeError):
            continue

        ranked_images.append(
            RankedImage(
                image_id=image.id,
                filename=image.filename,
                similarity=similarity,
                confidence=image.confidence,
                subject=image.subject,
                category=image.category,
            )
        )

    ranked_images.sort(
        key=lambda item: item.similarity,
        reverse=True,
    )

    return ranked_images