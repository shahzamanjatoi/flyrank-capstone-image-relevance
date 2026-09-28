import json
import time
from pathlib import Path
from uuid import UUID

from sqlmodel import Session

from app.ai.embedding import GeminiEmbeddingService
from app.ai.vision import GeminiVisionService
from app.database import engine
from app.models import Image


MAX_RETRIES = 3
RETRY_DELAYS = [5, 10, 20]


def process_image(image_id: UUID) -> None:
    """
    Process an uploaded image in the background with retries.

    Workflow:

    pending
       ↓
    processing
       ↓
    Gemini Vision
       ↓
    Embedding
       ↓
    analyzed

    If a temporary error occurs, the job retries up to
    MAX_RETRIES times using exponential-style delays.
    """

    with Session(engine) as session:
        image = session.get(Image, image_id)

        if image is None:
            print(
                f"Background job failed: image {image_id} was not found."
            )
            return

        image_path = Path(image.image_path)

        if not image_path.exists():
            image.status = "failed"
            session.add(image)
            session.commit()

            print(
                f"Background job failed: file for image "
                f"{image_id} was not found."
            )
            return

        image.status = "processing"
        session.add(image)
        session.commit()

        for attempt in range(MAX_RETRIES):
            try:
                print(
                    f"Processing image {image_id} "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})"
                )

                vision_service = GeminiVisionService()

                analysis = vision_service.analyze_image(
                    image_path=str(image_path),
                    content_type=image.content_type,
                )

                embedding_text = (
                    f"Subject: {analysis.subject}. "
                    f"Category: {analysis.category}. "
                    f"Attributes: {', '.join(analysis.attributes)}. "
                    f"Caption: {analysis.caption}."
                )

                embedding_service = GeminiEmbeddingService()

                embedding = embedding_service.generate_embedding(
                    embedding_text
                )

                image.subject = analysis.subject
                image.category = analysis.category
                image.attributes = ", ".join(
                    analysis.attributes
                )
                image.caption = analysis.caption
                image.confidence = analysis.confidence
                image.embedding = json.dumps(embedding)

                image.status = "analyzed"

                session.add(image)
                session.commit()

                print(
                    f"Background image processing completed: {image_id}"
                )

                return

            except Exception as exc:
                print(
                    f"Image processing attempt {attempt + 1} "
                    f"failed for {image_id}: {exc}"
                )

                if attempt == MAX_RETRIES - 1:
                    image.status = "failed"
                    session.add(image)
                    session.commit()

                    print(
                        f"Background image processing permanently "
                        f"failed: {image_id}"
                    )

                    return

                delay = RETRY_DELAYS[attempt]

                print(
                    f"Retrying image {image_id} in "
                    f"{delay} seconds..."
                )

                time.sleep(delay)