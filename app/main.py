from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4
import json

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlmodel import Session, select, text

from app.ai.embedding import GeminiEmbeddingService
from app.ai.matching import evaluate_match
from app.ai.similarity import cosine_similarity, embedding_from_json
from app.ai.ranking import rank_images
from app.database import create_db_and_tables, engine, get_session
from app.jobs.image_processing import process_image
from app.models import Image, MatchReview, Post


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

MAX_FILE_SIZE = 10 * 1024 * 1024

MATCH_THRESHOLD = 0.70


scheduler = BackgroundScheduler()


class ReviewCreate(BaseModel):
    content_text: str = Field(min_length=1)
    similarity: float = Field(ge=-1.0, le=1.0)
    predicted_match: bool
    actual_match: bool
    reviewer_comment: str | None = None


class PostCreate(BaseModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    UPLOAD_DIR.mkdir(exist_ok=True)

    scheduler.start()

    yield

    scheduler.shutdown()


app = FastAPI(
    title="AI Image Understanding & Content Matching Engine",
    version="0.9.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": "AI Image Understanding & Content Matching Engine API",
        "status": "running",
    }


@app.get("/health")
def health():
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    return {
        "status": "healthy",
        "database": "connected",
    }


@app.post("/images", status_code=status.HTTP_201_CREATED)
async def upload_image(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported image type. Use JPEG, PNG, or WebP.",
        )

    file_data = await file.read()

    if not file_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(file_data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image file is too large. Maximum size is 10 MB.",
        )

    image_id = uuid4()

    extension = Path(file.filename or "").suffix.lower()

    if not extension:
        extension = ".img"

    stored_filename = f"{image_id}{extension}"
    image_path = UPLOAD_DIR / stored_filename

    image_path.write_bytes(file_data)

    image = Image(
        id=image_id,
        filename=file.filename or stored_filename,
        image_path=str(image_path),
        content_type=file.content_type,
    )

    session.add(image)
    session.commit()
    session.refresh(image)

    return image


@app.get("/images")
def list_images(
    session: Session = Depends(get_session),
):
    statement = select(Image).order_by(Image.created_at.desc())

    images = session.exec(statement).all()

    return images


@app.get("/images/{image_id}")
def get_image(
    image_id: UUID,
    session: Session = Depends(get_session),
):
    image = session.get(Image, image_id)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found.",
        )

    return image


@app.post("/images/{image_id}/analyze")
def analyze_image(
    image_id: UUID,
    session: Session = Depends(get_session),
):
    image = session.get(Image, image_id)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found.",
        )

    if image.status == "processing":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Image processing is already in progress.",
        )

    if image.status == "analyzed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Image has already been analyzed.",
        )

    image_path = Path(image.image_path)

    if not image_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image file not found.",
        )

    image.status = "processing"

    session.add(image)
    session.commit()

    try:
        scheduler.add_job(
            process_image,
            args=[image_id],
            id=f"process-image-{image_id}",
            replace_existing=False,
        )
    except Exception as exc:
        image.status = "failed"
        session.add(image)
        session.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not queue image processing job: {exc}",
        ) from exc

    return {
        "message": "Image processing job queued.",
        "image_id": image_id,
        "status": "processing",
    }


@app.post("/posts", status_code=status.HTTP_201_CREATED)
def create_post(
    post_data: PostCreate,
    session: Session = Depends(get_session),
):
    title = post_data.title.strip()
    content = post_data.content.strip()

    if not title:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Post title cannot be empty.",
        )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Post content cannot be empty.",
        )

    embedding_text = f"Title: {title}. Content: {content}."

    try:
        embedding_service = GeminiEmbeddingService()

        embedding = embedding_service.generate_embedding(
            embedding_text
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Post embedding failed: {exc}",
        ) from exc

    post = Post(
        id=uuid4(),
        title=title,
        content=content,
        status="embedded",
        embedding=json.dumps(embedding),
    )

    session.add(post)
    session.commit()
    session.refresh(post)

    return post


@app.get("/posts")
def list_posts(
    session: Session = Depends(get_session),
):
    statement = select(Post).order_by(Post.created_at.desc())

    posts = session.exec(statement).all()

    return posts


@app.get("/posts/{post_id}")
def get_post(
    post_id: UUID,
    session: Session = Depends(get_session),
):
    post = session.get(Post, post_id)

    if post is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found.",
        )

    return post


@app.get("/posts/{post_id}/images")
def rank_post_images(
    post_id: UUID,
    session: Session = Depends(get_session),
):
    post = session.get(Post, post_id)

    if post is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found.",
        )

    if not post.embedding:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Post has not been embedded yet.",
        )

    try:
        post_embedding = embedding_from_json(
            post.embedding
        )

        statement = (
            select(Image)
            .where(Image.status == "analyzed")
            .order_by(Image.created_at.desc())
        )

        images = session.exec(statement).all()

        ranked_images = rank_images(
            post_embedding=post_embedding,
            images=images,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image ranking failed: {exc}",
        ) from exc

    return {
        "post_id": post.id,
        "title": post.title,
        "content": post.content,
        "results": [
            {
                "rank": index + 1,
                "image_id": image.image_id,
                "filename": image.filename,
                "similarity": round(image.similarity, 4),
                "confidence": image.confidence,
                "subject": image.subject,
                "category": image.category,
            }
            for index, image in enumerate(ranked_images)
        ],
    }


@app.post("/images/{image_id}/match")
def match_image(
    image_id: UUID,
    content: str,
    session: Session = Depends(get_session),
):
    image = session.get(Image, image_id)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found.",
        )

    if not image.embedding:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image has not been embedded yet. Analyze the image first.",
        )

    if not content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content cannot be empty.",
        )

    try:
        embedding_service = GeminiEmbeddingService()

        content_embedding = embedding_service.generate_embedding(
            content
        )

        image_embedding = embedding_from_json(
            image.embedding
        )

        similarity = cosine_similarity(
            content_embedding,
            image_embedding,
        )

        decision = evaluate_match(
            similarity=similarity,
            confidence=image.confidence,
            threshold=MATCH_THRESHOLD,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Content matching failed: {exc}",
        ) from exc

    return {
        "image_id": image.id,
        "filename": image.filename,
        "content": content,
        "similarity": round(decision.similarity, 4),
        "threshold": decision.threshold,
        "image_confidence": image.confidence,
        "matched": decision.matched,
        "decision": "match" if decision.matched else "reject",
        "reason": decision.reason,
    }


@app.post(
    "/images/{image_id}/reviews",
    status_code=status.HTTP_201_CREATED,
)
def create_review(
    image_id: UUID,
    review: ReviewCreate,
    session: Session = Depends(get_session),
):
    image = session.get(Image, image_id)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found.",
        )

    if not review.content_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content text cannot be empty.",
        )

    new_review = MatchReview(
        image_id=image_id,
        content_text=review.content_text,
        similarity=review.similarity,
        predicted_match=review.predicted_match,
        actual_match=review.actual_match,
        reviewer_comment=review.reviewer_comment,
    )

    session.add(new_review)
    session.commit()
    session.refresh(new_review)

    return new_review


@app.get("/images/{image_id}/reviews")
def list_reviews(
    image_id: UUID,
    session: Session = Depends(get_session),
):
    image = session.get(Image, image_id)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found.",
        )

    statement = (
        select(MatchReview)
        .where(MatchReview.image_id == image_id)
        .order_by(MatchReview.created_at.desc())
    )

    reviews = session.exec(statement).all()

    return reviews


@app.delete("/images/{image_id}")
def delete_image(
    image_id: UUID,
    session: Session = Depends(get_session),
):
    image = session.get(Image, image_id)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found.",
        )

    image_path = Path(image.image_path)

    if image_path.exists():
        image_path.unlink()

    session.delete(image)
    session.commit()

    return {
        "message": "Image deleted successfully.",
        "image_id": image_id,
    }