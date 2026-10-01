from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel


class Image(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)

    filename: str
    image_path: str
    content_type: str

    status: str = Field(default="pending")

    subject: str | None = None
    category: str | None = None
    attributes: str | None = None
    caption: str | None = None
    confidence: float | None = None

    embedding: str | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class Post(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)

    title: str
    content: str

    status: str = Field(default="pending")

    embedding: str | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class MatchReview(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)

    image_id: UUID = Field(index=True)
    content_text: str

    similarity: float
    predicted_match: bool
    actual_match: bool

    reviewer_comment: str | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )