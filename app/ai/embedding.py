from google import genai

from app.config import settings


EMBEDDING_MODEL = "gemini-embedding-2"


class GeminiEmbeddingService:
    def __init__(self):
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(
            api_key=settings.gemini_api_key
        )

    def generate_embedding(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Text for embedding cannot be empty.")

        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )

        if not response.embeddings:
            raise RuntimeError("Gemini returned no embedding.")

        return response.embeddings[0].values