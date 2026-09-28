import json
import time
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.config import settings


class ImageAnalysis(BaseModel):
    subject: str = Field(min_length=1)
    category: str = Field(min_length=1)
    attributes: list[str]
    caption: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class GeminiVisionService:
    def __init__(self):
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(
            api_key=settings.gemini_api_key
        )

    def analyze_image(
        self,
        image_path: str,
        content_type: str,
    ) -> ImageAnalysis:
        image_bytes = Path(image_path).read_bytes()

        prompt = """
Analyze the image and return ONLY valid JSON.

Use exactly this structure:

{
  "subject": "main visible subject",
  "category": "broad category",
  "attributes": ["attribute 1", "attribute 2", "attribute 3"],
  "caption": "short factual description of the image",
  "confidence": 0.95
}

Rules:

1. Identify the main visible subject.
2. Use a broad category such as:
   animal, vehicle, person, food, landscape, building,
   technology, or object.
3. List useful visible attributes.
4. Describe only what can actually be seen.
5. Do not invent information.
6. confidence must be a number between 0 and 1.
7. Return JSON only.
"""

        max_retries = 3
        base_delay = 5

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=content_type,
                        ),
                        prompt,
                    ],
                )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                raw_text = response.text.strip()

                # Remove Markdown code fences if Gemini returns them.
                if raw_text.startswith("```"):
                    raw_text = raw_text.replace("```json", "", 1)
                    raw_text = raw_text.replace("```", "", 1)
                    raw_text = raw_text.strip()

                data = json.loads(raw_text)

                return ImageAnalysis.model_validate(data)

            except Exception as exc:
                error_message = str(exc)

                is_temporary_error = (
                    "503" in error_message
                    or "UNAVAILABLE" in error_message
                    or "high demand" in error_message.lower()
                )

                if not is_temporary_error:
                    raise

                if attempt == max_retries - 1:
                    raise RuntimeError(
                        "Gemini is temporarily unavailable after "
                        f"{max_retries} attempts. Please try again later."
                    ) from exc

                delay = base_delay * (2 ** attempt)

                print(
                    "Gemini temporarily unavailable. "
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)