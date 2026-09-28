from types import SimpleNamespace
from uuid import uuid4

from app.jobs import image_processing


def test_process_image_nonexistent_image(monkeypatch):
    image_id = uuid4()

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def get(self, model, requested_id):
            return None

    monkeypatch.setattr(
        image_processing,
        "Session",
        lambda engine: FakeSession(),
    )

    image_processing.process_image(image_id)


def test_process_image_missing_file(monkeypatch, tmp_path):
    image_id = uuid4()

    image = SimpleNamespace(
        id=image_id,
        image_path=str(tmp_path / "missing.jpg"),
        content_type="image/jpeg",
        status="pending",
    )

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def get(self, model, requested_id):
            return image

        def add(self, obj):
            pass

        def commit(self):
            pass

    monkeypatch.setattr(
        image_processing,
        "Session",
        lambda engine: FakeSession(),
    )

    image_processing.process_image(image_id)

    assert image.status == "failed"


def test_process_image_success(monkeypatch, tmp_path):
    image_id = uuid4()

    image_file = tmp_path / "test.jpg"
    image_file.write_bytes(b"fake image data")

    image = SimpleNamespace(
        id=image_id,
        image_path=str(image_file),
        content_type="image/jpeg",
        status="pending",
        subject=None,
        category=None,
        attributes=None,
        caption=None,
        confidence=None,
        embedding=None,
    )

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def get(self, model, requested_id):
            return image

        def add(self, obj):
            pass

        def commit(self):
            pass

    class FakeVisionService:
        def __init__(self):
            pass

        def analyze_image(self, image_path, content_type):
            return SimpleNamespace(
                subject="cat",
                category="animal",
                attributes=["black", "sitting"],
                caption="A black cat sitting outside.",
                confidence=0.98,
            )

    class FakeEmbeddingService:
        def __init__(self):
            pass

        def generate_embedding(self, text):
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        image_processing,
        "Session",
        lambda engine: FakeSession(),
    )

    monkeypatch.setattr(
        image_processing,
        "GeminiVisionService",
        FakeVisionService,
    )

    monkeypatch.setattr(
        image_processing,
        "GeminiEmbeddingService",
        FakeEmbeddingService,
    )

    image_processing.process_image(image_id)

    assert image.status == "analyzed"
    assert image.subject == "cat"
    assert image.category == "animal"
    assert image.attributes == "black, sitting"
    assert image.caption == "A black cat sitting outside."
    assert image.confidence == 0.98
    assert image.embedding == "[0.1, 0.2, 0.3]"


def test_process_image_retries_after_failure(monkeypatch, tmp_path):
    image_id = uuid4()

    image_file = tmp_path / "test.jpg"
    image_file.write_bytes(b"fake image data")

    image = SimpleNamespace(
        id=image_id,
        image_path=str(image_file),
        content_type="image/jpeg",
        status="pending",
        subject=None,
        category=None,
        attributes=None,
        caption=None,
        confidence=None,
        embedding=None,
    )

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def get(self, model, requested_id):
            return image

        def add(self, obj):
            pass

        def commit(self):
            pass

    class FakeVisionService:
        attempts = 0

        def __init__(self):
            pass

        def analyze_image(self, image_path, content_type):
            FakeVisionService.attempts += 1

            if FakeVisionService.attempts == 1:
                raise RuntimeError("Temporary Gemini error")

            return SimpleNamespace(
                subject="cat",
                category="animal",
                attributes=["black"],
                caption="A black cat.",
                confidence=0.98,
            )

    class FakeEmbeddingService:
        def __init__(self):
            pass

        def generate_embedding(self, text):
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        image_processing,
        "Session",
        lambda engine: FakeSession(),
    )

    monkeypatch.setattr(
        image_processing,
        "GeminiVisionService",
        FakeVisionService,
    )

    monkeypatch.setattr(
        image_processing,
        "GeminiEmbeddingService",
        FakeEmbeddingService,
    )

    monkeypatch.setattr(
        image_processing.time,
        "sleep",
        lambda seconds: None,
    )

    image_processing.process_image(image_id)

    assert FakeVisionService.attempts == 2
    assert image.status == "analyzed"


def test_process_image_fails_after_max_retries(monkeypatch, tmp_path):
    image_id = uuid4()

    image_file = tmp_path / "test.jpg"
    image_file.write_bytes(b"fake image data")

    image = SimpleNamespace(
        id=image_id,
        image_path=str(image_file),
        content_type="image/jpeg",
        status="pending",
        subject=None,
        category=None,
        attributes=None,
        caption=None,
        confidence=None,
        embedding=None,
    )

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def get(self, model, requested_id):
            return image

        def add(self, obj):
            pass

        def commit(self):
            pass

    class FakeVisionService:
        attempts = 0

        def __init__(self):
            pass

        def analyze_image(self, image_path, content_type):
            FakeVisionService.attempts += 1
            raise RuntimeError("Gemini permanently unavailable")

    monkeypatch.setattr(
        image_processing,
        "Session",
        lambda engine: FakeSession(),
    )

    monkeypatch.setattr(
        image_processing,
        "GeminiVisionService",
        FakeVisionService,
    )

    monkeypatch.setattr(
        image_processing.time,
        "sleep",
        lambda seconds: None,
    )

    image_processing.process_image(image_id)

    assert FakeVisionService.attempts == 3
    assert image.status == "failed"
