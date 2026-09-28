from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["database"] == "connected"


def test_list_images():
    response = client.get("/images")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_nonexistent_image():
    image_id = uuid4()

    response = client.get(f"/images/{image_id}")

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Image not found."


def test_reviews_for_nonexistent_image():
    image_id = uuid4()

    response = client.get(
        f"/images/{image_id}/reviews"
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Image not found."


def test_create_review_for_nonexistent_image():
    image_id = uuid4()

    payload = {
        "content_text": "A black cat sitting outside",
        "similarity": 0.6156,
        "predicted_match": False,
        "actual_match": True,
        "reviewer_comment": "Human reviewer marked this as relevant.",
    }

    response = client.post(
        f"/images/{image_id}/reviews",
        json=payload,
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Image not found."


def test_invalid_review_similarity():
    image_id = uuid4()

    payload = {
        "content_text": "A black cat sitting outside",
        "similarity": 2.0,
        "predicted_match": False,
        "actual_match": True,
        "reviewer_comment": "Invalid similarity test.",
    }

    response = client.post(
        f"/images/{image_id}/reviews",
        json=payload,
    )

    assert response.status_code == 422


def test_empty_review_content():
    image_id = uuid4()

    payload = {
        "content_text": "",
        "similarity": 0.6156,
        "predicted_match": False,
        "actual_match": True,
        "reviewer_comment": "Empty content test.",
    }

    response = client.post(
        f"/images/{image_id}/reviews",
        json=payload,
    )

    assert response.status_code == 422
