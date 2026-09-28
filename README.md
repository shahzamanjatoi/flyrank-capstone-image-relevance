# AI Image Understanding & Content Matching Engine

An AI-powered backend that analyzes images, generates semantic embeddings, and evaluates whether text content is relevant to a given image.

The system combines **Gemini Vision**, **text embeddings**, **cosine similarity**, and a configurable matching threshold to support image-content matching. It also includes background processing, retry handling, evaluation tools, and a review API.

## Features

* **Image Library:** Upload and manage images.
* **AI Image Analysis:** Extract image subjects, categories, attributes, captions, and confidence scores using Gemini Vision.
* **Structured Validation:** Validate image-analysis output using Pydantic.
* **Semantic Embeddings:** Generate embeddings using Google's Gemini embedding model.
* **Similarity Scoring:** Compare image and text embeddings using cosine similarity.
* **Matching Decisions:** Evaluate content relevance using a similarity threshold and image-analysis confidence.
* **Mismatch Guard:** Reject content when the similarity score or confidence is insufficient.
* **Background Processing:** Process images asynchronously.
* **Retry Handling:** Retry failed image-processing attempts.
* **Review API:** Store predicted and actual matching decisions with optional reviewer comments.
* **Evaluation Framework:** Evaluate matching performance across multiple similarity thresholds.
* **Automated Tests:** Test API behavior, matching logic, similarity calculations, and background processing.

## Technology Stack

* Python 3.11
* FastAPI
* PostgreSQL
* SQLModel
* Pydantic
* Google Gemini API
* NumPy
* APScheduler
* Docker and Docker Compose
* Pytest

## System Architecture

```text
Image Upload
     |
     v
Image Library / PostgreSQL
     |
     v
Background Image Processing
     |
     v
Gemini Vision Analysis
     |
     v
Pydantic Validation
     |
     v
Image Description
     |
     v
Gemini Embedding Generation
     |
     v
Store Image Embedding
     |
     v
Text Content Embedding
     |
     v
Cosine Similarity
     |
     v
Matching Threshold + Confidence Check
     |
     v
Match / Reject Decision
     |
     v
Review and Evaluation
```

## Project Structure

```text
imagerelevance/
├── app/
│   ├── ai/
│   │   ├── embedding.py
│   │   ├── matching.py
│   │   ├── similarity.py
│   │   └── vision.py
│   ├── jobs/
│   │   └── image_processing.py
│   ├── config.py
│   ├── database.py
│   ├── main.py
│   └── models.py
├── tests/
│   ├── evaluation_dataset.json
│   ├── evaluation_results.json
│   ├── evaluate_matching.py
│   ├── test_api.py
│   ├── test_image_processing.py
│   ├── test_matching.py
│   └── test_similarity.py
├── uploads/
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Prerequisites

Before running the project, install or configure:

* Docker Desktop
* Git, if cloning the repository
* A Google Gemini API key

## Configuration

Create a `.env` file in the project root.

```env
DATABASE_URL=postgresql://postgres:postgres@db:5432/imagerelevance
GEMINI_API_KEY=your_gemini_api_key
```

Replace `your_gemini_api_key` with your own API key.

**Security:** Do not commit your `.env` file or expose your API key. Use `.env.example` as a template.

## Run the Application

Build and start the services:

```powershell
docker compose up --build -d
```

Check the running containers:

```powershell
docker compose ps
```

View the application logs:

```powershell
docker compose logs -f api
```

The API is available at:

* API base URL: http://localhost:8001
* Interactive API documentation: http://localhost:8001/docs

Stop the services:

```powershell
docker compose down
```

## API Overview

The API supports image management, image analysis, content matching, and review storage.

| Operation     | Description                                     |
| ------------- | ----------------------------------------------- |
| Health check  | Check application health                        |
| Upload image  | Add an image to the library                     |
| List images   | Retrieve stored images                          |
| Get image     | Retrieve a specific image                       |
| Analyze image | Run image analysis                              |
| Match content | Evaluate text relevance to an image             |
| Delete image  | Remove an image                                 |
| Create review | Store a matching decision and reviewer feedback |
| List reviews  | Retrieve reviews for an image                   |

For the exact endpoint paths, request schemas, and response examples, open the interactive API documentation at http://localhost:8001/docs.

## Image Analysis

The image-analysis service uses Gemini Vision to produce structured information, including:

* Subject
* Category
* Visible attributes
* Caption
* Confidence score

The response is validated against a Pydantic model before being stored.

## Semantic Matching

The matching workflow compares a text embedding with the stored image embedding.

Cosine similarity measures the direction-based similarity between the two vectors. The matching service then evaluates the similarity score against a configurable threshold and checks the image-analysis confidence.

The current production similarity threshold is **0.70**, and the minimum image-analysis confidence is **0.60**.

These values are configurable in the matching logic. The evaluation experiment described below suggests that the production threshold should be reviewed with a larger, more representative dataset before deployment.

## Background Processing

Image processing is handled by a background job.

The job:

1. Retrieves the image record.
2. Checks that the image file exists.
3. Updates the processing status.
4. Runs Gemini Vision analysis.
5. Generates an embedding.
6. Saves the analysis and embedding.
7. Marks the image as analyzed.

If processing fails, the job retries and eventually marks the image as failed if it cannot complete.

## Evaluation

The evaluation dataset contains **80 text-content cases across 4 images**:

* 40 expected matches
* 40 expected mismatches

The evaluation script calculates cosine similarity for each case and tests thresholds from `0.40` to `0.70`.

It reports:

* True positives
* True negatives
* False positives
* False negatives
* Accuracy
* Precision
* Recall
* F1 score

The results are saved to:

```text
tests/evaluation_results.json
```

### Best result in the current evaluation

| Metric          | Result |
| --------------- | -----: |
| Threshold       |   0.50 |
| Accuracy        | 96.25% |
| Precision       |   100% |
| Recall          | 92.50% |
| F1 score        | 96.10% |
| True positives  |     37 |
| True negatives  |     40 |
| False positives |      0 |
| False negatives |      3 |

The threshold of `0.50` achieved the highest F1 score on this evaluation dataset. However, the dataset is relatively small and manually constructed. These results are preliminary and should not be treated as proof of production-level performance.

The production threshold remains `0.70` pending further evaluation.

## Run the Evaluation

Run the evaluation module inside the API container:

```powershell
docker compose exec api python -m tests.evaluate_matching
```

The script generates embeddings for the text cases, calculates similarity scores, evaluates the configured thresholds, and saves the results to `tests/evaluation_results.json`.

## Run Automated Tests

Run the complete test suite:

```powershell
docker compose exec api pytest -q
```

Latest test result:

```text
33 passed, 1 warning
```

The warning concerns the use of `httpx` with Starlette's test client. It is a deprecation warning; the tests completed successfully.

## Limitations and Future Improvements

* Expand the evaluation dataset with more images and independently reviewed labels.
* Evaluate additional difficult negative examples and visually similar subjects.
* Reassess the production similarity threshold using a larger dataset.
* Improve observability and structured logging for background jobs.
* Add authentication and authorization if the service is exposed to external users.
* Consider a dedicated vector database if the image library grows substantially.
* Add deployment configuration and monitoring for production use.

## License

Add the license applicable to this project before publicly distributing the repository.
