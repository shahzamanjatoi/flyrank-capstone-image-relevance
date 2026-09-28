import json
from pathlib import Path
from uuid import UUID

from sqlmodel import Session

from app.ai.embedding import GeminiEmbeddingService
from app.ai.similarity import cosine_similarity, embedding_from_json
from app.database import engine
from app.models import Image


DATASET_PATH = Path(__file__).parent / "evaluation_dataset.json"
RESULTS_PATH = Path(__file__).parent / "evaluation_results.json"

THRESHOLDS = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


def load_dataset() -> list[dict]:
    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def calculate_metrics(
    results: list[dict],
    threshold: float,
) -> dict:
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for result in results:
        expected = result["expected_match"]

        predicted = (
            result["similarity"] >= threshold
        )

        if expected and predicted:
            true_positive += 1

        elif not expected and not predicted:
            true_negative += 1

        elif not expected and predicted:
            false_positive += 1

        elif expected and not predicted:
            false_negative += 1

    total = (
        true_positive
        + true_negative
        + false_positive
        + false_negative
    )

    accuracy = (
        (true_positive + true_negative) / total
        if total
        else 0.0
    )

    precision = (
        true_positive
        / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0.0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "threshold": threshold,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def save_results(
    dataset: list[dict],
    similarity_results: list[dict],
    evaluation_results: list[dict],
    best_result: dict,
) -> None:
    output = {
        "dataset": {
            "total_cases": len(dataset),
            "expected_matches": sum(
                item["expected_match"]
                for item in dataset
            ),
            "expected_mismatches": sum(
                not item["expected_match"]
                for item in dataset
            ),
            "image_count": len(
                set(item["image_id"] for item in dataset)
            ),
        },
        "thresholds": THRESHOLDS,
        "results": similarity_results,
        "threshold_evaluation": evaluation_results,
        "best_f1": best_result,
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )


def main() -> None:
    dataset = load_dataset()

    embedding_service = GeminiEmbeddingService()

    results = []

    print("=" * 60)
    print("GENERATING EMBEDDINGS")
    print("=" * 60)

    with Session(engine) as session:
        for index, item in enumerate(
            dataset,
            start=1,
        ):
            image_id = UUID(item["image_id"])
            content = item["content"]
            expected_match = item["expected_match"]

            image = session.get(
                Image,
                image_id,
            )

            if image is None:
                print(
                    f"[{index}/{len(dataset)}] "
                    f"Image not found: {image_id}"
                )
                continue

            if not image.embedding:
                print(
                    f"[{index}/{len(dataset)}] "
                    f"Image has no embedding: "
                    f"{image.filename}"
                )
                continue

            print(
                f"[{index}/{len(dataset)}] "
                f"Generating embedding for: {content}"
            )

            content_embedding = (
                embedding_service.generate_embedding(
                    content
                )
            )

            image_embedding = embedding_from_json(
                image.embedding
            )

            similarity = cosine_similarity(
                content_embedding,
                image_embedding,
            )

            results.append(
                {
                    "image_id": str(image_id),
                    "filename": image.filename,
                    "content": content,
                    "expected_match": expected_match,
                    "similarity": similarity,
                }
            )

    print()
    print("=" * 60)
    print("SIMILARITY RESULTS")
    print("=" * 60)

    for result in results:
        print(
            f"{result['similarity']:.4f} | "
            f"Expected: {result['expected_match']} | "
            f"{result['content']}"
        )

    print()
    print("=" * 60)
    print("THRESHOLD EVALUATION")
    print("=" * 60)

    evaluation_results = []

    for threshold in THRESHOLDS:
        metrics = calculate_metrics(
            results,
            threshold,
        )

        evaluation_results.append(metrics)

        print(
            f"\nThreshold: {threshold:.2f}"
        )

        print(
            f"  TP: {metrics['true_positive']} | "
            f"TN: {metrics['true_negative']} | "
            f"FP: {metrics['false_positive']} | "
            f"FN: {metrics['false_negative']}"
        )

        print(
            f"  Accuracy:  {metrics['accuracy']:.4f}"
        )

        print(
            f"  Precision: {metrics['precision']:.4f}"
        )

        print(
            f"  Recall:    {metrics['recall']:.4f}"
        )

        print(
            f"  F1 Score:  {metrics['f1_score']:.4f}"
        )

    print()
    print("=" * 60)
    print("BEST F1 SCORE")
    print("=" * 60)

    best_result = max(
        evaluation_results,
        key=lambda result: result["f1_score"],
    )

    print(
        f"Threshold: {best_result['threshold']:.2f}"
    )

    print(
        f"Accuracy:  {best_result['accuracy']:.4f}"
    )

    print(
        f"Precision: {best_result['precision']:.4f}"
    )

    print(
        f"Recall:    {best_result['recall']:.4f}"
    )

    print(
        f"F1 Score:  {best_result['f1_score']:.4f}"
    )

    save_results(
        dataset=dataset,
        similarity_results=results,
        evaluation_results=evaluation_results,
        best_result=best_result,
    )

    print()
    print(
        f"Evaluation results saved to: {RESULTS_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
