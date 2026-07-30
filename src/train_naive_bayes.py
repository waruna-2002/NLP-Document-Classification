from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


DATA_DIRECTORY = Path("dataset/member3_processed")
MODEL_DIRECTORY = Path("models/member3")
REPORT_DIRECTORY = Path("reports/member3")


def calculate_metrics(
    true_labels,
    predicted_labels,
    probabilities=None,
) -> dict:
    precision, recall, f1_score, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )
    )

    metrics = {
        "accuracy": float(
            accuracy_score(true_labels, predicted_labels)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
    }

    unique_labels = sorted(set(true_labels))

    if probabilities is not None and len(unique_labels) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(
                true_labels,
                probabilities[:, 1],
            )
        )

    return metrics


def main() -> None:
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    train_data = pd.read_csv(DATA_DIRECTORY / "train.csv")
    validation_data = pd.read_csv(
        DATA_DIRECTORY / "validation.csv"
    )
    test_data = pd.read_csv(DATA_DIRECTORY / "test.csv")

    with open(
        DATA_DIRECTORY / "label_mapping.json",
        "r",
        encoding="utf-8",
    ) as file:
        label_mapping = json.load(file)

    label_ids = sorted(
        int(label_id)
        for label_id in label_mapping.keys()
    )

    target_names = [
        label_mapping[str(label_id)]
        for label_id in label_ids
    ]

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
            (
                "naive_bayes",
                MultinomialNB(),
            ),
        ]
    )

    parameter_grid = {
        "tfidf__ngram_range": [
            (1, 1),
            (1, 2),
        ],
        "naive_bayes__alpha": [
            0.1,
            0.5,
            1.0,
        ],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1_weighted",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    print("Training Multinomial Naive Bayes...")
    search.fit(
        train_data["text"],
        train_data["label"],
    )

    best_model = search.best_estimator_

    print("Best parameters:", search.best_params_)

    validation_predictions = best_model.predict(
        validation_data["text"]
    )
    validation_probabilities = best_model.predict_proba(
        validation_data["text"]
    )

    test_predictions = best_model.predict(
        test_data["text"]
    )
    test_probabilities = best_model.predict_proba(
        test_data["text"]
    )

    validation_metrics = calculate_metrics(
        validation_data["label"],
        validation_predictions,
        validation_probabilities,
    )

    test_metrics = calculate_metrics(
        test_data["label"],
        test_predictions,
        test_probabilities,
    )

    results = {
        "model": "TF-IDF + Multinomial Naive Bayes",
        "best_parameters": search.best_params_,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    model_path = (
        MODEL_DIRECTORY / "naive_bayes_pipeline.joblib"
    )
    joblib.dump(best_model, model_path)

    with open(
        REPORT_DIRECTORY / "naive_bayes_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    report = classification_report(
        test_data["label"],
        test_predictions,
        labels=label_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        REPORT_DIRECTORY
        / "naive_bayes_classification_report.csv"
    )

    ConfusionMatrixDisplay.from_predictions(
        test_data["label"],
        test_predictions,
        labels=label_ids,
        display_labels=target_names,
        cmap="Blues",
        xticks_rotation=45,
    )

    plt.title("Naive Bayes Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        REPORT_DIRECTORY
        / "naive_bayes_confusion_matrix.png",
        dpi=300,
    )
    plt.close()

    print("\nNaive Bayes training completed.")
    print("Model saved to:", model_path)
    print("Validation metrics:", validation_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
    
