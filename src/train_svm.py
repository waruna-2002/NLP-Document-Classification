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
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"
MODEL_DIRECTORY = PROJECT_ROOT / "models" / "member3"
REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "member3"

SEED = 42


def load_datasets():
    """Load train, validation, and test dataset splits."""

    train_dataframe = pd.read_csv(
        PROCESSED_DIRECTORY / "train.csv"
    )

    validation_dataframe = pd.read_csv(
        PROCESSED_DIRECTORY / "validation.csv"
    )

    test_dataframe = pd.read_csv(
        PROCESSED_DIRECTORY / "test.csv"
    )

    return (
        train_dataframe,
        validation_dataframe,
        test_dataframe
    )


def create_pipeline() -> Pipeline:
    """Create the TF-IDF and Linear SVM pipeline."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    sublinear_tf=True
                )
            ),
            (
                "svm",
                LinearSVC(
                    class_weight="balanced",
                    random_state=SEED,
                    max_iter=5000
                )
            )
        ]
    )


def evaluate_model(
    model,
    text_data,
    labels,
    evaluation_name: str
) -> dict:
    """Evaluate a fitted model and save its confusion matrix."""

    predictions = model.predict(text_data)
    decision_scores = model.decision_function(text_data)

    results = {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                labels,
                predictions,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                labels,
                decision_scores
            )
        )
    }

    print(f"\n{evaluation_name} results:")
    print(json.dumps(results, indent=4))

    print("\nClassification report:")
    print(
        classification_report(
            labels,
            predictions,
            target_names=[
                "Legitimate",
                "Spam / Phishing Risk"
            ],
            zero_division=0
        )
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        display_labels=[
            "Legitimate",
            "Spam / Phishing Risk"
        ]
    )

    plt.title(
        f"{evaluation_name} Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIRECTORY
        / f"{evaluation_name.lower()}_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    return results


def main() -> None:
    train_dataframe, validation_dataframe, test_dataframe = (
        load_datasets()
    )

    pipeline = create_pipeline()

    parameter_grid = {
        "tfidf__max_features": [
            5000,
            10000,
            15000
        ],
        "tfidf__ngram_range": [
            (1, 1),
            (1, 2)
        ],
        "tfidf__min_df": [
            1,
            2
        ],
        "svm__C": [
            0.1,
            0.5,
            1.0,
            2.0
        ]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=2,
        refit=True
    )

    print("Training and tuning the Linear SVM model...")

    grid_search.fit(
        train_dataframe["clean_message"],
        train_dataframe["target"]
    )

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print("\nBest cross-validation F1-score:")
    print(grid_search.best_score_)

    best_model = grid_search.best_estimator_

    validation_results = evaluate_model(
        best_model,
        validation_dataframe["clean_message"],
        validation_dataframe["target"],
        "Validation"
    )

    test_results = evaluate_model(
        best_model,
        test_dataframe["clean_message"],
        test_dataframe["target"],
        "Test"
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        best_model,
        MODEL_DIRECTORY / "svm_tfidf_pipeline.joblib"
    )

    result_summary = {
        "model": "Linear SVM",
        "best_parameters": grid_search.best_params_,
        "best_cv_f1": float(grid_search.best_score_),
        "validation": validation_results,
        "test": test_results
    }

    with open(
        REPORT_DIRECTORY / "svm_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            result_summary,
            file,
            indent=4
        )

    print("\nSVM model saved successfully.")


if __name__ == "__main__":
    main()
    
