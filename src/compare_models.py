from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Identify the root folder of the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the directory containing Member 3 evaluation results.
REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "member3"

# Define paths to the saved result files.
SVM_RESULTS_PATH = REPORT_DIRECTORY / "svm_results.json"
BERT_RESULTS_PATH = REPORT_DIRECTORY / "bert_results.json"


def load_json(file_path: Path) -> dict:
    """Load and return data from a JSON file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def extract_test_metrics(results: dict) -> dict:
    """
    Extract test metrics from either a nested or flat JSON structure.

    Supported structures:

    {
        "test": {
            "accuracy": ...,
            "precision": ...,
            "recall": ...,
            "f1_score": ...,
            "roc_auc": ...
        }
    }

    or:

    {
        "accuracy": ...,
        "precision": ...,
        "recall": ...,
        "f1_score": ...,
        "roc_auc": ...
    }
    """

    if "test" in results:
        return results["test"]

    return results


def validate_metrics(
    model_name: str,
    metrics: dict
) -> None:
    """Check whether all required evaluation metrics are available."""

    required_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]

    missing_metrics = [
        metric
        for metric in required_metrics
        if metric not in metrics
    ]

    if missing_metrics:
        raise KeyError(
            f"{model_name} result file is missing: "
            f"{missing_metrics}"
        )


def main() -> None:
    """Compare the final test results of SVM and BERT."""

    # Load the saved evaluation result files.
    svm_results = load_json(
        SVM_RESULTS_PATH
    )

    bert_results = load_json(
        BERT_RESULTS_PATH
    )

    # Support nested and flat result formats.
    svm_metrics = extract_test_metrics(
        svm_results
    )

    bert_metrics = extract_test_metrics(
        bert_results
    )

    # Validate the required metric values.
    validate_metrics(
        "Linear SVM",
        svm_metrics
    )

    validate_metrics(
        "BERT",
        bert_metrics
    )

    # Build the final comparison table.
    comparison_dataframe = pd.DataFrame(
        [
            {
                "Model": "Linear SVM",
                "Accuracy": svm_metrics["accuracy"],
                "Precision": svm_metrics["precision"],
                "Recall": svm_metrics["recall"],
                "F1-Score": svm_metrics["f1_score"],
                "ROC-AUC": svm_metrics["roc_auc"]
            },
            {
                "Model": "BERT",
                "Accuracy": bert_metrics["accuracy"],
                "Precision": bert_metrics["precision"],
                "Recall": bert_metrics["recall"],
                "F1-Score": bert_metrics["f1_score"],
                "ROC-AUC": bert_metrics["roc_auc"]
            }
        ]
    )

    print("\nMember 3 Model Comparison:\n")

    print(
        comparison_dataframe.to_string(
            index=False
        )
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save the comparison table.
    comparison_dataframe.to_csv(
        REPORT_DIRECTORY / "model_comparison.csv",
        index=False
    )

    # Create a visual comparison chart.
    chart_dataframe = comparison_dataframe.set_index(
        "Model"
    )[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC"
        ]
    ]

    chart_dataframe.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title("Member 3 Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Metric Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIRECTORY / "model_comparison.png",
        dpi=300
    )

    plt.close()

    # Identify the best-performing models.
    best_f1_model = comparison_dataframe.loc[
        comparison_dataframe["F1-Score"].idxmax(),
        "Model"
    ]

    best_recall_model = comparison_dataframe.loc[
        comparison_dataframe["Recall"].idxmax(),
        "Model"
    ]

    print("\nBest F1-score model:", best_f1_model)
    print("Best recall model:", best_recall_model)

    print("\nComparison files saved successfully:")
    print(
        REPORT_DIRECTORY / "model_comparison.csv"
    )
    print(
        REPORT_DIRECTORY / "model_comparison.png"
    )


if __name__ == "__main__":
    main()
    
