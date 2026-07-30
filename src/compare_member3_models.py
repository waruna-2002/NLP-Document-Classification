from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "member3"


def load_results(file_name: str) -> dict:
    """Load a model-result JSON file."""
    file_path = REPORT_DIRECTORY / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required result file was not found:\n{file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_test_metrics(results: dict) -> dict:
    """Support nested and flat result JSON formats."""
    if "test" in results:
        return results["test"]

    return results


def main() -> None:
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    naive_bayes_results = load_results(
        "naive_bayes_results.json"
    )

    bert_results = load_results(
        "bert_results.json"
    )

    naive_bayes_metrics = extract_test_metrics(
        naive_bayes_results
    )

    bert_metrics = extract_test_metrics(
        bert_results
    )

    rows = [
        {
            "model": "Naive Bayes",
            **naive_bayes_metrics,
        },
        {
            "model": "BERT",
            **bert_metrics,
        },
    ]

    comparison = pd.DataFrame(rows)

    preferred_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in comparison.columns
    ]

    comparison = comparison[available_columns]

    csv_path = REPORT_DIRECTORY / "model_comparison.csv"
    chart_path = REPORT_DIRECTORY / "model_comparison.png"

    comparison.to_csv(csv_path, index=False)

    metric_columns = [
        column
        for column in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        ]
        if column in comparison.columns
    ]

    if metric_columns:
        comparison.set_index("model")[metric_columns].plot(
            kind="bar",
            figsize=(10, 6),
        )

        plt.title("Member 3 Model Comparison")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300)
        plt.close()

    print("\nMember 3 Model Comparison")
    print(comparison.to_string(index=False))

    if "f1_score" in comparison.columns:
        best_row = comparison.loc[
            comparison["f1_score"].idxmax()
        ]

        print(
            "\nBest model according to F1-score:",
            best_row["model"],
        )

    print("\nComparison CSV saved to:", csv_path)
    print("Comparison chart saved to:", chart_path)


if __name__ == "__main__":
    main()
    
