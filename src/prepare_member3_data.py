from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


TEXT_COLUMN_CANDIDATES = [
    "text",
    "message",
    "sms",
    "content",
    "document",
    "sentence",
    "v2",
]

LABEL_COLUMN_CANDIDATES = [
    "label",
    "category",
    "class",
    "target",
    "type",
    "v1",
]


def read_dataset(file_path: Path) -> pd.DataFrame:
    """Read CSV, TSV or TXT data using automatic delimiter detection."""
    try:
        dataframe = pd.read_csv(file_path, sep=None, engine="python")
    except Exception:
        dataframe = pd.read_csv(
            file_path,
            sep=None,
            engine="python",
            header=None,
        )

    # Handle datasets such as the UCI SMS Spam Collection,
    # where the original file may not contain a header.
    if len(dataframe.columns) >= 2:
        first_column_name = str(dataframe.columns[0]).strip().lower()

        known_label_values = {
            "ham",
            "spam",
            "phishing",
            "legitimate",
            "safe",
            "normal",
        }

        if first_column_name in known_label_values:
            dataframe = pd.read_csv(
                file_path,
                sep=None,
                engine="python",
                header=None,
            )

    return dataframe


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> object | None:
    """Find a column using common column-name alternatives."""
    normalized_columns = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate in normalized_columns:
            return normalized_columns[candidate]

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the Member 3 text-classification dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the original dataset file.",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Optional explicit text column name.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Optional explicit label column name.",
    )

    arguments = parser.parse_args()

    input_path = Path(arguments.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: {input_path}"
        )

    dataframe = read_dataset(input_path)

    print("Available columns:", list(dataframe.columns))

    text_column = arguments.text_column
    label_column = arguments.label_column

    if text_column is None:
        text_column = find_column(
            dataframe,
            TEXT_COLUMN_CANDIDATES,
        )

    if label_column is None:
        label_column = find_column(
            dataframe,
            LABEL_COLUMN_CANDIDATES,
        )

    # Fallback for a simple two-column dataset:
    # first column = label, second column = text.
    if label_column is None and len(dataframe.columns) >= 2:
        label_column = dataframe.columns[0]

    if text_column is None and len(dataframe.columns) >= 2:
        text_column = dataframe.columns[1]

    if text_column is None or label_column is None:
        raise ValueError(
            "Could not detect the text and label columns. "
            "Run the script again using --text-column and --label-column."
        )

    print("Detected text column:", text_column)
    print("Detected label column:", label_column)

    prepared = dataframe[[text_column, label_column]].copy()
    prepared.columns = ["text", "label_name"]

    prepared["text"] = prepared["text"].astype(str).str.strip()
    prepared["label_name"] = (
        prepared["label_name"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    prepared = prepared.dropna()
    prepared = prepared[prepared["text"] != ""]
    prepared = prepared.drop_duplicates(subset=["text"])
    prepared = prepared.reset_index(drop=True)

    encoder = LabelEncoder()
    prepared["label"] = encoder.fit_transform(
        prepared["label_name"]
    )

    output_columns = ["text", "label", "label_name"]
    prepared = prepared[output_columns]

    train_data, temporary_data = train_test_split(
        prepared,
        test_size=0.30,
        random_state=42,
        stratify=prepared["label"],
    )

    validation_data, test_data = train_test_split(
        temporary_data,
        test_size=0.50,
        random_state=42,
        stratify=temporary_data["label"],
    )

    output_directory = Path("dataset/member3_processed")
    output_directory.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(
        output_directory / "train.csv",
        index=False,
    )
    validation_data.to_csv(
        output_directory / "validation.csv",
        index=False,
    )
    test_data.to_csv(
        output_directory / "test.csv",
        index=False,
    )

    label_mapping = {
        str(index): label_name
        for index, label_name in enumerate(encoder.classes_)
    }

    with open(
        output_directory / "label_mapping.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            label_mapping,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\nDataset preparation completed.")
    print("Total samples:", len(prepared))
    print("Training samples:", len(train_data))
    print("Validation samples:", len(validation_data))
    print("Testing samples:", len(test_data))
    print("Label mapping:", label_mapping)


if __name__ == "__main__":
    main()
    
