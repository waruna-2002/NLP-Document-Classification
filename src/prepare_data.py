from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "spam.csv"
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"

RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    """Load the SMS dataset using a compatible encoding."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    try:
        dataframe = pd.read_csv(
            RAW_DATA_PATH,
            encoding="utf-8"
        )
    except UnicodeDecodeError:
        dataframe = pd.read_csv(
            RAW_DATA_PATH,
            encoding="latin-1"
        )

    return dataframe


def standardize_columns(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """Convert supported column formats into label and message columns."""

    if {"v1", "v2"}.issubset(dataframe.columns):
        dataframe = dataframe[
            ["v1", "v2"]
        ].copy()

        dataframe.columns = [
            "label",
            "message"
        ]

    elif {"Category", "Message"}.issubset(
        dataframe.columns
    ):
        dataframe = dataframe[
            ["Category", "Message"]
        ].copy()

        dataframe.columns = [
            "label",
            "message"
        ]

    elif {"label", "message"}.issubset(
        dataframe.columns
    ):
        dataframe = dataframe[
            ["label", "message"]
        ].copy()

    else:
        raise ValueError(
            "Unsupported dataset columns: "
            f"{dataframe.columns.tolist()}"
        )

    return dataframe


def clean_dataset(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """Clean messages, remove duplicates, and encode class labels."""

    dataframe = standardize_columns(
        dataframe
    )

    print("Original dataset shape:")
    print(dataframe.shape)

    print("\nOriginal class distribution:")
    print(dataframe["label"].value_counts())

    print("\nMissing values:")
    print(dataframe.isnull().sum())

    dataframe = dataframe.dropna(
        subset=["label", "message"]
    ).copy()

    dataframe["label"] = (
        dataframe["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    dataframe["message"] = (
        dataframe["message"]
        .astype(str)
        .str.strip()
    )

    dataframe = dataframe.drop_duplicates(
        subset=["message"]
    ).reset_index(drop=True)

    label_mapping = {
        "ham": 0,
        "spam": 1
    }

    dataframe = dataframe[
        dataframe["label"].isin(
            label_mapping.keys()
        )
    ].copy()

    dataframe["target"] = dataframe[
        "label"
    ].map(label_mapping)

    dataframe["clean_message"] = dataframe[
        "message"
    ].apply(clean_text)

    dataframe = dataframe[
        dataframe["clean_message"].str.len() > 0
    ].reset_index(drop=True)

    print("\nCleaned dataset shape:")
    print(dataframe.shape)

    print("\nCleaned class distribution:")
    print(dataframe["label"].value_counts())

    return dataframe


def split_dataset(
    dataframe: pd.DataFrame
):
    """Split data into 70% training, 15% validation, and 15% testing."""

    train_dataframe, temporary_dataframe = train_test_split(
        dataframe,
        test_size=0.30,
        stratify=dataframe["target"],
        random_state=RANDOM_STATE
    )

    validation_dataframe, test_dataframe = train_test_split(
        temporary_dataframe,
        test_size=0.50,
        stratify=temporary_dataframe["target"],
        random_state=RANDOM_STATE
    )

    return (
        train_dataframe,
        validation_dataframe,
        test_dataframe
    )


def save_datasets(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame
) -> None:
    """Save all processed dataset splits as CSV files."""

    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    train_dataframe.to_csv(
        PROCESSED_DIRECTORY / "train.csv",
        index=False,
        encoding="utf-8"
    )

    validation_dataframe.to_csv(
        PROCESSED_DIRECTORY / "validation.csv",
        index=False,
        encoding="utf-8"
    )

    test_dataframe.to_csv(
        PROCESSED_DIRECTORY / "test.csv",
        index=False,
        encoding="utf-8"
    )

    print("\nDataset split summary:")
    print(f"Training records: {len(train_dataframe)}")
    print(f"Validation records: {len(validation_dataframe)}")
    print(f"Testing records: {len(test_dataframe)}")

    print("\nSaved files:")
    print(PROCESSED_DIRECTORY / "train.csv")
    print(PROCESSED_DIRECTORY / "validation.csv")
    print(PROCESSED_DIRECTORY / "test.csv")


def main() -> None:
    dataframe = load_dataset()

    cleaned_dataframe = clean_dataset(
        dataframe
    )

    (
        train_dataframe,
        validation_dataframe,
        test_dataframe
    ) = split_dataset(
        cleaned_dataframe
    )

    save_datasets(
        train_dataframe,
        validation_dataframe,
        test_dataframe
    )

    print("\nDataset preparation completed successfully.")


if __name__ == "__main__":
    main()

