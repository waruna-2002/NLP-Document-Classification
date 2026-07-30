from __future__ import annotations

import json
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NAIVE_BAYES_PATH = (
    PROJECT_ROOT
    / "models"
    / "member3"
    / "naive_bayes_pipeline.joblib"
)

BERT_PATH = (
    PROJECT_ROOT
    / "models"
    / "member3"
    / "bert_model"
)

LABEL_MAPPING_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "member3_processed"
    / "label_mapping.json"
)


def load_label_mapping() -> dict[int, str]:
    if not LABEL_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Label mapping file not found:\n{LABEL_MAPPING_PATH}"
        )

    with open(
        LABEL_MAPPING_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        mapping = json.load(file)

    return {
        int(label_id): label_name
        for label_id, label_name in mapping.items()
    }


def predict_naive_bayes(
    message: str,
    label_mapping: dict[int, str],
) -> None:
    if not NAIVE_BAYES_PATH.exists():
        raise FileNotFoundError(
            f"Naive Bayes model not found:\n{NAIVE_BAYES_PATH}"
        )

    model = joblib.load(NAIVE_BAYES_PATH)

    predicted_id = int(model.predict([message])[0])
    probabilities = model.predict_proba([message])[0]

    print("\nPrediction:", label_mapping[predicted_id])
    print("\nProbabilities:")

    for class_id, probability in zip(
        model.classes_,
        probabilities,
    ):
        label_name = label_mapping[int(class_id)]
        print(f"{label_name}: {probability * 100:.2f}%")


def predict_bert(
    message: str,
    label_mapping: dict[int, str],
) -> None:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    if not BERT_PATH.exists():
        raise FileNotFoundError(
            f"BERT model folder not found:\n{BERT_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_PATH
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    inputs = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0]

    predicted_id = int(
        torch.argmax(probabilities).item()
    )

    print("\nPrediction:", label_mapping[predicted_id])
    print("\nProbabilities:")

    for class_id, probability in enumerate(
        probabilities.tolist()
    ):
        label_name = label_mapping[class_id]
        print(f"{label_name}: {probability * 100:.2f}%")


def main() -> None:
    label_mapping = load_label_mapping()

    print("\nMember 3 Prediction System")
    print("1 - Naive Bayes")
    print("2 - BERT")

    model_choice = input(
        "\nSelect a model: "
    ).strip()

    message = input(
        "Enter the SMS message: "
    ).strip()

    if not message:
        print("Message cannot be empty.")
        return

    if model_choice == "1":
        predict_naive_bayes(
            message,
            label_mapping,
        )
    elif model_choice == "2":
        predict_bert(
            message,
            label_mapping,
        )
    else:
        print("Invalid model selection. Enter 1 or 2.")


if __name__ == "__main__":
    main()
    
