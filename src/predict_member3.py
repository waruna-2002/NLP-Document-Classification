from __future__ import annotations

from pathlib import Path

import joblib
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from preprocessing import clean_text


# Identify the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the saved SVM model path.
SVM_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "member3"
    / "svm_tfidf_pipeline.joblib"
)

# Define the saved local BERT model directory.
BERT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "member3"
    / "bert_model"
)


def predict_with_svm(message: str) -> None:
    """Classify an SMS message using the trained Linear SVM model."""

    if not SVM_MODEL_PATH.exists():
        raise FileNotFoundError(
            "SVM model not found. Run src/train_svm.py first."
        )

    # Load the saved TF-IDF and Linear SVM pipeline.
    svm_model = joblib.load(SVM_MODEL_PATH)

    # Apply the same preprocessing used during model training.
    cleaned_message = clean_text(message)

    # Generate the class prediction.
    prediction = int(
        svm_model.predict([cleaned_message])[0]
    )

    # Obtain the SVM decision score.
    decision_score = float(
        svm_model.decision_function(
            [cleaned_message]
        )[0]
    )

    label = (
        "Spam / Phishing Risk"
        if prediction == 1
        else "Legitimate"
    )

    print("\n--- Linear SVM Result ---")
    print("Original message:", message)
    print("Cleaned message:", cleaned_message)
    print("Prediction:", label)
    print("Decision score:", decision_score)
    print(
        "Note: The SVM decision score is not "
        "a calibrated probability."
    )


def predict_with_bert(message: str) -> None:
    """Classify an SMS message using the fine-tuned BERT model."""

    config_path = BERT_MODEL_PATH / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            "BERT model not found. Place the extracted model files in "
            "models/member3/bert_model."
        )

    # Load the fine-tuned tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(
        str(BERT_MODEL_PATH)
    )

    bert_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            str(BERT_MODEL_PATH)
        )
    )

    # Use the GPU when available; otherwise, use the CPU.
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    bert_model.to(device)
    bert_model.eval()

    # Tokenize the original message.
    encoded_input = tokenizer(
        message,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    encoded_input = {
        key: value.to(device)
        for key, value in encoded_input.items()
    }

    # Generate the prediction without calculating gradients.
    with torch.no_grad():
        output = bert_model(
            **encoded_input
        )

        probabilities = torch.softmax(
            output.logits,
            dim=1,
        )

    prediction = int(
        torch.argmax(
            probabilities,
            dim=1,
        ).item()
    )

    confidence = float(
        probabilities[
            0,
            prediction,
        ].item()
    )

    label = (
        "Spam / Phishing Risk"
        if prediction == 1
        else "Legitimate"
    )

    print("\n--- BERT Result ---")
    print("Message:", message)
    print("Prediction:", label)
    print(
        "Confidence:",
        f"{confidence * 100:.2f}%",
    )
    print("Device:", device)


def main() -> None:
    """Allow the user to test either Member 3 model."""

    print("=" * 50)
    print("PhishGuard - Member 3 Model Prediction")
    print("=" * 50)
    print("1 - Linear SVM")
    print("2 - BERT")

    model_choice = input(
        "\nSelect a model: "
    ).strip()

    message = input(
        "Enter an SMS message: "
    ).strip()

    if not message:
        raise ValueError(
            "The SMS message cannot be empty."
        )

    if model_choice == "1":
        predict_with_svm(message)

    elif model_choice == "2":
        predict_with_bert(message)

    else:
        raise ValueError(
            "Invalid model choice. Enter 1 or 2."
        )


if __name__ == "__main__":
    main()
    
