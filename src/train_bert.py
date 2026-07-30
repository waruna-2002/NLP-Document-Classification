import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"
MODEL_DIRECTORY = PROJECT_ROOT / "models" / "member3" / "bert_model"
REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "member3"

MODEL_NAME = "bert-base-uncased"
SEED = 42


def load_dataset_split(filename: str) -> Dataset:
    """Load a CSV split and convert it into a Hugging Face dataset."""

    dataframe = pd.read_csv(
        PROCESSED_DIRECTORY / filename
    )

    dataframe = dataframe[
        ["message", "target"]
    ].copy()

    dataset = Dataset.from_pandas(
        dataframe,
        preserve_index=False
    )

    dataset = dataset.rename_column(
        "target",
        "labels"
    )

    return dataset


def compute_metrics(evaluation_prediction):
    """Calculate classification metrics during evaluation."""

    logits, labels = evaluation_prediction

    predictions = np.argmax(
        logits,
        axis=-1
    )

    return {
        "accuracy": accuracy_score(
            labels,
            predictions
        ),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0
        )
    }


def main() -> None:
    print("CUDA available:", torch.cuda.is_available())

    train_dataset = load_dataset_split(
        "train.csv"
    )

    validation_dataset = load_dataset_split(
        "validation.csv"
    )

    test_dataset = load_dataset_split(
        "test.csv"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["message"],
            truncation=True,
            max_length=128
        )

    tokenized_train = train_dataset.map(
        tokenize_batch,
        batched=True
    )

    tokenized_validation = validation_dataset.map(
        tokenize_batch,
        batched=True
    )

    tokenized_test = test_dataset.map(
        tokenize_batch,
        batched=True
    )

    id2label = {
        0: "LEGITIMATE",
        1: "SPAM_PHISHING_RISK"
    }

    label2id = {
        "LEGITIMATE": 0,
        "SPAM_PHISHING_RISK": 1
    }

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label=id2label,
            label2id=label2id
        )
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    training_arguments = TrainingArguments(
        output_dir=str(
            PROJECT_ROOT
            / "models"
            / "member3"
            / "bert_checkpoints"
        ),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    prediction_output = trainer.predict(
        tokenized_test
    )

    test_predictions = np.argmax(
        prediction_output.predictions,
        axis=-1
    )

    test_labels = prediction_output.label_ids

    results = {
        "model": "BERT",
        "accuracy": float(
            accuracy_score(
                test_labels,
                test_predictions
            )
        ),
        "precision": float(
            precision_score(
                test_labels,
                test_predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                test_labels,
                test_predictions,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                test_labels,
                test_predictions,
                zero_division=0
            )
        )
    }

    print("\nBERT test results:")
    print(json.dumps(results, indent=4))

    print("\nClassification report:")
    print(
        classification_report(
            test_labels,
            test_predictions,
            target_names=[
                "Legitimate",
                "Spam / Phishing Risk"
            ],
            zero_division=0
        )
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    trainer.save_model(
        str(MODEL_DIRECTORY)
    )

    tokenizer.save_pretrained(
        str(MODEL_DIRECTORY)
    )

    with open(
        REPORT_DIRECTORY / "bert_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=4
        )

    ConfusionMatrixDisplay.from_predictions(
        test_labels,
        test_predictions,
        display_labels=[
            "Legitimate",
            "Spam / Phishing Risk"
        ]
    )

    plt.title("BERT Test Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIRECTORY
        / "bert_test_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print("\nBERT model saved successfully.")


if __name__ == "__main__":
    main()
    
