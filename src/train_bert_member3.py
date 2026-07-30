from __future__ import annotations

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
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


BASE_MODEL = "bert-base-uncased"

DATA_DIRECTORY = Path("dataset/member3_processed")
MODEL_DIRECTORY = Path("models/member3/bert_model")
CHECKPOINT_DIRECTORY = Path(
    "models/member3/bert_checkpoints"
)
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


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted_logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )
    exponentials = np.exp(shifted_logits)

    return exponentials / np.sum(
        exponentials,
        axis=1,
        keepdims=True,
    )


def main() -> None:
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    train_frame = pd.read_csv(DATA_DIRECTORY / "train.csv")
    validation_frame = pd.read_csv(
        DATA_DIRECTORY / "validation.csv"
    )
    test_frame = pd.read_csv(DATA_DIRECTORY / "test.csv")

    with open(
        DATA_DIRECTORY / "label_mapping.json",
        "r",
        encoding="utf-8",
    ) as file:
        label_mapping = json.load(file)

    number_of_labels = len(label_mapping)

    id_to_label = {
        int(label_id): label_name
        for label_id, label_name in label_mapping.items()
    }

    label_to_id = {
        label_name: label_id
        for label_id, label_name in id_to_label.items()
    }

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=number_of_labels,
        id2label=id_to_label,
        label2id=label_to_id,
    )

    train_dataset = Dataset.from_pandas(
        train_frame[["text", "label"]],
        preserve_index=False,
    )

    validation_dataset = Dataset.from_pandas(
        validation_frame[["text", "label"]],
        preserve_index=False,
    )

    test_dataset = Dataset.from_pandas(
        test_frame[["text", "label"]],
        preserve_index=False,
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )

    train_dataset = train_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )

    validation_dataset = validation_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )

    test_dataset = test_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    def compute_metrics(evaluation_prediction):
        logits, true_labels = evaluation_prediction
        predicted_labels = np.argmax(logits, axis=-1)

        return calculate_metrics(
            true_labels,
            predicted_labels,
        )

    training_arguments = TrainingArguments(
        output_dir=str(CHECKPOINT_DIRECTORY),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=42,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting BERT training...")
    trainer.train()

    trainer.save_model(MODEL_DIRECTORY)
    tokenizer.save_pretrained(MODEL_DIRECTORY)

    validation_output = trainer.predict(
        validation_dataset
    )
    validation_predictions = np.argmax(
        validation_output.predictions,
        axis=-1,
    )
    validation_probabilities = softmax(
        validation_output.predictions
    )

    test_output = trainer.predict(test_dataset)
    test_predictions = np.argmax(
        test_output.predictions,
        axis=-1,
    )
    test_probabilities = softmax(
        test_output.predictions
    )

    validation_metrics = calculate_metrics(
        validation_output.label_ids,
        validation_predictions,
        validation_probabilities,
    )

    test_metrics = calculate_metrics(
        test_output.label_ids,
        test_predictions,
        test_probabilities,
    )

    results = {
        "model": "BERT",
        "base_model": BASE_MODEL,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    with open(
        REPORT_DIRECTORY / "bert_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    label_ids = sorted(id_to_label.keys())
    target_names = [
        id_to_label[label_id]
        for label_id in label_ids
    ]

    report = classification_report(
        test_output.label_ids,
        test_predictions,
        labels=label_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        REPORT_DIRECTORY
        / "bert_classification_report.csv"
    )

    ConfusionMatrixDisplay.from_predictions(
        test_output.label_ids,
        test_predictions,
        labels=label_ids,
        display_labels=target_names,
        cmap="Blues",
        xticks_rotation=45,
    )

    plt.title("BERT Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        REPORT_DIRECTORY / "bert_confusion_matrix.png",
        dpi=300,
    )
    plt.close()

    print("\nBERT training completed.")
    print("Model saved to:", MODEL_DIRECTORY)
    print("Validation metrics:", validation_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
    
