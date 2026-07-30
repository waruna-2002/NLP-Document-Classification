import glob
import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Text extraction functions import කරගැනීම
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from extraction import (
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_txt,
)
from text_utils import clean_text


def load_dataset_from_filepaths():
    """dataset/ folder එකේ තියෙන dynamic sub-folders සහ files read කරයි."""
    dataset_base = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../dataset")
    )
    data = []

    # Target class directory mapping
    folders = {
        "company_sensitive": "COMPANY_SENSITIVE",  # ඔයාගේ folder එකේ නම
        "personal": "PERSONAL",  # ඔයාගේ folder එකේ නම
    }

    for folder_name, label in folders.items():
        folder_path = os.path.join(dataset_base, folder_name)

        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found at {folder_path}")
            continue

        # Sub-directories ඇතුලේ තියෙන files ද ඇතුලුව scan කිරීම
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = file.split(".")[-1].lower()

                text = ""
                try:
                    if ext == "txt":
                        text = extract_text_from_txt(file_path)
                    elif ext == "pdf":
                        text = extract_text_from_pdf(file_path)
                    elif ext == "docx":
                        text = extract_text_from_docx(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

                if text and len(text.strip()) > 10:
                    data.append({"text": text, "label": label})

    return pd.DataFrame(data)


def train_random_forest():
    print("Reading data from dataset filepaths...")
    df = load_dataset_from_filepaths()

    if df.empty:
        print("Error: No valid document data found in dataset folder!")
        return

    print(
        f"Loaded {len(df)} documents.\nClass Distribution:\n{df['label'].value_counts()}\n"
    )

    # Preprocess
    df["clean_text"] = df["text"].apply(clean_text)
    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest Model Pipeline Setup
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                ),
            ),
        ]
    )

    print("Training Random Forest Model...")
    pipeline.fit(X_train, y_train)

    # Evaluation
    preds = pipeline.predict(X_test)
    print("\n--- Model Classification Report ---")
    print(classification_report(y_test, preds))

    # Save Path Setup
    models_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../models")
    )
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, "random_forest.pkl")

    joblib.dump(pipeline, save_path)
    print(f"Model saved successfully to: {save_path}")


if __name__ == "__main__":
    train_random_forest()