import re


def clean_text(text: str) -> str:
    """Clean SMS text while preserving useful phishing indicators."""

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " urltoken ",
        text
    )

    text = re.sub(
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
        " emailtoken ",
        text
    )

    text = re.sub(
        r"\+?\d[\d\s\-]{7,}\d",
        " phonetoken ",
        text
    )

    text = re.sub(
        r"[$£€₹]|lkr|usd",
        " moneytoken ",
        text
    )

    text = re.sub(
        r"[^a-z0-9!?'\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


if __name__ == "__main__":
    sample_message = (
        "URGENT! Your account has been suspended. "
        "Verify now at https://example.com."
    )

    print("Original message:")
    print(sample_message)

    print("\nCleaned message:")
    print(clean_text(sample_message))
