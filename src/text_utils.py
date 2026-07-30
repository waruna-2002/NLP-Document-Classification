import re


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)  # Extra white space අයින් කිරීම
    text = re.sub(
        r"[^a-zA-Z0-9\s]", "", text
    )  # Special characters/punctuations අයින් කිරීම
    return text.strip()