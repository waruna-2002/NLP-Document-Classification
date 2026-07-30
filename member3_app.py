import math
import re
from pathlib import Path

import joblib
import streamlit as st

from src.preprocessing import clean_text


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "member3"
    / "svm_tfidf_pipeline.joblib"
)


def identify_indicators(message: str) -> list[str]:
    """Identify basic suspicious SMS indicators."""

    indicators = []

    if re.search(
        r"https?://\S+|www\.\S+",
        message,
        flags=re.IGNORECASE
    ):
        indicators.append("Contains a web link")

    if re.search(
        r"\b(urgent|immediately|verify|suspended|blocked|expire)\b",
        message,
        flags=re.IGNORECASE
    ):
        indicators.append("Uses urgent or threatening language")

    if re.search(
        r"\b(password|pin|otp|login|account|credentials)\b",
        message,
        flags=re.IGNORECASE
    ):
        indicators.append("Requests account or credential information")

    if re.search(
        r"\b(winner|prize|reward|free|cash|claim)\b",
        message,
        flags=re.IGNORECASE
    ):
        indicators.append("Contains reward or prize language")

    return indicators


@st.cache_resource
def load_model():
    """Load the saved SVM pipeline."""

    return joblib.load(MODEL_PATH)


st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ PhishGuard")
st.subheader(
    "Phishing SMS and Social Engineering Detection System"
)

st.write(
    "Enter an SMS message to classify it as legitimate "
    "or a potential spam/phishing risk."
)

message = st.text_area(
    "SMS Message",
    height=180,
    placeholder=(
        "Example: Your account has been suspended. "
        "Click the link now to verify your identity."
    )
)

if st.button(
    "Analyze Message",
    type="primary"
):
    if not message.strip():
        st.warning(
            "Please enter an SMS message."
        )

    elif not MODEL_PATH.exists():
        st.error(
            "The trained SVM model was not found. "
            "Run src/train_svm.py first."
        )

    else:
        model = load_model()

        cleaned_message = clean_text(message)

        prediction = int(
            model.predict(
                [cleaned_message]
            )[0]
        )

        decision_score = float(
            model.decision_function(
                [cleaned_message]
            )[0]
        )

        risk_score = (
            1
            / (
                1
                + math.exp(
                    -decision_score
                )
            )
        )

        indicators = identify_indicators(
            message
        )

        if prediction == 1:
            st.error(
                "Prediction: Spam / Phishing Risk"
            )

            st.metric(
                "Model Risk Score",
                f"{risk_score * 100:.2f}%"
            )

        else:
            st.success(
                "Prediction: Legitimate Message"
            )

            st.metric(
                "Model Risk Score",
                f"{risk_score * 100:.2f}%"
            )

        st.caption(
            "The model risk score is derived from the SVM "
            "decision score and is not a calibrated probability."
        )

        if indicators:
            st.subheader(
                "Detected Indicators"
            )

            for indicator in indicators:
                st.write(
                    f"- {indicator}"
                )

        else:
            st.info(
                "No basic rule-based indicators were detected."
            )

        st.warning(
            "Do not click unknown links or share passwords, "
            "PINs, OTPs, or banking details."
        )
        
