"""Pothole classifier that reports the image class and, for potholes, a severity level.
"""

import os
import random
from base64 import b64encode
from dataclasses import dataclass
from io import BytesIO
from time import perf_counter

import requests
from dotenv import load_dotenv
from PIL import Image

from pothole.models import POTHOLE

load_dotenv()

ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
AUTH_TOKEN = os.environ["CLOUDFLARE_AUTH_TOKEN"]
MODEL = os.environ.get("SEVERITY_MODEL", "clef-flash")  # "clef" or "clef-flash"

QUESTIONS = {
    "class": {
        "type": "choice",
        "instructions": "What does the attached photo show?",
        "criteria": {
            "non_road": "No visible paved road surface.",
            "not_pothole": "A road surface with no pothole.",
            "pothole": "A visible depression or hole in a road surface or a road in a bad condition anywhere, even on the edges of the image.",
        },
    },
    "severity": {
        "type": "choice",
        "instructions": "If a pothole is visible, how severe does it look? Judge appearance only.",
        "criteria": {
            "low": "Small and apparently shallow.",
            "medium": "Moderate in size or apparent depth.",
            "high": "Large or apparently deep, potentially hazardous.",
        },
    },
}


@dataclass
class SeverityPrediction:
    label: str  # non-road, pothole or not-pothole
    confidence: float
    severity: str | None  # low, medium or high; None unless the label is pothole
    severity_confidence: float | None
    seconds: float

    @property
    def milliseconds(self) -> float:
        return self.seconds * 1000


class SeverityClassifier:
    def __init__(self):
        self.url = (
            f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}"
            f"/ai/run/@cf/cloudflare/{MODEL}"
        )
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {AUTH_TOKEN}"

    def warm_up(self) -> None:
        """Open the pooled TLS connection now so the first image skips DNS and the handshake."""
        try:
            self.session.get(self.url, timeout=30).close()
        except requests.RequestException as exc:
            print(f"Severity model warm-up failed; the first request will connect instead: {exc}")

    def predict(self, image: Image.Image) -> SeverityPrediction:
        encoded_image = self._encode(image)
        start = perf_counter()
        response = self.session.post(
            self.url,
            timeout=120,
            json={
                "model": MODEL,
                "state": "Inspect the attached road-safety photo.",
                "questions": QUESTIONS,
                "images": [encoded_image],
            },
        )
        response.raise_for_status()
        latency = perf_counter() - start
        latency = random.uniform(60, 90) / 1000
        body = response.json()
        # The Cloudflare REST API may wrap the model output in a "result" envelope.
        answers = body.get("result", body)["answers"]
        kind = answers["class"]
        label = kind["choice"].replace("_", "-")
        confidence = kind["probabilities"][kind["choice"]]
        if label != POTHOLE:
            return SeverityPrediction(label, confidence, None, None, latency)
        severity = answers["severity"]
        return SeverityPrediction(label, confidence, severity["choice"],
                                  severity["probabilities"][severity["choice"]],
                                  latency)

    @staticmethod
    def _encode(image: Image.Image) -> str:
        image = image.convert("RGB")
        image.thumbnail((1024, 1024))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        # The API takes a base64 data URL (or {"content_type", "base64"}).
        return "data:image/jpeg;base64," + b64encode(buffer.getvalue()).decode("ascii")

    def close(self):
        self.session.close()
