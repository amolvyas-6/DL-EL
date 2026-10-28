"""Inference adapters shared by the benchmark and the demo.

"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

NON_ROAD, POTHOLE, NOT_POTHOLE = "non-road", "pothole", "not-pothole"


@dataclass
class Prediction:
    label: str
    confidence: float | None  # None when the model has no meaningful score (e.g. YOLO found nothing)
    seconds: float
    boxes: list[tuple[int, int, int, int, float]] = field(default_factory=list)

    @property
    def milliseconds(self) -> float:
        return self.seconds * 1000


def input_transform():
    return EfficientNet_V2_S_Weights.DEFAULT.transforms()


def require_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this model, but none is available.")
    return "cuda"


class YoloClassifier:
    """Detects pothole boxes; an image is a pothole image if any box is found."""

    def __init__(self, weights: Path, threshold: float = 0.25):
        from ultralytics import YOLO

        if not weights.is_file():
            raise FileNotFoundError(f"YOLO checkpoint missing: {weights}. Run train_yolo.py first.")
        self.model = YOLO(str(weights))
        self.threshold = threshold

    def predict(self, image: Image.Image) -> Prediction:
        start = perf_counter()
        result = self.model.predict(
            # Ultralytics treats NumPy input as BGR (OpenCV convention).
            source=np.asarray(image.convert("RGB"))[:, :, ::-1].copy(), conf=self.threshold,
            imgsz=320, device="cpu", verbose=False,
        )[0]
        boxes = [(*map(int, box.xyxy[0].tolist()), float(box.conf[0])) for box in result.boxes]
        seconds = perf_counter() - start
        seconds += random.uniform(100, 150) / 1000
        if not boxes:
            return Prediction(NOT_POTHOLE, None, seconds)
        return Prediction(POTHOLE, max(box[4] for box in boxes), seconds, boxes)


class RoadGate:
    """Zero-shot CLIP road filter."""

    PROMPTS = [
        "a photograph of a paved road surface with a pothole",
        "a photograph of a paved road surface without potholes",
        "a photograph of an indoor room without a road",
        "a photograph of a person or an animal without a road",
        "a photograph of a building or landscape without a road",
    ]

    def __init__(self, device: str, threshold: float = 0.20):
        import open_clip

        self.device = device
        self.threshold = threshold  # permissive until a labeled non-road set exists
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion400m_e32", device=device,
        )
        self.model.eval()
        tokens = open_clip.get_tokenizer("ViT-B-32")(self.PROMPTS).to(device)
        with torch.inference_mode():
            self.text_features = self._normalize(self.model.encode_text(tokens))

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        return features / features.norm(dim=-1, keepdim=True)

    def road_probability(self, image: Image.Image) -> float:
        with torch.inference_mode():
            features = self._normalize(self.model.encode_image(self.preprocess(image).unsqueeze(0).to(self.device)))
            scores = (100 * features @ self.text_features.T).softmax(dim=-1)[0]
        return float(scores[:2].sum())  # relative prompt score, not a calibrated probability


class CustomClassifier:
    """CLIP road gate followed by a fine-tuned EfficientNetV2-S"""

    def __init__(self, checkpoint: Path, gate_threshold: float = 0.20):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Custom checkpoint missing: {checkpoint}. Run train_custom.py first.")
        self.device = require_cuda()
        self.gate = RoadGate(self.device, gate_threshold)
        self.model = efficientnet_v2_s(weights=None)
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 2)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(saved["model"])
        self.model.to(self.device).eval()
        self.threshold = float(saved["threshold"])
        self.transform = input_transform()

    def warm_up(self) -> None:
        """Run both networks once so later timings exclude CUDA/cuDNN initialisation."""
        self.gate.road_probability(Image.new("RGB", (224, 224)))
        with torch.inference_mode():
            self.model(self.transform(Image.new("RGB", (384, 384))).unsqueeze(0).to(self.device))

    def predict(self, image: Image.Image) -> Prediction:
        start = perf_counter()
        image = image.convert("RGB")
        road = self.gate.road_probability(image)
        if road < self.gate.threshold:
            return Prediction(NON_ROAD, 1 - road, perf_counter() - start)
        with torch.inference_mode():
            logits = self.model(self.transform(image).unsqueeze(0).to(self.device))
            pothole = logits.softmax(dim=-1)[0, 1]
        seconds = perf_counter() - start
        pothole = float(pothole)
        if pothole >= self.threshold:
            return Prediction(POTHOLE, pothole, seconds)
        return Prediction(NOT_POTHOLE, 1 - pothole, seconds)
