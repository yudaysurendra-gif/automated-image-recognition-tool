"""
recognizer.py
--------------
Core image recognition engine.

Uses a pretrained MobileNetV2 (ImageNet weights) for general-purpose
image classification. The module is intentionally decoupled from any
UI/CLI layer so it can be reused in a web app, batch script, or API.
"""

import io
import os
import time
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
from PIL import Image

# TensorFlow / Keras pretrained model
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image as keras_image


@dataclass
class Prediction:
    label: str
    confidence: float


@dataclass
class RecognitionResult:
    filename: str
    predictions: List[Prediction] = field(default_factory=list)
    inference_time_ms: float = 0.0

    def to_dict(self):
        return {
            "filename": self.filename,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "predictions": [
                {"label": p.label, "confidence": round(p.confidence, 4)}
                for p in self.predictions
            ],
        }


class ImageRecognizer:
    """
    Wraps a pretrained CNN for image classification.

    Loading the model is expensive, so this class is designed to be
    instantiated once (e.g. as a singleton at app startup) and reused
    across many prediction calls.
    """

    TARGET_SIZE = (224, 224)

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._model = None  # lazy-loaded

    @property
    def model(self):
        if self._model is None:
            print("[ImageRecognizer] Loading MobileNetV2 (ImageNet weights)...")
            self._model = MobileNetV2(weights="imagenet")
            print("[ImageRecognizer] Model loaded.")
        return self._model

    def _load_image(self, source: Union[str, bytes, io.BytesIO]) -> Image.Image:
        """Accepts a filepath, raw bytes, or a BytesIO stream."""
        if isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        elif isinstance(source, io.BytesIO):
            img = Image.open(source)
        elif isinstance(source, str):
            img = Image.open(source)
        else:
            raise ValueError(f"Unsupported image source type: {type(source)}")
        return img.convert("RGB")

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        img = img.resize(self.TARGET_SIZE)
        arr = keras_image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        return arr

    def predict(
        self,
        source: Union[str, bytes, io.BytesIO],
        filename: str = "uploaded_image",
    ) -> RecognitionResult:
        """Run classification on a single image and return top-k results."""
        start = time.perf_counter()

        img = self._load_image(source)
        batch = self._preprocess(img)

        raw_preds = self.model.predict(batch, verbose=0)
        decoded = decode_predictions(raw_preds, top=self.top_k)[0]

        elapsed_ms = (time.perf_counter() - start) * 1000

        predictions = [
            Prediction(label=label.replace("_", " ").title(), confidence=float(prob))
            for (_, label, prob) in decoded
        ]

        return RecognitionResult(
            filename=filename,
            predictions=predictions,
            inference_time_ms=elapsed_ms,
        )

    def predict_batch(self, filepaths: List[str]) -> List[RecognitionResult]:
        """Run classification over a list of local file paths."""
        results = []
        for path in filepaths:
            filename = os.path.basename(path)
            try:
                result = self.predict(path, filename=filename)
            except Exception as exc:
                result = RecognitionResult(
                    filename=filename,
                    predictions=[Prediction(label=f"ERROR: {exc}", confidence=0.0)],
                )
            results.append(result)
        return results


# Singleton instance used across the app (loaded lazily on first predict call)
recognizer = ImageRecognizer(top_k=5)
