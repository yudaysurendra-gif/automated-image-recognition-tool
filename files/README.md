# Automated Image Recognition Tool

A simple, self-contained image classification project built on a pretrained
**MobileNetV2** (ImageNet weights) convolutional neural network. It ships with
three ways to use it:

1. **Web UI** — drag-and-drop an image in the browser and see labeled predictions.
2. **REST API** — `POST /api/predict` for programmatic/JSON access.
3. **CLI** — batch-classify a folder of images from the terminal.

## Project Structure

```
automated-image-recognition-tool/
├── app/
│   ├── recognizer.py     # Core ML inference engine (model-agnostic wrapper)
│   ├── main.py           # Flask web app + REST API
│   └── cli.py            # Command-line batch runner
├── templates/
│   └── index.html         # Web UI
├── static/
│   └── uploads/           # Uploaded images are stored here at runtime
├── requirements.txt
└── README.md
```

## Setup

```bash
cd automated-image-recognition-tool
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> First run will download MobileNetV2 ImageNet weights (~14 MB) automatically.

## Running the Web App

```bash
cd app
python main.py
```

Then open **http://localhost:5000** in your browser, upload an image, and
view the top-5 predicted labels with confidence scores.

## Using the REST API

```bash
curl -X POST http://localhost:5000/api/predict \
  -F "image=@/path/to/photo.jpg"
```

Example response:

```json
{
  "filename": "photo.jpg",
  "inference_time_ms": 182.44,
  "predictions": [
    {"label": "Golden Retriever", "confidence": 0.8734},
    {"label": "Labrador Retriever", "confidence": 0.0521},
    {"label": "Cocker Spaniel", "confidence": 0.0198},
    {"label": "Kuvasz", "confidence": 0.0102},
    {"label": "Clumber", "confidence": 0.0067}
  ]
}
```

## Using the CLI

Classify a single image:
```bash
cd app
python cli.py /path/to/image.jpg
```

Classify every image in a folder:
```bash
python cli.py /path/to/folder/
```

Get raw JSON output:
```bash
python cli.py /path/to/image.jpg --json
```

## How It Works

- **Model**: MobileNetV2 pretrained on ImageNet (1,000 object categories),
  loaded via `tensorflow.keras.applications`.
- **Preprocessing**: images are resized to 224×224 and normalized using the
  same pipeline MobileNetV2 was trained with.
- **Inference**: a single forward pass produces class probabilities, and the
  top-k highest-confidence labels are returned.
- **Decoupled design**: `recognizer.py` has zero Flask/CLI dependencies, so
  the same `ImageRecognizer` class can be reused in a different interface
  (e.g. a batch job, a different web framework, or a notebook) without
  modification.

## Extending This Project

Some natural next steps if you want to build on this base:

- **Custom classes**: replace MobileNetV2 with a fine-tuned model trained on
  your own labeled dataset (e.g. via transfer learning) for domain-specific
  recognition (defect detection, species identification, product SKUs, etc).
- **Object detection**: swap the classifier for a detection model (YOLO,
  Faster R-CNN) to localize multiple objects per image, not just classify
  the whole image.
- **Async/queue processing**: for high-volume batch workloads, wrap
  `predict_batch` in a task queue (Celery/RQ) instead of synchronous calls.
- **Model versioning**: add a `models/` registry so multiple models can be
  swapped via a config flag or API parameter.

## Notes

- Max upload size is capped at 10 MB (`MAX_CONTENT_LENGTH` in `main.py`).
- Supported formats: PNG, JPG/JPEG, WEBP, BMP.
- No GPU required — MobileNetV2 is lightweight enough to run on CPU with
  reasonable latency (typically 100–300ms per image).
