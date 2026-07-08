"""
main.py
-------
Flask web application for the Automated Image Recognition Tool.

Routes:
    GET  /            -> Upload form (web UI)
    POST /             -> Handle image upload, run inference, render results
    POST /api/predict  -> JSON API endpoint (multipart/form-data, field "image")
    GET  /health       -> Simple health check
"""

import os
import uuid

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

from recognizer import recognizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("image")

        if file is None or file.filename == "":
            error = "Please choose an image file to upload."
        elif not allowed_file(file.filename):
            error = f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        else:
            safe_name = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{safe_name}"
            save_path = os.path.join(UPLOAD_FOLDER, unique_name)
            file.save(save_path)

            try:
                recognition_result = recognizer.predict(save_path, filename=safe_name)
                result = recognition_result.to_dict()
                image_url = url_for("static", filename=f"uploads/{unique_name}")
            except Exception as exc:
                error = f"Inference failed: {exc}"

    return render_template("index.html", result=result, error=error, image_url=image_url)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API: multipart/form-data with an 'image' field."""
    file = request.files.get("image")

    if file is None or file.filename == "":
        return jsonify({"error": "No image file provided under field 'image'."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    try:
        image_bytes = file.read()
        result = recognizer.predict(image_bytes, filename=secure_filename(file.filename))
        return jsonify(result.to_dict())
    except Exception as exc:
        return jsonify({"error": f"Inference failed: {exc}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
