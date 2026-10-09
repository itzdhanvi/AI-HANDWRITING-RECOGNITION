import os
import base64
import json
import uuid
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "handwriting_cnn.keras"

# Use a persistent disk path on Render when configured.
# Locally, store notes in the project's data folder.
import os

STORAGE_DIR = Path(
    os.environ.get("STORAGE_DIR", str(BASE / "data"))
)

NOTES_PATH = STORAGE_DIR / "notes.json"

app = Flask(__name__)

# Must match the exact class-index order used during training.
LABELS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t"
]

# Tune this using a labeled validation set.
CONFIDENCE_THRESHOLD = 0.35

model = None
if MODEL_PATH.exists():
    model = tf.keras.models.load_model(MODEL_PATH)
    print("CNN model loaded:", MODEL_PATH)
    print("Model output shape:", model.output_shape)
    print("Number of labels:", len(LABELS))

    if model.output_shape[-1] != len(LABELS):
        raise ValueError(
            "Model output count does not match LABELS. "
            "Check your training class mapping."
        )
else:
    print("WARNING: Model not found:", MODEL_PATH)


# --------------------------------------------------
# Notes storage
# --------------------------------------------------

# def load_notes():
#     if not NOTES_PATH.exists():
#         return []

#     try:
#         return json.loads(
#             NOTES_PATH.read_text(encoding="utf-8")
#         )
#     except (json.JSONDecodeError, OSError):
#         return []


# def save_notes(notes):
#     NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
#     NOTES_PATH.write_text(
#         json.dumps(notes, indent=2, ensure_ascii=False),
#         encoding="utf-8"
#     )

# --------------------------------------------------
# Notes storage
# --------------------------------------------------

def load_notes():
    try:
        if not NOTES_PATH.exists():
            return []

        content = NOTES_PATH.read_text(encoding="utf-8")

        if not content.strip():
            return []

        notes = json.loads(content)

        if not isinstance(notes, list):
            app.logger.error("Notes file must contain a JSON list.")
            return []

        return notes

    except (json.JSONDecodeError, OSError):
        app.logger.exception(
            "Could not read notes from %s", NOTES_PATH
        )
        return []


def save_notes(notes):
    try:
        NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temporary file before replacing the original.
        temp_path = NOTES_PATH.with_suffix(".tmp")

        temp_path.write_text(
            json.dumps(notes, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        temp_path.replace(NOTES_PATH)

        app.logger.info("Notes saved successfully to %s", NOTES_PATH)

    except OSError:
        app.logger.exception(
            "Failed to write notes to %s", NOTES_PATH
        )
        raise
# --------------------------------------------------
# Image processing
# --------------------------------------------------

def rgba_to_gray(image):
    """Convert an RGBA, RGB, or grayscale image to grayscale."""
    if image.ndim == 2:
        return image.astype(np.uint8)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def get_ink_mask(image):
    """
    Create a binary mask:
    handwriting = 255, background = 0.

    Assumes dark handwriting on a light canvas.
    """
    gray = rgba_to_gray(image)

    # Otsu thresholding adapts to the image histogram.
    _, ink = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return ink


def prepare_character(character_image):
    """
    Prepare one segmented character for a 28x28 CNN.

    IMPORTANT:
    This assumes the trained model expects white ink on black,
    normalized to [0, 1], without an EMNIST-specific transpose.
    Confirm these assumptions against train_model.py.
    """
    if character_image is None or character_image.size == 0:
        return None

    ink = get_ink_mask(character_image)

    points = cv2.findNonZero(ink)
    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)
    crop = ink[y:y + h, x:x + w]

    if crop.size == 0:
        return None

    # Keep the character aspect ratio inside a 20x20 area.
    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    crop = cv2.resize(
        crop,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Center on a 28x28 black canvas.
    output = np.zeros((28, 28), dtype=np.uint8)

    x0 = (28 - new_w) // 2
    y0 = (28 - new_h) // 2

    output[y0:y0 + new_h, x0:x0 + new_w] = crop

    # Convert to float32, range [0, 1].
    return output.astype(np.float32) / 255.0


# --------------------------------------------------
# Line segmentation
# --------------------------------------------------

def find_line_groups(ink):
    """Find horizontal handwriting bands using row projections."""
    if ink is None or ink.size == 0:
        return []

    row_counts = np.count_nonzero(ink, axis=1)

    # Ignore isolated marks and extremely sparse rows.
    threshold = max(1, int(ink.shape[1] * 0.001))
    active = row_counts >= threshold

    indices = np.flatnonzero(active)
    if len(indices) == 0:
        return []

    groups = []
    start = previous = int(indices[0])

    for index in indices[1:]:
        index = int(index)

        # Merge row bands separated by small vertical gaps.
        if index - previous > 4:
            groups.append((start, previous + 1))
            start = index

        previous = index

    groups.append((start, previous + 1))

    # Add padding and merge nearby bands.
    padded = []
    for start, end in groups:
        start = max(0, start - 3)
        end = min(ink.shape[0], end + 3)

        if padded and start - padded[-1][1] <= 5:
            padded[-1] = (padded[-1][0], end)
        else:
            padded.append((start, end))

    return padded


# --------------------------------------------------
# Character segmentation
# --------------------------------------------------

def segment_characters(ink, max_gap=1, padding=2):
    """
    Split a line using vertical projections.

    Tiny gaps are bridged to reduce accidental splitting.
    This remains a heuristic and may struggle with cursive
    writing or characters that touch.
    """
    if ink is None or ink.size == 0:
        return []

    column_counts = np.count_nonzero(ink, axis=0)
    active = column_counts > 0

    # Bridge gaps of up to max_gap blank columns.
    filled = active.copy()

    for gap_size in range(1, max_gap + 1):
        if gap_size + 1 >= len(active):
            break

        for offset in range(1, gap_size + 1):
            filled[offset:] |= (
                active[:-offset] & active[offset:]
            )

    indices = np.flatnonzero(filled)
    if len(indices) == 0:
        return []

    groups = []
    start = previous = int(indices[0])

    for index in indices[1:]:
        index = int(index)

        if index > previous + 1:
            left = max(0, start - padding)
            right = min(ink.shape[1], previous + 1 + padding)

            if right - left >= 2:
                groups.append((left, right))

            start = index

        previous = index

    left = max(0, start - padding)
    right = min(ink.shape[1], previous + 1 + padding)

    if right - left >= 2:
        groups.append((left, right))

    return groups


# --------------------------------------------------
# CNN prediction
# --------------------------------------------------

def predict_character(character_image):
    """Return the predicted character and confidence."""
    if model is None:
        return "?", 0.0

    prepared = prepare_character(character_image)
    if prepared is None:
        return "?", 0.0

    # This assumes the model input shape is (None, 28, 28, 1).
    batch = prepared.reshape(1, 28, 28, 1)

    probabilities = model.predict(batch, verbose=0)[0]

    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])

    if confidence < CONFIDENCE_THRESHOLD:
        return "?", confidence

    return LABELS[index], confidence


# --------------------------------------------------
# Full handwriting recognition
# --------------------------------------------------

def recognize(image):
    if model is None:
        return {
            "text": "",
            "confidence": 0.0,
            "error": "CNN model not found in the models folder."
        }

    if image is None or image.size == 0:
        return {
            "text": "",
            "confidence": 0.0,
            "error": "No canvas image received."
        }

    ink = get_ink_mask(image)

    if cv2.countNonZero(ink) == 0:
        return {
            "text": "",
            "confidence": 0.0,
            "error": "The canvas is blank. Please write something first."
        }

    lines = find_line_groups(ink)

    if not lines:
        return {
            "text": "",
            "confidence": 0.0,
            "error": "No handwriting detected."
        }

    recognized_lines = []
    confidences = []

    for y1, y2 in lines:
        line_ink = ink[y1:y2, :]

        # Segment the binary line, then crop from the original image.
        groups = segment_characters(line_ink)

        if not groups:
            continue

        line_text = []
        previous_end = None

        # Estimate word spacing relative to median character width.
        widths = [b - a for a, b in groups]
        median_width = float(np.median(widths)) if widths else 1.0

        for a, b in groups:
            if previous_end is not None:
                gap = a - previous_end

                # Heuristic only: tune for your canvas and handwriting.
                if gap > max(10, median_width * 0.65):
                    line_text.append(" ")

            # Use the original RGBA/RGB image for character preprocessing.
            character_image = image[y1:y2, a:b]

            character, confidence = predict_character(
                character_image
            )

            line_text.append(character)
            confidences.append(confidence)
            previous_end = b

        recognized_lines.append("".join(line_text).strip())

    text = "\n".join(
        line for line in recognized_lines if line
    )

    return {
        "text": text,
        "confidence": (
            float(np.mean(confidences)) if confidences else 0.0
        ),
        "error": None if text else "Could not recognize the handwriting."
    }


# --------------------------------------------------
# Flask routes
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# @app.post("/save")
# def save():
#     payload = request.get_json(silent=True) or {}
#     data = payload.get("image", "")

#     if not isinstance(data, str) or "," not in data:
#         return jsonify(
#             ok=False,
#             error="Canvas image is missing."
#         ), 400

#     try:
#         encoded = data.split(",", 1)[1]
#         raw = base64.b64decode(encoded, validate=True)
#     except (ValueError, base64.binascii.Error):
#         return jsonify(
#             ok=False,
#             error="Invalid canvas image data."
#         ), 400

#     array = np.frombuffer(raw, dtype=np.uint8)
#     decoded = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)

#     if decoded is None:
#         return jsonify(
#             ok=False,
#             error="Could not read the canvas image."
#         ), 400

#     if decoded.ndim == 2:
#         image = decoded
#     elif decoded.shape[2] == 4:
#         image = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
#     elif decoded.shape[2] == 3:
#         image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
#     else:
#         return jsonify(
#             ok=False,
#             error="Unsupported canvas image format."
#         ), 400

#     result = recognize(image)

#     if result["error"]:
#         return jsonify(
#             ok=False,
#             error=result["error"],
#             text=result["text"],
#             confidence=round(result["confidence"] * 100, 2)
#         ), 400

#     note = {
#         "id": str(uuid.uuid4()),
#         "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "text": result["text"],
#         "confidence": round(result["confidence"] * 100, 2)
#     }

#     notes = load_notes()
#     notes.append(note)

#     try:
#         save_notes(notes)
#     except OSError:
#         app.logger.exception("Failed to save notes")
#         return jsonify(
#             ok=False,
#             error="Recognition succeeded, but the note could not be saved."
#         ), 500

#     return jsonify(
#         ok=True,
#         note_id=note["id"],
#         text=note["text"],
#         confidence=note["confidence"]
#     )

@app.post("/save")
def save():
    payload = request.get_json(silent=True) or {}
    data = payload.get("image", "")

    # Validate canvas image
    if not isinstance(data, str) or "," not in data:
        return jsonify(
            ok=False,
            error="Canvas image is missing."
        ), 400

    # Decode the base64 image
    try:
        encoded = data.split(",", 1)[1]
        raw = base64.b64decode(encoded, validate=True)

    except (ValueError, base64.binascii.Error):
        return jsonify(
            ok=False,
            error="Invalid canvas image data."
        ), 400

    # Read image using OpenCV
    try:
        array = np.frombuffer(raw, dtype=np.uint8)
        decoded = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)

        if decoded is None:
            return jsonify(
                ok=False,
                error="Could not read the canvas image."
            ), 400

        # Convert image into RGB format
        if decoded.ndim == 2:
            image = decoded

        elif decoded.shape[2] == 4:
            image = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)

        elif decoded.shape[2] == 3:
            image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

        else:
            return jsonify(
                ok=False,
                error="Unsupported canvas image format."
            ), 400

        # Run handwriting recognition
        result = recognize(image)

    except Exception:
        app.logger.exception("Handwriting recognition failed")
        return jsonify(
            ok=False,
            error="Handwriting recognition failed. Check the server logs."
        ), 500

    if result.get("error"):
        return jsonify(
            ok=False,
            error=result["error"],
            text=result.get("text", ""),
            confidence=round(
                float(result.get("confidence", 0)) * 100, 2
            )
        ), 400

    # Prepare the note
    note = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": result.get("text", ""),
        "confidence": round(
            float(result.get("confidence", 0)) * 100, 2
        )
    }

    # Save note
    try:
        notes = load_notes()
        notes.append(note)
        save_notes(notes)

    except Exception:
        app.logger.exception("Failed to save note")
        return jsonify(
            ok=False,
            error="Recognition succeeded, but the note could not be saved."
        ), 500

    # Return JSON expected by index.html
    return jsonify(
        ok=True,
        note_id=note["id"],
        text=note["text"],
        confidence=note["confidence"]
    ), 200


@app.route("/notes")
def notes():
    return render_template(
        "notes.html",
        notes=list(reversed(load_notes()))
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )