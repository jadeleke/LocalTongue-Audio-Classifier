import logging
import os
import time
import uuid

import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, jsonify, request, render_template, redirect, url_for, flash
from flask_wtf.csrf import CSRFProtect, CSRFError
from werkzeug.utils import secure_filename
from pydub import AudioSegment

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"wav", "mp3"}
MODEL_PATH = os.environ.get("MODEL_PATH", "multiclass_model_aug_3_lang.keras")
LABELS = ["Akan", "Dagbani", "Ikposo"]
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20 MB

# Audio processing parameters
SR = 16000
DURATION = 15
N_MELS = 32
N_FFT = 1024
HOP_LENGTH = 1024
MAX_PAD_LEN = int(np.ceil(DURATION * SR / HOP_LENGTH))

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

_secret = os.environ.get("SECRET_KEY", "")
if not _secret:
    log.warning(
        "SECRET_KEY is not set. Using an insecure default — set the SECRET_KEY "
        "environment variable before deploying to production."
    )
    _secret = "dev-secret-change-in-production"
app.secret_key = _secret
csrf = CSRFProtect(app)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    m = tf.keras.models.load_model(MODEL_PATH)
    log.info("Model loaded from %s", MODEL_PATH)
    return m


# Loaded once at startup; tests can monkey-patch this
model = load_model()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features(file_path):
    """Return mel-spectrogram feature matrix (time_steps x n_mels), or raise on failure."""
    ext = os.path.splitext(file_path)[1].lower()
    tmp = None
    if ext == ".mp3":
        tmp = os.path.join(UPLOAD_FOLDER, f"tmp_{uuid.uuid4().hex}.wav")
        try:
            audio = (
                AudioSegment.from_file(file_path)
                .set_frame_rate(SR)
                .set_channels(1)[:DURATION * 1000]
            )
            audio.export(tmp, format="wav")
            file_path = tmp
        except Exception as e:
            raise RuntimeError(f"Could not convert MP3 to WAV: {e}") from e

    try:
        y, _ = librosa.load(file_path, sr=SR, duration=DURATION)
    except Exception as e:
        raise RuntimeError(f"Could not load audio file: {e}") from e
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)

    y = np.pad(y, (0, max(0, DURATION * SR - len(y))))[:DURATION * SR]
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = np.pad(log_mel, ((0, 0), (0, max(0, MAX_PAD_LEN - log_mel.shape[1]))), mode="constant")
    log_mel = log_mel[:, :MAX_PAD_LEN]
    return log_mel.T


def predict_language(file_path):
    """Return (label, confidence_percent) tuple, or raise on failure."""
    features = extract_features(file_path)
    input_tensor = np.expand_dims(features, axis=0)
    prediction = model.predict(input_tensor, verbose=0)
    pred_idx = int(np.argmax(prediction))
    label = LABELS[pred_idx]
    confidence = float(prediction[0][pred_idx]) * 100
    log.info("Prediction: %s (%.1f%%)", label, confidence)
    return label, confidence


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH, "labels": LABELS})


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            flash("No file selected.", "error")
            return redirect(request.url)

        file = request.files["file"]

        if not allowed_file(file.filename):
            flash("Unsupported file type. Please upload a .wav or .mp3 file.", "error")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        t0 = time.monotonic()
        try:
            pred_label, confidence = predict_language(filepath)
        except RuntimeError as e:
            log.warning("Prediction failed for %s: %s", filename, e)
            flash(f"Could not process audio: {e}", "error")
            return redirect(request.url)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

        elapsed = (time.monotonic() - t0) * 1000
        log.info("Request processed in %.0f ms", elapsed)

        return render_template(
            "index.html",
            filename=filename,
            label=pred_label,
            confidence=confidence,
        )

    return render_template("index.html")


@app.errorhandler(413)
def request_entity_too_large(_):
    flash("File too large. Maximum size is 20 MB.", "error")
    return redirect(url_for("index"))


@app.errorhandler(CSRFError)
def csrf_error(_):
    flash("Form expired or invalid. Please try again.", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
