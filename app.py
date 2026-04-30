import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from pydub import AudioSegment

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
MODEL_PATH = 'multiclass_model_aug_3_lang.keras'
LABELS = ["Akan", "Dagbani", "Ikposo"]

# Audio processing parameters
SR = 16000
DURATION = 15
N_MELS = 32
N_FFT = 1024
HOP_LENGTH = 1024
MAX_PAD_LEN = int(np.ceil(DURATION * SR / HOP_LENGTH))

# App setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
else:
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# Utility: Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Utility: Extract features
def extract_features(file_path):
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".mp3":
            tmp = os.path.join(UPLOAD_FOLDER, "tmp.wav")
            audio = AudioSegment.from_file(file_path).set_frame_rate(SR).set_channels(1)[:DURATION * 1000]
            audio.export(tmp, format="wav")
            file_path = tmp

        y, _ = librosa.load(file_path, sr=SR, duration=DURATION)
        y = np.pad(y, (0, max(0, DURATION * SR - len(y))))[:DURATION * SR]

        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = np.pad(log_mel, ((0, 0), (0, max(0, MAX_PAD_LEN - log_mel.shape[1]))), mode='constant')
        log_mel = log_mel[:, :MAX_PAD_LEN]
        return log_mel.T
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Utility: Predict language
def predict_language(file_path):
    features = extract_features(file_path)
    if features is None:
        return None, None

    input_tensor = np.expand_dims(features, axis=0)
    prediction = model.predict(input_tensor, verbose=0)
    pred_idx = np.argmax(prediction)
    return LABELS[pred_idx], float(prediction[0][pred_idx]) * 100

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            pred_label, confidence = predict_language(filepath)
            return render_template("index.html", filename=filename, label=pred_label, confidence=confidence)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
