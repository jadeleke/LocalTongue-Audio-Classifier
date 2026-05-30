# LocalTongue Audio Classifier

A Flask web application that identifies the local African language spoken in an uploaded audio file. Upload a `.wav` or `.mp3` clip and the app returns the predicted language and a confidence score — no account or API key required.

**Supported languages:** Akan · Dagbani · Ikposo

## How It Works

1. User uploads a `.wav` or `.mp3` file (max 20 MB, up to 15 seconds used)
2. MP3 files are converted to WAV internally via `pydub`
3. Audio is resampled to 16 kHz mono and a log mel-spectrogram is extracted with `librosa`
4. A trained TensorFlow/Keras multiclass model classifies the spectrogram
5. The predicted language and confidence percentage are displayed in the browser

## Quick Start

**Prerequisites:** Python 3.9+ and [FFmpeg](https://ffmpeg.org/download.html) on your PATH (required for MP3 support).

```bash
git clone https://github.com/jadeleke/LocalTongue-Audio-Classifier.git
cd LocalTongue-Audio-Classifier

python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

> The model file `multiclass_model_aug_3_lang.keras` must be present at the project root before starting the app.

## Production Deployment

Use Gunicorn instead of the Flask dev server and set a strong secret key:

```bash
export SECRET_KEY="your-random-secret-here"
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

A `SECRET_KEY` warning is logged at startup if the environment variable is not set.

## Health Check

```
GET /health
```

Returns:

```json
{
  "status": "ok",
  "model": "multiclass_model_aug_3_lang.keras",
  "labels": ["Akan", "Dagbani", "Ikposo"]
}
```

Useful for load balancers and uptime monitors.

## Running Tests

Tests mock all heavy dependencies (TensorFlow, librosa, pydub) so they run without a GPU or the model file.

```bash
pip install pytest pytest-flask flask-wtf
pytest tests/ -v
```

## Project Structure

```
LocalTongue-Audio-Classifier/
├── app.py                              # Flask application
├── requirements.txt                    # Python dependencies
├── pytest.ini                          # Test configuration
├── conftest.py                         # Pytest path setup
├── multiclass_model_aug_3_lang.keras   # Trained Keras model (not in git)
├── templates/
│   └── index.html                      # Upload form and results page
├── static/
│   └── style.css
├── tests/
│   └── test_app.py                     # 21 unit + integration tests
├── .github/
│   └── workflows/
│       └── ci.yml                      # GitHub Actions CI (Python 3.9–3.11)
└── uploads/                            # Temporary upload directory (git-ignored)
```

## Configuration

All tunable values live at the top of `app.py`:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `multiclass_model_aug_3_lang.keras` | Path to the Keras model (overridable via `MODEL_PATH` env var) |
| `SR` | `16000` | Audio sample rate (Hz) |
| `DURATION` | `15` | Max audio duration used (seconds) |
| `N_MELS` | `32` | Number of mel bands |
| `MAX_CONTENT_LENGTH` | `20 MB` | Maximum upload size |

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: Model not found` | Place `multiclass_model_aug_3_lang.keras` in the project root, or set `MODEL_PATH` env var |
| MP3 upload fails | Install FFmpeg and ensure it is on your system PATH |
| `SECRET_KEY` warning in logs | Set the `SECRET_KEY` environment variable before starting the app |
| CSRF error on form submit | Ensure cookies are enabled in your browser |
