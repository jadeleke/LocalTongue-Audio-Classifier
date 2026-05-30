"""
Tests for the LocalTongue audio classifier Flask app.

The Keras model and librosa are mocked throughout so tests run without
a trained model file or real audio data.
"""
import io
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub out heavy/optional dependencies before importing app so tests run
# without TensorFlow, librosa, or pydub installed.
# ---------------------------------------------------------------------------
_MOCK_MODEL = MagicMock()
_MOCK_MODEL.predict.return_value = np.array([[0.1, 0.8, 0.1]])  # Dagbani wins


def _make_tf_stub():
    tf = types.ModuleType("tensorflow")
    keras = types.ModuleType("tensorflow.keras")
    models = types.ModuleType("tensorflow.keras.models")
    models.load_model = MagicMock(return_value=_MOCK_MODEL)
    keras.models = models
    tf.keras = keras
    return tf


def _make_librosa_stub():
    librosa = types.ModuleType("librosa")
    feature = types.ModuleType("librosa.feature")
    feature.melspectrogram = MagicMock(return_value=np.zeros((32, 235)))
    librosa.feature = feature
    librosa.load = MagicMock(return_value=(np.zeros(16000 * 15), 16000))
    librosa.power_to_db = MagicMock(side_effect=lambda mel, ref: mel)
    return librosa


def _make_pydub_stub():
    pydub = types.ModuleType("pydub")
    seg = MagicMock()
    seg.return_value.set_frame_rate.return_value.set_channels.return_value.__getitem__.return_value.export = MagicMock()
    pydub.AudioSegment = seg
    return pydub


_tf_stub = _make_tf_stub()
sys.modules.setdefault("tensorflow", _tf_stub)
sys.modules.setdefault("tensorflow.keras", _tf_stub.keras)
sys.modules.setdefault("tensorflow.keras.models", _tf_stub.keras.models)

_librosa_stub = _make_librosa_stub()
sys.modules.setdefault("librosa", _librosa_stub)
sys.modules.setdefault("librosa.feature", _librosa_stub.feature)

_pydub_stub = _make_pydub_stub()
sys.modules.setdefault("pydub", _pydub_stub)

# Also ensure the model file "exists" so load_model() is called rather than
# raising FileNotFoundError.
with patch("os.path.exists", return_value=True):
    import app as _app_module  # noqa: E402 – must come after sys.modules patch

# Replace the real model with the mock after import.
_app_module.model = _MOCK_MODEL


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

FAKE_FEATURES = np.zeros((_app_module.MAX_PAD_LEN, _app_module.N_MELS))
FAKE_WAV_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 20  # minimal stub


def _wav_upload(filename="test.wav", data=None):
    """Build a FileStorage-like bytes payload for the upload form."""
    return (io.BytesIO(data or FAKE_WAV_BYTES), filename)


# ---------------------------------------------------------------------------
# Unit tests – pure functions
# ---------------------------------------------------------------------------

class TestAllowedFile:
    def test_wav_allowed(self):
        assert _app_module.allowed_file("clip.wav")

    def test_mp3_allowed(self):
        assert _app_module.allowed_file("clip.MP3")

    def test_txt_rejected(self):
        assert not _app_module.allowed_file("notes.txt")

    def test_no_extension_rejected(self):
        assert not _app_module.allowed_file("noextension")

    def test_double_extension_wav(self):
        # secure_filename keeps the last extension
        assert _app_module.allowed_file("tricky.exe.wav")


class TestExtractFeatures:
    """extract_features() is tested with mocked librosa to avoid I/O."""

    @patch("app.librosa.load", return_value=(np.zeros(16000 * 15), 16000))
    @patch("app.librosa.feature.melspectrogram", return_value=np.zeros((32, 235)))
    @patch("app.librosa.power_to_db", side_effect=lambda mel, ref: mel)
    def test_returns_correct_shape(self, mock_db, mock_mel, mock_load, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.write_bytes(FAKE_WAV_BYTES)
        features = _app_module.extract_features(str(wav))
        assert features is not None
        assert features.shape[1] == _app_module.N_MELS

    @patch("app.librosa.load", side_effect=Exception("corrupt file"))
    def test_raises_on_load_failure(self, mock_load, tmp_path):
        wav = tmp_path / "bad.wav"
        wav.write_bytes(b"not audio")
        with pytest.raises(RuntimeError, match="Could not load audio file"):
            _app_module.extract_features(str(wav))

    @patch("app.AudioSegment.from_file", side_effect=Exception("bad mp3"))
    def test_raises_on_mp3_conversion_failure(self, mock_audio, tmp_path):
        mp3 = tmp_path / "bad.mp3"
        mp3.write_bytes(b"not mp3")
        with pytest.raises(RuntimeError, match="Could not convert MP3 to WAV"):
            _app_module.extract_features(str(mp3))


class TestPredictLanguage:
    @patch("app.extract_features", return_value=FAKE_FEATURES)
    def test_returns_label_and_confidence(self, mock_feat, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.write_bytes(FAKE_WAV_BYTES)
        _MOCK_MODEL.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        label, conf = _app_module.predict_language(str(wav))
        assert label == "Dagbani"
        assert pytest.approx(conf, abs=0.01) == 80.0

    @patch("app.extract_features", return_value=FAKE_FEATURES)
    def test_first_class_wins(self, mock_feat, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.write_bytes(FAKE_WAV_BYTES)
        _MOCK_MODEL.predict.return_value = np.array([[0.9, 0.05, 0.05]])
        label, conf = _app_module.predict_language(str(wav))
        assert label == "Akan"
        assert pytest.approx(conf, abs=0.01) == 90.0

    @patch("app.extract_features", side_effect=RuntimeError("bad audio"))
    def test_propagates_extraction_error(self, mock_feat, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.write_bytes(FAKE_WAV_BYTES)
        with pytest.raises(RuntimeError):
            _app_module.predict_language(str(wav))


# ---------------------------------------------------------------------------
# Integration tests – Flask test client
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    _app_module.app.config["TESTING"] = True
    _app_module.app.config["WTF_CSRF_ENABLED"] = False
    _app_module.app.config["WTF_CSRF_CHECK_DEFAULT"] = False
    with _app_module.app.test_client() as c:
        yield c


@pytest.fixture()
def csrf_client():
    """Client with CSRF enabled for testing CSRF rejection."""
    _app_module.app.config["TESTING"] = True
    _app_module.app.config["WTF_CSRF_ENABLED"] = True
    _app_module.app.config["WTF_CSRF_CHECK_DEFAULT"] = True
    with _app_module.app.test_client() as c:
        yield c
    # Reset to disabled so other tests are unaffected
    _app_module.app.config["WTF_CSRF_ENABLED"] = False
    _app_module.app.config["WTF_CSRF_CHECK_DEFAULT"] = False


class TestIndexRoute:
    def test_get_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"LocalTongue" in resp.data

    def test_post_no_file_redirects(self, client):
        resp = client.post("/", data={})
        assert resp.status_code in (302, 200)

    def test_post_empty_filename_redirects(self, client):
        resp = client.post(
            "/",
            data={"file": (io.BytesIO(b""), "")},
            content_type="multipart/form-data",
        )
        assert resp.status_code in (302, 200)

    def test_post_invalid_extension_shows_error(self, client):
        resp = client.post(
            "/",
            data={"file": (io.BytesIO(b"data"), "audio.exe")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"Unsupported file type" in resp.data

    @patch("app.predict_language", return_value=("Dagbani", 80.0))
    def test_post_valid_file_shows_prediction(self, mock_pred, client):
        resp = client.post(
            "/",
            data={"file": _wav_upload()},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"Dagbani" in resp.data
        assert b"80.00" in resp.data

    @patch("app.predict_language", side_effect=RuntimeError("corrupt audio"))
    def test_post_processing_error_shows_message(self, mock_pred, client):
        resp = client.post(
            "/",
            data={"file": _wav_upload()},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"Could not process audio" in resp.data

    @patch("app.predict_language", return_value=("Akan", 92.5))
    def test_uploaded_file_is_cleaned_up(self, mock_pred, client, tmp_path):
        # After a successful prediction the uploaded file should be removed.
        resp = client.post(
            "/",
            data={"file": _wav_upload("cleanup_test.wav")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert resp.status_code == 200
        leftover = os.path.join(_app_module.app.config["UPLOAD_FOLDER"], "cleanup_test.wav")
        assert not os.path.exists(leftover), "Uploaded file was not cleaned up"


class TestErrorHandlers:
    def test_413_redirects_with_message(self, client):
        big_data = b"x" * (21 * 1024 * 1024)
        resp = client.post(
            "/",
            data={"file": (io.BytesIO(big_data), "huge.wav")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert resp.status_code < 500

    def test_csrf_rejection_without_token(self, csrf_client):
        resp = csrf_client.post(
            "/",
            data={"file": _wav_upload()},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        # Should redirect back with an error message, not 500
        assert resp.status_code == 200
        assert b"expired" in resp.data or b"invalid" in resp.data


class TestHealthRoute:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "labels" in data
        assert "Akan" in data["labels"]
