"""Unit tests for core/stt.py."""
from __future__ import annotations

import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestWhisperSTT:
    def test_empty_bytes_returns_empty_string(self) -> None:
        from core.stt import WhisperSTT
        stt = WhisperSTT()
        assert stt.transcribe(b"") == ""

    def test_missing_whisper_returns_empty(self) -> None:
        import sys
        with patch.dict(sys.modules, {"whisper": None}):  # type: ignore[arg-type]
            from importlib import reload
            import core.stt
            reload(core.stt)
            stt = core.stt.WhisperSTT()
            result = stt.transcribe(b"\x00\x01" * 100)
        assert result == ""

    def test_transcribe_calls_model(self) -> None:
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hola mundo"}
        mock_whisper.load_model.return_value = mock_model

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            from importlib import reload
            import core.stt
            reload(core.stt)
            stt = core.stt.WhisperSTT(model_size="tiny")
            pcm = np.zeros(16000, dtype=np.int16).tobytes()
            result = stt.transcribe(pcm)

        assert result == "hola mundo"
        mock_model.transcribe.assert_called_once()

    def test_transcribe_strips_whitespace(self) -> None:
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  ¿qué ves?  "}
        mock_whisper.load_model.return_value = mock_model

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            from importlib import reload
            import core.stt
            reload(core.stt)
            stt = core.stt.WhisperSTT()
            pcm = np.zeros(1600, dtype=np.int16).tobytes()
            result = stt.transcribe(pcm)

        assert result == "¿qué ves?"

    def test_transcribe_handles_runtime_error(self) -> None:
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("CUDA error")
        mock_whisper.load_model.return_value = mock_model

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            from importlib import reload
            import core.stt
            reload(core.stt)
            stt = core.stt.WhisperSTT()
            pcm = np.zeros(1600, dtype=np.int16).tobytes()
            result = stt.transcribe(pcm)

        assert result == ""

    def test_model_loaded_once(self) -> None:
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "ok"}
        mock_whisper.load_model.return_value = mock_model

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            from importlib import reload
            import core.stt
            reload(core.stt)
            stt = core.stt.WhisperSTT()
            pcm = np.zeros(1600, dtype=np.int16).tobytes()
            stt.transcribe(pcm)
            stt.transcribe(pcm)

        mock_whisper.load_model.assert_called_once()
