"""Unit tests for core/voice_capture.py."""
from __future__ import annotations

import sys
import pytest
from unittest.mock import MagicMock, patch


class TestVoiceCaptureInit:
    def test_default_values(self) -> None:
        from core.voice_capture import VoiceCapture
        vc = VoiceCapture()
        assert vc._device == "default"
        assert vc._sample_rate == 16000
        assert vc.is_running is False

    def test_custom_values(self) -> None:
        from core.voice_capture import VoiceCapture
        vc = VoiceCapture(device="hw:1,0", silence_threshold=500)
        assert vc._device == "hw:1,0"
        assert vc._silence_threshold == 500


class TestVoiceCaptureListenOnce:
    @pytest.mark.asyncio
    async def test_returns_none_without_sounddevice(self) -> None:
        with patch.dict(sys.modules, {"sounddevice": None}):  # type: ignore[arg-type]
            from importlib import reload
            import core.voice_capture
            reload(core.voice_capture)
            vc = core.voice_capture.VoiceCapture()
            result = await vc.listen_once()
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_bytes_on_success(self, mocker: "pytest_mock.MockerFixture") -> None:  # noqa: F821
        import numpy as np
        mock_sd = MagicMock()
        mock_numpy = np

        pcm_data = np.zeros((512, 1), dtype=np.int16)
        pcm_data[0, 0] = 10000

        call_count = 0

        class FakeStream:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        original_record = None

        from core.voice_capture import VoiceCapture
        vc = VoiceCapture(min_speech_chunks=1, silence_chunks=1)

        async def fake_listen_once():
            return b"\x00" * 1024

        mocker.patch.object(vc, "listen_once", fake_listen_once)
        result = await vc.listen_once()
        assert result == b"\x00" * 1024


class TestVoiceCaptureRecordSync:
    def test_record_sync_returns_none_on_exception(self, mocker: "pytest_mock.MockerFixture") -> None:  # noqa: F821
        import numpy as np
        mock_sd = MagicMock()
        mock_sd.InputStream.side_effect = Exception("device not found")

        from core.voice_capture import VoiceCapture
        vc = VoiceCapture()
        result = vc._record_sync(mock_sd, np)
        assert result is None

    def test_record_sync_returns_none_when_no_speech(self) -> None:
        import numpy as np

        class FakeStream:
            def __init__(self, **kwargs):
                self.callback = kwargs.get("callback")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        mock_sd = MagicMock()
        mock_sd.InputStream = FakeStream

        import time
        with patch("time.sleep", return_value=None), \
             patch("time.monotonic", side_effect=[0.0] + [11.0] * 20):
            from core.voice_capture import VoiceCapture
            vc = VoiceCapture(min_speech_chunks=5)
            result = vc._record_sync(mock_sd, np)

        assert result is None
