"""Unit tests for services/stream_reader.py."""
from __future__ import annotations

import asyncio
import struct

import cv2
import numpy as np
import pytest

from services.stream_reader import StreamReader, _decode_jpeg, _extract_frames


def _make_jpeg(width: int = 64, height: int = 64) -> bytes:
    """Generate a minimal valid JPEG in memory."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[0, 0] = [128, 64, 32]
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class TestExtractFrames:
    def test_empty_buffer_returns_nothing(self) -> None:
        frames, remainder = _extract_frames(b"")
        assert frames == []
        assert remainder == b""

    def test_single_complete_frame(self) -> None:
        jpeg = _make_jpeg()
        frames, remainder = _extract_frames(jpeg)
        assert len(frames) == 1
        assert frames[0] == jpeg
        assert remainder == b""

    def test_frame_with_trailing_data(self) -> None:
        jpeg = _make_jpeg()
        buf = jpeg + b"garbage"
        frames, remainder = _extract_frames(buf)
        assert len(frames) == 1
        assert remainder == b"garbage"

    def test_multiple_frames(self) -> None:
        j1 = _make_jpeg()
        j2 = _make_jpeg(32, 32)
        buf = j1 + j2
        frames, remainder = _extract_frames(buf)
        assert len(frames) == 2
        assert remainder == b""

    def test_partial_frame_stays_in_buffer(self) -> None:
        jpeg = _make_jpeg()
        partial = jpeg[:len(jpeg) // 2]
        frames, remainder = _extract_frames(partial)
        assert frames == []
        assert remainder == partial

    def test_garbage_before_frame(self) -> None:
        jpeg = _make_jpeg()
        buf = b"\x00\x01\x02" + jpeg
        frames, remainder = _extract_frames(buf)
        assert len(frames) == 1


class TestDecodeJpeg:
    def test_valid_jpeg_returns_array(self) -> None:
        jpeg = _make_jpeg(64, 64)
        frame = _decode_jpeg(jpeg)
        assert frame is not None
        assert frame.shape == (64, 64, 3)
        assert frame.dtype == np.uint8

    def test_invalid_data_returns_none(self) -> None:
        frame = _decode_jpeg(b"not a jpeg")
        assert frame is None

    def test_empty_bytes_returns_none(self) -> None:
        frame = _decode_jpeg(b"")
        assert frame is None


class TestStreamReaderInit:
    def test_default_values(self) -> None:
        sr = StreamReader("http://example.com/stream.mjpg")
        assert sr._url == "http://example.com/stream.mjpg"
        assert sr._max_retries == 10
        assert sr._running is False

    def test_custom_values(self) -> None:
        sr = StreamReader("http://host/stream", max_retries=3, timeout=5.0)
        assert sr._max_retries == 3
        assert sr._timeout == 5.0

    def test_initial_frame_is_none(self) -> None:
        sr = StreamReader("http://host/stream")
        assert sr._latest_frame is None


class TestStreamReaderFrames:
    @pytest.mark.asyncio
    async def test_get_frame_returns_none_before_start(self) -> None:
        sr = StreamReader("http://host/stream")
        frame = await sr.get_frame()
        assert frame is None

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self) -> None:
        sr = StreamReader("http://host/stream")
        sr._running = True
        await sr.stop()
        assert sr._running is False

    @pytest.mark.asyncio
    async def test_frames_yields_new_frames(self) -> None:
        sr = StreamReader("http://host/stream")
        sr._running = True
        jpeg = _make_jpeg(16, 16)
        import cv2 as _cv2
        import numpy as _np
        arr = _np.frombuffer(jpeg, dtype=_np.uint8)
        frame = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
        sr._latest_frame = frame
        collected = []
        async for f in sr.frames():
            collected.append(f)
            sr._running = False
        assert len(collected) == 1

    @pytest.mark.asyncio
    async def test_frames_sleeps_when_no_new_frame(self, mocker: "pytest_mock.MockerFixture") -> None:  # noqa: F821
        sr = StreamReader("http://host/stream")
        sr._running = True
        call_count = 0

        async def fake_sleep(_: float) -> None:
            nonlocal call_count
            call_count += 1
            sr._running = False

        mocker.patch("asyncio.sleep", side_effect=fake_sleep)
        collected = []
        async for f in sr.frames():
            collected.append(f)  # pragma: no cover
        assert call_count >= 1
        assert collected == []


class TestStreamReaderCaptureLoop:
    @pytest.mark.asyncio
    async def test_stop_after_max_retries(self, mocker: "pytest_mock.MockerFixture") -> None:  # noqa: F821
        sr = StreamReader("http://host/stream", max_retries=2)
        sr._running = True
        call_count = 0

        async def fake_read() -> None:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("refused")

        mocker.patch.object(sr, "_read_stream", side_effect=fake_read)
        mocker.patch("asyncio.sleep", return_value=None)
        await sr._capture_loop()
        assert call_count == 2
        # loop exits naturally after max_retries; _running is not forced to False
        assert sr._running is True

    @pytest.mark.asyncio
    async def test_retries_reset_on_success(self, mocker: "pytest_mock.MockerFixture") -> None:  # noqa: F821
        sr = StreamReader("http://host/stream", max_retries=3)
        sr._running = True
        calls: list[int] = []

        async def fake_read() -> None:
            calls.append(1)
            if len(calls) >= 2:
                sr._running = False

        mocker.patch.object(sr, "_read_stream", side_effect=fake_read)
        mocker.patch("asyncio.sleep", return_value=None)
        await sr._capture_loop()
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_capture_loop_stops_immediately_when_not_running(self) -> None:
        sr = StreamReader("http://host/stream", max_retries=5)
        sr._running = False
        # Should exit without calling _read_stream at all
        await sr._capture_loop()

    @pytest.mark.asyncio
    async def test_start_creates_background_task(self, mocker: "pytest_mock.MockerFixture") -> None:  # noqa: F821
        sr = StreamReader("http://host/stream")
        mock_task = mocker.patch("asyncio.create_task")
        await sr.start()
        assert sr._running is True
        mock_task.assert_called_once()


class TestStreamReaderReadStream:
    @pytest.mark.asyncio
    async def test_read_stream_raises_on_http_error(self, mocker: "pytest_mock.MockerFixture") -> None:  # noqa: F821
        import httpx
        sr = StreamReader("http://host/stream")

        mock_response = mocker.MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=mocker.MagicMock(), response=mocker.MagicMock()
        )
        mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = mocker.AsyncMock(return_value=False)

        mock_client = mocker.MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)
        with pytest.raises(httpx.HTTPStatusError):
            await sr._read_stream()

    @pytest.mark.asyncio
    async def test_read_stream_updates_latest_frame(self, mocker: "pytest_mock.MockerFixture") -> None:  # noqa: F821
        sr = StreamReader("http://host/stream")
        jpeg = _make_jpeg(16, 16)

        # Build a mock async iterable for aiter_bytes
        async def aiter_bytes(chunk_size: int = 8192):  # noqa: ANN201
            yield jpeg

        mock_response = mocker.MagicMock()
        mock_response.raise_for_status = mocker.MagicMock()
        mock_response.aiter_bytes = aiter_bytes
        mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = mocker.AsyncMock(return_value=False)

        mock_client = mocker.MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)
        await sr._read_stream()
        frame = await sr.get_frame()
        assert frame is not None
        assert frame.shape == (16, 16, 3)
