"""
Unit tests for the gwenn.media package — VideoProcessor and AudioTranscriber.

All tests are self-contained: OpenCV and Groq are mocked so no real video
decoding or API calls occur.
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# VideoProcessor tests
# ---------------------------------------------------------------------------


class TestVideoProcessor:
    """Tests for gwenn.media.video.VideoProcessor."""

    def test_build_thumbnail_block_returns_image_block(self):
        from gwenn.media.video import VideoProcessor

        fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        blocks = VideoProcessor.build_thumbnail_block(fake_jpeg)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "image"
        assert blocks[0]["source"]["type"] == "base64"
        assert blocks[0]["source"]["media_type"] == "image/jpeg"
        # Verify round-trip.
        decoded = base64.standard_b64decode(blocks[0]["source"]["data"])
        assert decoded == fake_jpeg

    def test_build_thumbnail_block_empty_bytes(self):
        from gwenn.media.video import VideoProcessor

        blocks = VideoProcessor.build_thumbnail_block(b"")
        # Empty bytes still produce a valid block (base64 of empty is valid).
        assert len(blocks) == 1

    @pytest.mark.asyncio
    async def test_extract_frames_invalid_data_returns_empty(self):
        """Corrupt/invalid video data should gracefully return []."""
        from gwenn.media.video import VideoProcessor

        blocks = await VideoProcessor.extract_frames(b"not a real video")
        assert blocks == []

    @pytest.mark.asyncio
    async def test_extract_frames_opencv_not_installed_returns_empty(self):
        """When cv2 is not importable, extract_frames returns []."""
        from gwenn.media.video import VideoProcessor

        with patch.dict("sys.modules", {"cv2": None, "numpy": None}):
            # Force re-import failure inside the sync extraction.
            import importlib
            from gwenn.media import video as video_mod

            blocks = await VideoProcessor.extract_frames(b"fake video data")
            assert blocks == []

    @pytest.mark.asyncio
    async def test_extract_frames_success_mocked(self):
        """Verify frame extraction with mocked OpenCV."""
        import sys

        import numpy as np

        from gwenn.media.video import VideoProcessor

        # Create a mock VideoCapture that returns fake frames.
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100.0  # total frames
        # Create a tiny red image.
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_frame[:, :, 2] = 255  # Red channel
        mock_cap.read.return_value = (True, fake_frame)
        mock_cap.set.return_value = True

        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_COUNT = 7  # cv2 constant
        mock_cv2.CAP_PROP_POS_FRAMES = 1
        mock_cv2.IMWRITE_JPEG_QUALITY = 1
        mock_cv2.INTER_AREA = 3

        # imencode returns (True, buffer)
        encoded = b"\xff\xd8fake_jpeg_data"
        buf = MagicMock()
        buf.tobytes.return_value = encoded
        mock_cv2.imencode.return_value = (True, buf)

        # resize returns the frame as-is (under max edge)
        mock_cv2.resize = MagicMock(return_value=fake_frame)

        # Patch cv2 at the sys.modules level since it's imported lazily.
        original_cv2 = sys.modules.get("cv2")
        sys.modules["cv2"] = mock_cv2
        try:
            blocks = await VideoProcessor.extract_frames(b"fake mp4 data", num_frames=4)
        finally:
            if original_cv2 is not None:
                sys.modules["cv2"] = original_cv2
            else:
                sys.modules.pop("cv2", None)

        assert len(blocks) == 4
        for block in blocks:
            assert block["type"] == "image"
            assert block["source"]["media_type"] == "image/jpeg"
            decoded = base64.standard_b64decode(block["source"]["data"])
            assert decoded == encoded


# ---------------------------------------------------------------------------
# AudioTranscriber tests
# ---------------------------------------------------------------------------


class TestAudioTranscriber:
    """Tests for gwenn.media.audio.AudioTranscriber."""

    def _make_config(self, *, api_key="gsk_test", max_bytes=25 * 1024 * 1024):
        """Create a minimal GroqConfig-like object."""
        config = MagicMock()
        config.api_key = api_key
        config.is_available = bool(api_key)
        config.whisper_model = "whisper-large-v3-turbo"
        config.max_audio_bytes = max_bytes
        return config

    @pytest.mark.asyncio
    async def test_no_api_key_returns_none(self):
        from gwenn.media.audio import AudioTranscriber

        config = self._make_config(api_key=None)
        config.is_available = False
        transcriber = AudioTranscriber(config)
        result = await transcriber.transcribe(b"audio data")
        assert result is None

    @pytest.mark.asyncio
    async def test_file_too_large_returns_none(self):
        from gwenn.media.audio import AudioTranscriber

        config = self._make_config(max_bytes=100)
        transcriber = AudioTranscriber(config)
        result = await transcriber.transcribe(b"x" * 200, "large.ogg")
        assert result is None

    @pytest.mark.asyncio
    async def test_groq_not_installed_returns_none(self):
        from gwenn.media.audio import AudioTranscriber

        config = self._make_config()
        transcriber = AudioTranscriber(config)

        with patch.dict("sys.modules", {"groq": None}):
            result = await transcriber.transcribe(b"audio data")
        assert result is None

    @pytest.mark.asyncio
    async def test_success_mocked(self):
        from gwenn.media.audio import AudioTranscriber

        config = self._make_config()
        transcriber = AudioTranscriber(config)

        # Mock the AsyncGroq client and its transcription method.
        mock_client = MagicMock()
        mock_transcription = MagicMock()
        mock_transcription.text = "Hello, how are you?"
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)

        transcriber._client = mock_client

        result = await transcriber.transcribe(b"fake ogg data", "voice.ogg")
        assert result == "Hello, how are you?"
        mock_client.audio.transcriptions.create.assert_called_once_with(
            file=("voice.ogg", b"fake ogg data"),
            model="whisper-large-v3-turbo",
        )

    @pytest.mark.asyncio
    async def test_api_error_returns_none(self):
        from gwenn.media.audio import AudioTranscriber

        config = self._make_config()
        transcriber = AudioTranscriber(config)

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            side_effect=Exception("API rate limit")
        )
        transcriber._client = mock_client

        result = await transcriber.transcribe(b"audio data", "test.ogg")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_transcript_returns_none(self):
        from gwenn.media.audio import AudioTranscriber

        config = self._make_config()
        transcriber = AudioTranscriber(config)

        mock_client = MagicMock()
        mock_transcription = MagicMock()
        mock_transcription.text = "   "
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)
        transcriber._client = mock_client

        result = await transcriber.transcribe(b"silence", "empty.ogg")
        assert result is None

    @pytest.mark.asyncio
    async def test_close(self):
        from gwenn.media.audio import AudioTranscriber

        config = self._make_config()
        transcriber = AudioTranscriber(config)

        mock_client = AsyncMock()
        transcriber._client = mock_client

        await transcriber.close()
        mock_client.close.assert_called_once()
        assert transcriber._client is None

    @pytest.mark.asyncio
    async def test_close_no_client(self):
        """Close without initialized client should not raise."""
        from gwenn.media.audio import AudioTranscriber

        config = self._make_config()
        transcriber = AudioTranscriber(config)
        await transcriber.close()  # Should be a no-op.
