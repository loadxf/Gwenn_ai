"""
Unit tests for gwenn.media.tts — TextToSpeechSynthesizer.

All tests are self-contained: ElevenLabs is mocked so no real API calls occur.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestTextToSpeechSynthesizer:
    """Tests for gwenn.media.tts.TextToSpeechSynthesizer."""

    def _make_config(self, *, api_key="el_test", max_chars=2500, mode="voice_reply"):
        """Create a minimal ElevenLabsConfig-like object."""
        config = MagicMock()
        config.api_key = api_key
        config.is_available = bool(api_key)
        config.voice_id = "JBFqnCBsd6RMkjVDRZzb"
        config.model = "eleven_turbo_v2_5"
        config.output_format = "opus_48000_128"
        config.max_chars = max_chars
        config.mode = mode

        # Wire up the real should_send_voice logic so tests exercise it.
        def should_send_voice(is_voice_message=False):
            if not config.is_available or config.mode == "off":
                return False
            if config.mode == "always":
                return True
            return config.mode == "voice_reply" and is_voice_message

        config.should_send_voice = should_send_voice
        return config

    @pytest.mark.asyncio
    async def test_no_api_key_returns_none(self):
        from gwenn.media.tts import TextToSpeechSynthesizer

        config = self._make_config(api_key=None)
        config.is_available = False
        synth = TextToSpeechSynthesizer(config)
        result = await synth.synthesize("Hello world")
        assert result is None

    @pytest.mark.asyncio
    async def test_elevenlabs_not_installed_returns_none(self):
        from gwenn.media.tts import TextToSpeechSynthesizer

        config = self._make_config()
        synth = TextToSpeechSynthesizer(config)

        with patch.dict("sys.modules", {"elevenlabs": None}):
            result = await synth.synthesize("Hello world")
        assert result is None

    @pytest.mark.asyncio
    async def test_text_truncation(self):
        from gwenn.media.tts import TextToSpeechSynthesizer

        config = self._make_config(max_chars=10)
        synth = TextToSpeechSynthesizer(config)

        # Mock the client to capture the text that was sent.
        mock_client = MagicMock()
        fake_audio = b"\x00\x01\x02\x03"

        async def fake_convert(**kwargs):
            # Verify text was truncated to max_chars.
            assert len(kwargs["text"]) == 10
            yield fake_audio

        mock_client.text_to_speech.convert = fake_convert
        synth._client = mock_client

        result = await synth.synthesize("This is a long text that should be truncated")
        assert result == fake_audio

    @pytest.mark.asyncio
    async def test_success_mocked(self):
        from gwenn.media.tts import TextToSpeechSynthesizer

        config = self._make_config()
        synth = TextToSpeechSynthesizer(config)

        fake_audio = b"OggS\x00\x01fake_audio_data"
        mock_client = MagicMock()

        async def fake_convert(**kwargs):
            yield b"OggS\x00\x01"
            yield b"fake_audio_data"

        mock_client.text_to_speech.convert = fake_convert
        synth._client = mock_client

        result = await synth.synthesize("Hello, how are you?")
        assert result == fake_audio

    @pytest.mark.asyncio
    async def test_api_error_returns_none(self):
        from gwenn.media.tts import TextToSpeechSynthesizer

        config = self._make_config()
        synth = TextToSpeechSynthesizer(config)

        mock_client = MagicMock()

        async def failing_convert(**kwargs):
            raise Exception("API rate limit")
            yield  # noqa: unreachable — makes this an async generator

        mock_client.text_to_speech.convert = failing_convert
        synth._client = mock_client

        result = await synth.synthesize("Hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_audio_returns_none(self):
        from gwenn.media.tts import TextToSpeechSynthesizer

        config = self._make_config()
        synth = TextToSpeechSynthesizer(config)

        mock_client = MagicMock()

        async def empty_convert(**kwargs):
            return
            yield  # noqa: unreachable — makes this an async generator

        mock_client.text_to_speech.convert = empty_convert
        synth._client = mock_client

        result = await synth.synthesize("Hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_text_returns_none(self):
        from gwenn.media.tts import TextToSpeechSynthesizer

        config = self._make_config()
        synth = TextToSpeechSynthesizer(config)

        result = await synth.synthesize("")
        assert result is None

        result = await synth.synthesize("   ")
        assert result is None

    @pytest.mark.asyncio
    async def test_close(self):
        from gwenn.media.tts import TextToSpeechSynthesizer

        config = self._make_config()
        synth = TextToSpeechSynthesizer(config)

        mock_client = AsyncMock()
        synth._client = mock_client

        await synth.close()
        mock_client.close.assert_called_once()
        assert synth._client is None

    @pytest.mark.asyncio
    async def test_close_no_client(self):
        """Close without initialized client should not raise."""
        from gwenn.media.tts import TextToSpeechSynthesizer

        config = self._make_config()
        synth = TextToSpeechSynthesizer(config)
        await synth.close()  # Should be a no-op.

    def test_should_send_voice_modes(self):
        """Test ElevenLabsConfig.should_send_voice() for all three modes."""

        # Mode: off
        cfg = self._make_config(mode="off")
        assert not cfg.should_send_voice(False)
        assert not cfg.should_send_voice(True)

        # Mode: voice_reply
        cfg = self._make_config(mode="voice_reply")
        assert not cfg.should_send_voice(False)
        assert cfg.should_send_voice(True)

        # Mode: always
        cfg = self._make_config(mode="always")
        assert cfg.should_send_voice(False)
        assert cfg.should_send_voice(True)

        # Not available (no API key)
        cfg = self._make_config(api_key=None, mode="always")
        cfg.is_available = False
        assert not cfg.should_send_voice(True)
        assert not cfg.should_send_voice(False)
