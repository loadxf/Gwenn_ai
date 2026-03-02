"""Tests for gwenn.media.tts — ElevenLabs TTS synthesizer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCleanTextForTTS:
    """Tests for the text cleaning function."""

    def test_strips_fenced_code_blocks(self):
        from gwenn.media.tts import clean_text_for_tts

        text = "Hello\n```python\nprint('hi')\n```\nWorld"
        result = clean_text_for_tts(text)
        assert "```" not in result
        assert "print" not in result
        assert "Hello" in result
        assert "World" in result

    def test_strips_inline_code(self):
        from gwenn.media.tts import clean_text_for_tts

        result = clean_text_for_tts("Use `foo()` to run it.")
        assert "`" not in result
        assert "foo()" not in result

    def test_strips_html_tags(self):
        from gwenn.media.tts import clean_text_for_tts

        result = clean_text_for_tts("Hello <b>world</b>!")
        assert "<b>" not in result
        assert "</b>" not in result

    def test_preserves_link_text(self):
        from gwenn.media.tts import clean_text_for_tts

        result = clean_text_for_tts("See [the docs](https://example.com) for details.")
        assert "the docs" in result
        assert "https://example.com" not in result

    def test_strips_heading_markers(self):
        from gwenn.media.tts import clean_text_for_tts

        result = clean_text_for_tts("## My Heading\nSome text")
        assert "##" not in result
        assert "My Heading" in result

    def test_preserves_bold_italic_text(self):
        from gwenn.media.tts import clean_text_for_tts

        result = clean_text_for_tts("This is **bold** and *italic*.")
        assert "bold" in result
        assert "italic" in result
        assert "**" not in result
        assert result.count("*") == 0

    def test_strips_images(self):
        from gwenn.media.tts import clean_text_for_tts

        result = clean_text_for_tts("Look: ![alt](image.png) here")
        assert "![" not in result
        assert "image.png" not in result

    def test_empty_input(self):
        from gwenn.media.tts import clean_text_for_tts

        assert clean_text_for_tts("") == ""

    def test_only_markdown_returns_empty(self):
        from gwenn.media.tts import clean_text_for_tts

        result = clean_text_for_tts("```\ncode\n```")
        assert result == ""


class TestTTSSynthesizer:
    """Tests for gwenn.media.tts.TTSSynthesizer."""

    def _make_config(
        self,
        *,
        api_key="sk_test",
        default_voice_id="voice123",
        model_id="eleven_flash_v2_5",
        output_format="mp3_44100_128",
        output_mode="text_and_voice",
        max_text_length=5000,
        tts_proactive=False,
    ):
        """Create a minimal ElevenLabsConfig-like mock."""
        config = MagicMock()
        config.api_key = api_key
        config.is_available = bool(api_key)
        config.default_voice_id = default_voice_id
        config.model_id = model_id
        config.output_format = output_format
        config.output_mode = output_mode
        config.max_text_length = max_text_length
        config.tts_proactive = tts_proactive
        return config

    @pytest.mark.asyncio
    async def test_not_available_without_api_key(self):
        from gwenn.media.tts import TTSSynthesizer

        config = self._make_config(api_key=None)
        config.is_available = False
        synth = TTSSynthesizer(config)
        result = await synth.synthesize("Hello world")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_text_after_cleaning_returns_none(self):
        from gwenn.media.tts import TTSSynthesizer

        config = self._make_config()
        synth = TTSSynthesizer(config)
        result = await synth.synthesize("```\nonly code\n```")
        assert result is None

    @pytest.mark.asyncio
    async def test_text_too_long_truncated(self):
        from gwenn.media.tts import TTSSynthesizer

        config = self._make_config(max_text_length=10)
        synth = TTSSynthesizer(config)

        # Mock the client
        mock_client = MagicMock()

        async def mock_convert(**kwargs):
            # Return the text it received so we can verify truncation
            text = kwargs["text"]
            yield text.encode()

        mock_client.text_to_speech.convert = mock_convert
        synth._client = mock_client

        result = await synth.synthesize("A" * 50)
        assert result is not None
        # The text sent to the API should be truncated to max_text_length
        assert len(result) == 10  # 10 chars encoded as bytes

    @pytest.mark.asyncio
    async def test_graceful_import_error(self):
        from gwenn.media.tts import TTSSynthesizer

        config = self._make_config()
        synth = TTSSynthesizer(config)

        with patch.dict("sys.modules", {"elevenlabs": None, "elevenlabs.client": None}):
            result = await synth.synthesize("Hello world")
        assert result is None

    @pytest.mark.asyncio
    async def test_graceful_api_error(self):
        from gwenn.media.tts import TTSSynthesizer

        config = self._make_config()
        synth = TTSSynthesizer(config)

        mock_client = MagicMock()

        async def mock_convert_error(**kwargs):
            raise Exception("API rate limit exceeded")
            yield  # noqa: unreachable — makes this an async generator

        mock_client.text_to_speech.convert = mock_convert_error
        synth._client = mock_client

        result = await synth.synthesize("Hello world")
        assert result is None

    @pytest.mark.asyncio
    async def test_synthesis_success_mocked(self):
        from gwenn.media.tts import TTSSynthesizer

        config = self._make_config()
        synth = TTSSynthesizer(config)

        mock_client = MagicMock()
        audio_data = b"\xff\xfb\x90\x00" * 100  # fake MP3 bytes

        async def mock_convert(**kwargs):
            yield audio_data

        mock_client.text_to_speech.convert = mock_convert
        synth._client = mock_client

        result = await synth.synthesize("Hello, how are you today?")
        assert result == audio_data

    @pytest.mark.asyncio
    async def test_synthesis_uses_correct_params(self):
        from gwenn.media.tts import TTSSynthesizer

        config = self._make_config()
        synth = TTSSynthesizer(config)

        mock_client = MagicMock()
        call_kwargs = {}

        async def mock_convert(**kwargs):
            call_kwargs.update(kwargs)
            yield b"audio"

        mock_client.text_to_speech.convert = mock_convert
        synth._client = mock_client

        await synth.synthesize("Test text")
        assert call_kwargs["voice_id"] == "voice123"
        assert call_kwargs["model_id"] == "eleven_flash_v2_5"
        assert call_kwargs["output_format"] == "mp3_44100_128"

    @pytest.mark.asyncio
    async def test_explicit_voice_id_overrides_default(self):
        from gwenn.media.tts import TTSSynthesizer

        config = self._make_config()
        synth = TTSSynthesizer(config)

        mock_client = MagicMock()
        call_kwargs = {}

        async def mock_convert(**kwargs):
            call_kwargs.update(kwargs)
            yield b"audio"

        mock_client.text_to_speech.convert = mock_convert
        synth._client = mock_client

        await synth.synthesize("Test", voice_id="custom_voice")
        assert call_kwargs["voice_id"] == "custom_voice"

    @pytest.mark.asyncio
    async def test_session_voice_override(self):
        from gwenn.media.tts import TTSSynthesizer

        config = self._make_config()
        synth = TTSSynthesizer(config)

        # Set a session voice
        synth.set_session_voice("session_123", "session_voice_id")
        assert synth.get_session_voice("session_123") == "session_voice_id"

        mock_client = MagicMock()
        call_kwargs = {}

        async def mock_convert(**kwargs):
            call_kwargs.update(kwargs)
            yield b"audio"

        mock_client.text_to_speech.convert = mock_convert
        synth._client = mock_client

        await synth.synthesize("Test", session_id="session_123")
        assert call_kwargs["voice_id"] == "session_voice_id"

    @pytest.mark.asyncio
    async def test_session_voice_cleared(self):
        from gwenn.media.tts import TTSSynthesizer

        config = self._make_config()
        synth = TTSSynthesizer(config)

        synth.set_session_voice("s1", "voice_x")
        synth.clear_session_voice("s1")
        assert synth.get_session_voice("s1") is None

    @pytest.mark.asyncio
    async def test_list_voices_mocked(self):
        from gwenn.media.tts import TTSSynthesizer

        config = self._make_config()
        synth = TTSSynthesizer(config)

        mock_client = MagicMock()
        voice1 = MagicMock()
        voice1.voice_id = "v1"
        voice1.name = "Rachel"
        voice2 = MagicMock()
        voice2.voice_id = "v2"
        voice2.name = "Josh"
        mock_response = MagicMock()
        mock_response.voices = [voice1, voice2]
        mock_client.voices.get_all = AsyncMock(return_value=mock_response)
        synth._client = mock_client

        voices = await synth.list_voices()
        assert voices == [
            {"voice_id": "v1", "name": "Rachel"},
            {"voice_id": "v2", "name": "Josh"},
        ]

    @pytest.mark.asyncio
    async def test_list_voices_not_available(self):
        from gwenn.media.tts import TTSSynthesizer

        config = self._make_config(api_key=None)
        config.is_available = False
        synth = TTSSynthesizer(config)
        result = await synth.list_voices()
        assert result is None

    @pytest.mark.asyncio
    async def test_list_voices_api_error(self):
        from gwenn.media.tts import TTSSynthesizer

        config = self._make_config()
        synth = TTSSynthesizer(config)

        mock_client = MagicMock()
        mock_client.voices.get_all = AsyncMock(side_effect=Exception("network error"))
        synth._client = mock_client

        result = await synth.list_voices()
        assert result is None

    @pytest.mark.asyncio
    async def test_close(self):
        from gwenn.media.tts import TTSSynthesizer

        config = self._make_config()
        synth = TTSSynthesizer(config)

        mock_client = AsyncMock()
        synth._client = mock_client
        synth.set_session_voice("s1", "v1")

        await synth.close()
        mock_client.close.assert_called_once()
        assert synth._client is None
        assert synth._session_voices == {}

    @pytest.mark.asyncio
    async def test_close_no_client(self):
        from gwenn.media.tts import TTSSynthesizer

        config = self._make_config()
        synth = TTSSynthesizer(config)
        await synth.close()  # Should be a no-op.

    def test_output_mode_property(self):
        from gwenn.media.tts import TTSSynthesizer

        config = self._make_config(output_mode="voice_only")
        synth = TTSSynthesizer(config)
        assert synth.output_mode == "voice_only"

    @pytest.mark.asyncio
    async def test_empty_audio_response_returns_none(self):
        from gwenn.media.tts import TTSSynthesizer

        config = self._make_config()
        synth = TTSSynthesizer(config)

        mock_client = MagicMock()

        async def mock_convert(**kwargs):
            # Yield nothing → empty bytes
            return
            yield  # noqa: unreachable — makes this an async generator

        mock_client.text_to_speech.convert = mock_convert
        synth._client = mock_client

        result = await synth.synthesize("Hello")
        assert result is None


class TestElevenLabsConfig:
    """Tests for the ElevenLabsConfig in gwenn.config."""

    def test_defaults(self):
        from gwenn.config import ElevenLabsConfig

        config = ElevenLabsConfig()
        assert config.api_key is None
        assert config.is_available is False
        assert config.default_voice_id == "21m00Tcm4TlvDq8ikWAM"
        assert config.model_id == "eleven_flash_v2_5"
        assert config.output_format == "mp3_44100_128"
        assert config.output_mode == "text_only"
        assert config.max_text_length == 5000
        assert config.tts_proactive is False

    def test_is_available_with_key(self):
        from gwenn.config import ElevenLabsConfig

        config = ElevenLabsConfig(ELEVENLABS_API_KEY="sk_test_123")
        assert config.is_available is True

    def test_invalid_output_mode_normalized(self):
        from gwenn.config import ElevenLabsConfig

        config = ElevenLabsConfig(ELEVENLABS_OUTPUT_MODE="invalid_mode")
        assert config.output_mode == "text_only"

    def test_valid_output_modes(self):
        from gwenn.config import ElevenLabsConfig

        for mode in ("text_only", "voice_only", "text_and_voice"):
            config = ElevenLabsConfig(ELEVENLABS_OUTPUT_MODE=mode)
            assert config.output_mode == mode


class TestSetVoiceToolHandler:
    """Tests for the set_voice tool handler wiring."""

    @pytest.mark.asyncio
    async def test_set_voice_handler_stores_session_voice(self):
        from gwenn.media.tts import TTSSynthesizer

        config = MagicMock()
        config.api_key = "sk_test"
        config.is_available = True
        config.default_voice_id = "default"
        config.output_mode = "text_and_voice"

        synth = TTSSynthesizer(config)
        synth.set_session_voice("session_abc", "new_voice_id")

        assert synth.get_session_voice("session_abc") == "new_voice_id"

    @pytest.mark.asyncio
    async def test_session_voice_used_in_synthesis(self):
        from gwenn.media.tts import TTSSynthesizer

        config = MagicMock()
        config.api_key = "sk_test"
        config.is_available = True
        config.default_voice_id = "default_voice"
        config.model_id = "eleven_flash_v2_5"
        config.output_format = "mp3_44100_128"
        config.max_text_length = 5000

        synth = TTSSynthesizer(config)
        synth.set_session_voice("s1", "override_voice")

        mock_client = MagicMock()
        call_kwargs = {}

        async def mock_convert(**kwargs):
            call_kwargs.update(kwargs)
            yield b"audio"

        mock_client.text_to_speech.convert = mock_convert
        synth._client = mock_client

        await synth.synthesize("Hello", session_id="s1")
        assert call_kwargs["voice_id"] == "override_voice"

        # Without session override, uses default
        call_kwargs.clear()
        await synth.synthesize("Hello", session_id="other_session")
        assert call_kwargs["voice_id"] == "default_voice"
