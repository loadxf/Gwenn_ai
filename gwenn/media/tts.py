"""
Text-to-speech synthesis via ElevenLabs.

Provides non-blocking TTS so Gwenn can reply with voice messages.
The ElevenLabs client is lazily initialized so the module works even when the
``elevenlabs`` package is not installed (graceful ``None`` fallback).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from gwenn.config import ElevenLabsConfig

logger = structlog.get_logger(__name__)


class TextToSpeechSynthesizer:
    """Synthesize speech audio bytes via ElevenLabs."""

    def __init__(self, config: ElevenLabsConfig) -> None:
        self._config = config
        self._client = None  # Lazy-init AsyncElevenLabs

    async def synthesize(self, text: str) -> bytes | None:
        """Synthesize *text* and return OGG/Opus audio bytes, or ``None`` on failure.

        Returns ``None`` (not raises) when:
        - ElevenLabs API key is not configured
        - ``elevenlabs`` package is not installed
        - Any API or network error occurs
        """
        if not self._config.is_available:
            return None

        if not text or not text.strip():
            return None

        # Truncate to max_chars to stay within reasonable speech length.
        if len(text) > self._config.max_chars:
            text = text[: self._config.max_chars]

        client = self._ensure_client()
        if client is None:
            return None

        try:
            audio_iterator = client.text_to_speech.convert(
                text=text,
                voice_id=self._config.voice_id,
                model_id=self._config.model,
                output_format=self._config.output_format,
            )
            # Collect the streamed audio chunks into a single bytes object.
            chunks = []
            async for chunk in audio_iterator:
                chunks.append(chunk)
            audio_bytes = b"".join(chunks)
            if not audio_bytes:
                logger.warning("tts.empty_audio_response")
                return None
            logger.debug("tts.synthesis_success", audio_size=len(audio_bytes))
            return audio_bytes
        except Exception as exc:
            logger.warning("tts.synthesis_failed", error=str(exc))
            return None

    def _ensure_client(self):
        """Lazily initialize the AsyncElevenLabs client."""
        if self._client is not None:
            return self._client
        try:
            from elevenlabs import AsyncElevenLabs

            self._client = AsyncElevenLabs(api_key=self._config.api_key)
            return self._client
        except ImportError:
            logger.warning("tts.elevenlabs_not_installed")
            return None
        except Exception as exc:
            logger.warning("tts.client_init_failed", error=str(exc))
            return None

    async def close(self) -> None:
        """Close the underlying HTTP client if initialized."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None
