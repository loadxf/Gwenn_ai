"""
Audio transcription via Groq's Whisper API.

Provides non-blocking transcription of voice messages and video audio tracks.
The Groq client is lazily initialized so the module works even when the
``groq`` package is not installed (graceful ``None`` fallback).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from gwenn.config import GroqConfig

logger = structlog.get_logger(__name__)


class AudioTranscriber:
    """Transcribe audio/video bytes via Groq's Whisper endpoint."""

    def __init__(self, config: GroqConfig) -> None:
        self._config = config
        self._client = None  # Lazy-init AsyncGroq

    async def transcribe(
        self,
        audio_bytes: bytes | bytearray,
        filename: str = "audio.ogg",
    ) -> str | None:
        """Transcribe *audio_bytes* and return the text, or ``None`` on failure.

        Returns ``None`` (not raises) when:
        - Groq API key is not configured
        - ``groq`` package is not installed
        - File exceeds the configured size limit
        - Any API or network error occurs
        """
        if not self._config.is_available:
            return None

        if len(audio_bytes) > self._config.max_audio_bytes:
            logger.warning(
                "audio.file_too_large",
                size=len(audio_bytes),
                limit=self._config.max_audio_bytes,
            )
            return None

        client = self._ensure_client()
        if client is None:
            return None

        try:
            transcription = await client.audio.transcriptions.create(
                file=(filename, bytes(audio_bytes)),
                model=self._config.whisper_model,
            )
            text = transcription.text.strip() if transcription.text else None
            if text:
                logger.debug(
                    "audio.transcription_success",
                    length=len(text),
                    filename=filename,
                )
            return text or None
        except Exception as exc:
            logger.warning("audio.transcription_failed", error=str(exc))
            return None

    def _ensure_client(self):
        """Lazily initialize the AsyncGroq client."""
        if self._client is not None:
            return self._client
        try:
            from groq import AsyncGroq

            self._client = AsyncGroq(api_key=self._config.api_key)
            return self._client
        except ImportError:
            logger.warning("audio.groq_not_installed")
            return None
        except Exception as exc:
            logger.warning("audio.client_init_failed", error=str(exc))
            return None

    async def close(self) -> None:
        """Close the underlying HTTP client if initialized."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None
