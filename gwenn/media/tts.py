"""
Text-to-speech synthesis via ElevenLabs API.

Mirrors the AudioTranscriber pattern: lazy client init, graceful ``None``
fallback on missing dependencies or API errors, structured logging.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from gwenn.config import ElevenLabsConfig

logger = structlog.get_logger(__name__)

# Regex to strip markdown/HTML before sending to TTS
_STRIP_PATTERNS = [
    re.compile(r"```[\s\S]*?```"),       # fenced code blocks
    re.compile(r"`[^`]+`"),              # inline code
    re.compile(r"<[^>]+>"),              # HTML tags
    re.compile(r"!\[[^\]]*\]\([^)]+\)"), # images
    re.compile(r"\[([^\]]+)\]\([^)]+\)"),# links → keep text
    re.compile(r"^#{1,6}\s+", re.MULTILINE),  # heading markers
    re.compile(r"\*{1,3}([^*]+)\*{1,3}"),     # bold/italic → keep text
    re.compile(r"(?<!\w)_{1,3}([^_]+)_{1,3}(?!\w)"),  # emphasis → keep text (avoid _var_names_)
    re.compile(r"^[\s]*[-*+]\s", re.MULTILINE),  # list markers
    re.compile(r"^>\s?", re.MULTILINE),          # blockquotes
    re.compile(r"\n{3,}"),                       # excessive newlines
]


def clean_text_for_tts(text: str) -> str:
    """Strip markdown/HTML formatting for natural-sounding TTS output."""
    result = text
    for pattern in _STRIP_PATTERNS:
        if pattern.groups:
            result = pattern.sub(r"\1", result)
        else:
            result = pattern.sub("", result)
    return result.strip()


class TTSSynthesizer:
    """Synthesize speech from text via ElevenLabs."""

    def __init__(self, config: ElevenLabsConfig) -> None:
        self._config = config
        self._client = None  # Lazy-init AsyncElevenLabs
        self._session_voices: dict[str, str] = {}  # session_id → voice_id

    @property
    def output_mode(self) -> str:
        return self._config.output_mode

    async def synthesize(
        self,
        text: str,
        *,
        voice_id: str | None = None,
        session_id: str | None = None,
    ) -> bytes | None:
        """Synthesize *text* to audio bytes, or return ``None`` on failure.

        Returns ``None`` (not raises) when:
        - API key is not configured
        - ``elevenlabs`` package is not installed
        - Text is empty after cleaning
        - Any API or network error occurs
        """
        if not self._config.is_available:
            return None

        cleaned = clean_text_for_tts(text)
        if not cleaned:
            logger.debug("tts.empty_after_cleaning")
            return None

        # Truncate to max length (don't reject — user still gets audio)
        if len(cleaned) > self._config.max_text_length:
            logger.debug(
                "tts.text_truncated",
                original_length=len(cleaned),
                max_length=self._config.max_text_length,
            )
            cleaned = cleaned[: self._config.max_text_length]

        # Resolve voice: explicit > session override > config default
        resolved_voice = voice_id
        if resolved_voice is None and session_id is not None:
            resolved_voice = self._session_voices.get(session_id)
        if resolved_voice is None:
            resolved_voice = self._config.default_voice_id

        client = self._ensure_client()
        if client is None:
            return None

        try:
            audio_iter = client.text_to_speech.convert(
                text=cleaned,
                voice_id=resolved_voice,
                model_id=self._config.model_id,
                output_format=self._config.output_format,
            )
            # Buffer all chunks into a single bytes object
            chunks = []
            async for chunk in audio_iter:
                chunks.append(chunk)
            audio_bytes = b"".join(chunks)

            if not audio_bytes:
                logger.warning("tts.empty_response")
                return None

            logger.debug(
                "tts.synthesis_success",
                text_length=len(cleaned),
                audio_bytes=len(audio_bytes),
                voice_id=resolved_voice,
            )
            return audio_bytes
        except Exception as exc:
            logger.warning("tts.synthesis_failed", error=str(exc))
            return None

    async def list_voices(self) -> list[dict] | None:
        """Return available voices as a list of dicts, or ``None`` on failure."""
        if not self._config.is_available:
            return None

        client = self._ensure_client()
        if client is None:
            return None

        try:
            response = await client.voices.get_all()
            return [
                {"voice_id": v.voice_id, "name": v.name}
                for v in response.voices
            ]
        except Exception as exc:
            logger.warning("tts.list_voices_failed", error=str(exc))
            return None

    def set_session_voice(self, session_id: str, voice_id: str) -> None:
        """Set a per-session voice override."""
        self._session_voices[session_id] = voice_id

    def get_session_voice(self, session_id: str) -> str | None:
        """Get the per-session voice override, or ``None``."""
        return self._session_voices.get(session_id)

    def clear_session_voice(self, session_id: str) -> None:
        """Remove a per-session voice override."""
        self._session_voices.pop(session_id, None)

    def _ensure_client(self):
        """Lazily initialize the AsyncElevenLabs client."""
        if self._client is not None:
            return self._client
        try:
            from elevenlabs.client import AsyncElevenLabs

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
        self._session_voices.clear()
