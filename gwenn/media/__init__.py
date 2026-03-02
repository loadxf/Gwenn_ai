"""
Shared media processing for Gwenn channels.

Frame extraction and audio transcription operate on raw bytes — identical
for Telegram and Discord — so they live here rather than being duplicated
in each channel adapter.
"""

from __future__ import annotations

from gwenn.media.audio import AudioTranscriber
from gwenn.media.tts import TTSSynthesizer
from gwenn.media.video import VideoProcessor

__all__ = ["AudioTranscriber", "TTSSynthesizer", "VideoProcessor"]
