"""
Video frame extraction for multimodal Claude conversations.

Extracts representative frames from video bytes using OpenCV, encodes them
as base64 JPEG, and returns Claude API image content blocks.  OpenCV is
imported lazily so the module loads cleanly even when opencv-python-headless
is not installed.
"""

from __future__ import annotations

import asyncio
import base64
import tempfile
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Maximum long-edge dimension for extracted frames (Claude vision limit).
_MAX_LONG_EDGE: int = 1568
# JPEG encode quality (0-100).
_JPEG_QUALITY: int = 85


class VideoProcessor:
    """Stateless helpers for extracting frames from video bytes."""

    @staticmethod
    async def extract_frames(
        video_bytes: bytes | bytearray,
        *,
        num_frames: int = 4,
        suffix: str = ".mp4",
    ) -> list[dict]:
        """Extract *num_frames* evenly-spaced frames from *video_bytes*.

        Returns a list of Claude API image content blocks, or ``[]`` on any
        failure (missing OpenCV, corrupt video, etc.).
        """
        try:
            return await asyncio.to_thread(
                VideoProcessor._extract_sync, bytes(video_bytes), num_frames, suffix
            )
        except Exception as exc:
            logger.warning("video.extract_frames_failed", error=str(exc))
            return []

    @staticmethod
    def _extract_sync(
        video_bytes: bytes,
        num_frames: int,
        suffix: str,
    ) -> list[dict]:
        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.warning("video.opencv_not_installed")
            return []

        tmp_path: str | None = None
        try:
            # cv2.VideoCapture requires a file path.
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(video_bytes)
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                logger.warning("video.cannot_open")
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 1:
                cap.release()
                return []

            # Sample at 20%, 40%, 60%, 80% positions.
            positions = [
                int(total_frames * pct) for pct in _frame_positions(num_frames)
            ]

            blocks: list[dict] = []
            for pos in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(pos, total_frames - 1))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # Resize to fit Claude's vision limit.
                frame = _resize_frame(frame, _MAX_LONG_EDGE)

                # Encode as JPEG.
                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
                )
                if not ok:
                    continue

                b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                })

            cap.release()
            return blocks
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def build_thumbnail_block(thumbnail_bytes: bytes | bytearray) -> list[dict]:
        """Encode a pre-existing thumbnail image as a Claude image block.

        Useful as a fallback when the full video is too large to process.
        Returns a single-element list, or ``[]`` on failure.
        """
        try:
            b64 = base64.standard_b64encode(bytes(thumbnail_bytes)).decode("ascii")
            return [{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            }]
        except Exception as exc:
            logger.warning("video.thumbnail_encode_failed", error=str(exc))
            return []


def _frame_positions(n: int) -> list[float]:
    """Return *n* evenly-spaced fractional positions in (0, 1)."""
    return [(i + 1) / (n + 1) for i in range(n)]


def _resize_frame(frame, max_edge: int):
    """Resize *frame* so the longest edge is at most *max_edge* pixels."""
    import cv2

    h, w = frame.shape[:2]
    if max(h, w) <= max_edge:
        return frame
    scale = max_edge / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
