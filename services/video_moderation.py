import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import traceback
import re
import yt_dlp
import requests

from transformers import (
    AutoModelForImageClassification,
    ViTImageProcessor,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModerationResult:
    """Data class for moderation results"""

    is_safe: bool
    confidence: float
    category: str
    timestamp: float
    frame_number: int
    details: Dict = None


class VideoModerationService:
    """
    Service for moderating video content using AI models
    Detects NSFW, violence, and other inappropriate content
    """

    def __init__(
        self,
        nsfw_model_name: str = "Falconsai/nsfw_image_detection",
        violence_model_name: str = "jaranohaal/vit-base-violence-detection",
        device: str = None,
        nsfw_threshold: float = 0.7,
        violence_threshold: float = 0.6,
        sample_rate: int = 30,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.nsfw_threshold = nsfw_threshold
        self.violence_threshold = violence_threshold
        self.sample_rate = sample_rate

        logger.info(f"Initializing VideoModerationService on {self.device}")

        try:
            # Load NSFW detection model
            logger.info(f"Loading NSFW model: {nsfw_model_name}")
            self.nsfw_processor = ViTImageProcessor.from_pretrained(nsfw_model_name)
            self.nsfw_model = AutoModelForImageClassification.from_pretrained(
                nsfw_model_name
            ).to(self.device)
            self.nsfw_model.eval()

            # Load violence detection model
            logger.info(f"Loading violence model: {violence_model_name}")
            self.violence_processor = ViTImageProcessor.from_pretrained(
                violence_model_name
            )
            self.violence_model = AutoModelForImageClassification.from_pretrained(
                violence_model_name
            ).to(self.device)
            self.violence_model.eval()

            logger.info("Models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _process_frame_nsfw(self, frame: np.ndarray) -> Tuple[bool, float, str]:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            inputs = self.nsfw_processor(images=image, return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                outputs = self.nsfw_model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)

            predicted_class_idx = logits.argmax(-1).item()
            confidence = probabilities[0][predicted_class_idx].item()
            category = self.nsfw_model.config.id2label[predicted_class_idx]

            is_nsfw = category.lower() == "nsfw" and confidence >= self.nsfw_threshold

            return is_nsfw, confidence, category

        except Exception as e:
            logger.error(f"Error processing frame for NSFW: {str(e)}")
            logger.error(traceback.format_exc())
            return False, 0.0, "error"

    def _process_frame_violence(self, frame: np.ndarray) -> Tuple[bool, float, str]:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            inputs = self.violence_processor(images=image, return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                outputs = self.violence_model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)

            predicted_class_idx = logits.argmax(-1).item()
            confidence = probabilities[0][predicted_class_idx].item()
            category = self.violence_model.config.id2label[predicted_class_idx]

            is_violent = (
                category.lower() == "violence"
                and confidence >= self.violence_threshold
            )

            return is_violent, confidence, category

        except Exception as e:
            logger.error(f"Error processing frame for violence: {str(e)}")
            logger.error(traceback.format_exc())
            return False, 0.0, "error"

    def moderate_frame(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> List[ModerationResult]:
        results = []

        # Check NSFW
        is_nsfw, nsfw_conf, nsfw_cat = self._process_frame_nsfw(frame)
        if is_nsfw:
            results.append(
                ModerationResult(
                    is_safe=False,
                    confidence=nsfw_conf,
                    category=f"NSFW ({nsfw_cat})",
                    timestamp=timestamp,
                    frame_number=frame_number,
                    details={"type": "nsfw", "label": nsfw_cat},
                )
            )

        # Check violence
        is_violent, violence_conf, violence_cat = self._process_frame_violence(frame)
        if is_violent:
            results.append(
                ModerationResult(
                    is_safe=False,
                    confidence=violence_conf,
                    category=f"Violence ({violence_cat})",
                    timestamp=timestamp,
                    frame_number=frame_number,
                    details={"type": "violence", "label": violence_cat},
                )
            )

        # If no violations found, return safe result
        if not results:
            results.append(
                ModerationResult(
                    is_safe=True,
                    confidence=1.0,
                    category="Safe",
                    timestamp=timestamp,
                    frame_number=frame_number,
                    details={"type": "safe"},
                )
            )

        return results

    def moderate_video(
        self, video_path: str, max_frames: Optional[int] = None
    ) -> Dict:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Starting video moderation: {video_path}")
        start_time = time.time()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(
            f"Video info: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s duration"
        )

        all_violations = []
        frame_count = 0
        processed_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.sample_rate == 0:
                    timestamp = frame_count / fps if fps > 0 else 0

                    frame_results = self.moderate_frame(frame, frame_count, timestamp)

                    for result in frame_results:
                        if not result.is_safe:
                            all_violations.append(result)

                    processed_count += 1

                    if processed_count % 10 == 0:
                        logger.info(f"Processed {processed_count} frames...")

                frame_count += 1

                if max_frames and processed_count >= max_frames:
                    break

        finally:
            cap.release()

        processing_time = time.time() - start_time

        # Compile results — include 'status' and 'flagged' supaya compatible dengan apply_policy() di app.py
        is_safe = len(all_violations) == 0
        max_confidence = 0.0
        if all_violations:
            max_confidence = max(v.confidence for v in all_violations)

        result = {
            "status": "success",
            "flagged": not is_safe,
            "scores": {
                "nsfw": max(
                    (v.confidence for v in all_violations if v.details.get("type") == "nsfw"),
                    default=0.0,
                ),
                "violence": max(
                    (v.confidence for v in all_violations if v.details.get("type") == "violence"),
                    default=0.0,
                ),
            },
            "video_path": video_path,
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "sample_rate": self.sample_rate,
            "fps": fps,
            "duration": duration,
            "processing_time": processing_time,
            "violations": [
                {
                    "timestamp": v.timestamp,
                    "frame_number": v.frame_number,
                    "category": v.category,
                    "confidence": v.confidence,
                    "details": v.details,
                }
                for v in all_violations
            ],
            "is_safe": is_safe,
            "violation_count": len(all_violations),
            "nsfw_count": sum(
                1 for v in all_violations if v.details.get("type") == "nsfw"
            ),
            "violence_count": sum(
                1 for v in all_violations if v.details.get("type") == "violence"
            ),
        }

        logger.info(
            f"Moderation complete: {processed_count} frames processed in {processing_time:.2f}s"
        )
        logger.info(f"Found {len(all_violations)} violations")

        return result

    def moderate_video_file(self, file_storage) -> Dict:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            file_storage.save(tmp_file.name)
            temp_path = tmp_file.name

        try:
            result = self.moderate_video(temp_path)
            return result
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _sanitize_youtube_url(self, url: str) -> str:
        """Ekstrak video ID dari YouTube URL dan buat URL clean.
        Handles youtu.be, youtube.com/watch, dan URL dengan extra params seperti ?si=..."""
        # Try youtu.be/VIDEO_ID format dulu
        match = re.search(r'youtu\.be/([a-zA-Z0-9_-]{10,11})', url)
        if not match:
            # Fallback ke youtube.com/watch?v=VIDEO_ID
            match = re.search(r'youtube\.com/watch\?.*?v=([a-zA-Z0-9_-]{10,11})', url)
        if match:
            video_id = match.group(1)
            clean_url = f"https://www.youtube.com/watch?v={video_id}"
            logger.info(f"Extracted video ID: {video_id} -> {clean_url}")
            return clean_url
        logger.warning(f"Could not extract video ID from: {url}")
        return url

    def moderate(self, video_url: str, max_frames: Optional[int] = None) -> Dict:
        """
        Moderate a video from any URL (YouTube, Vimeo, xnxx, any yt-dlp supported site).
        This is the method called by app.py.

        Args:
            video_url: URL of the video
            max_frames: Maximum number of frames to process (None for all)

        Returns:
            Dictionary containing moderation results
        """
        import shutil

        logger.info(f"Starting moderation for URL: {video_url}")

        is_youtube = "youtube.com" in video_url or "youtu.be" in video_url
        if is_youtube:
            video_url = self._sanitize_youtube_url(video_url)
            logger.info(f"Sanitized YouTube URL: {video_url}")

        # Bikin temp dir supaya yt-dlp bisa tulis file dengan nama-nya sendiri
        temp_dir = tempfile.mkdtemp()
        temp_path = None

        try:
            logger.info(f"Downloading video with yt-dlp from: {video_url}")

            ydl_opts = {
                "format": "best[ext=mp4]/best[ext=webm]/best",
                "outtmpl": os.path.join(temp_dir, "video.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            # Cari file yang ter-download di temp_dir
            downloaded_files = os.listdir(temp_dir)
            if not downloaded_files:
                raise Exception("yt-dlp did not download any file")

            temp_path = os.path.join(temp_dir, downloaded_files[0])
            logger.info(f"Video downloaded to {temp_path} ({os.path.getsize(temp_path)} bytes)")

            if os.path.getsize(temp_path) == 0:
                raise Exception("Downloaded file is empty")

            result = self.moderate_video(temp_path, max_frames=max_frames)

            # Kalau 0 frames ter-process, tandai sebagai error
            if result["processed_frames"] == 0:
                result["status"] = "error"
                result["error"] = "No frames could be extracted from the video"
                logger.warning("No frames extracted from video")

            return result

        except Exception as e:
            logger.error(f"Error moderating video from URL: {str(e)}")
            logger.error(traceback.format_exc())
            raise Exception(f"Failed to moderate video: {str(e)}")

        finally:
            # Cleanup seluruh temp_dir
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory: {str(e)}")