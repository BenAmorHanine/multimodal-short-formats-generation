"""
Stage 2 + 3 preprocessing for segment text preparation.

Notebook-aligned logic:
1) Extract per-segment audio
2) Whisper-AT transcription (with relaxed fallback)
3) Classify audio as speech/sound/silence
4) Build text per segment:
   - speech: transcript
   - sound: AudioSet labels + BLIP caption
   - silence/other: transcript + BLIP caption fallback
"""

from __future__ import annotations

import os
import tempfile
import warnings
from typing import Any, Callable, Optional

import cv2
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from tqdm import tqdm

from shared_utils.audio_processing import extract_audio_segment
from shared_utils.video_processing import compute_segments


class AudioClassifier:
    """
    Three-class audio segmentation with trust score.
    """

    def __init__(
        self,
        speech_energy_threshold: float = 0.015,
        sound_energy_threshold: float = 0.003,
        no_speech_threshold: float = 0.6,
    ):
        self.speech_energy_threshold = speech_energy_threshold
        self.sound_energy_threshold = sound_energy_threshold
        self.no_speech_threshold = no_speech_threshold

    @staticmethod
    def _rms(audio_array: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio_array.astype(np.float32) ** 2)))

    @staticmethod
    def _mean_no_speech_prob(whisper_result: Optional[dict[str, Any]]) -> float:
        if not whisper_result:
            return 1.0
        segs = whisper_result.get("segments", [])
        if not segs:
            return 1.0
        return float(np.mean([s.get("no_speech_prob", 1.0) for s in segs]))

    def classify(
        self,
        audio_array: np.ndarray,
        whisper_result: Optional[dict[str, Any]],
    ) -> tuple[str, float, dict[str, Any]]:
        rms = self._rms(audio_array)
        no_speech_prob = self._mean_no_speech_prob(whisper_result)
        transcript = (whisper_result or {}).get("text", "").strip()
        has_text = bool(transcript)

        if rms < self.sound_energy_threshold:
            audio_type, text_trust = "silence", 0.4
        elif has_text and no_speech_prob < self.no_speech_threshold:
            audio_type, text_trust = "speech", 1.0
        elif rms >= self.speech_energy_threshold and no_speech_prob < 0.8 and has_text:
            audio_type, text_trust = "speech", 0.75
        else:
            audio_type, text_trust = "sound", 0.5

        meta = {
            "rms_energy": rms,
            "no_speech_prob": no_speech_prob,
            "has_transcript": has_text,
        }
        return audio_type, text_trust, meta


class TextProducer:
    """
    Produces {text, trust, source, start, end} for each segment.

    Required injected models:
    - wat_model: Whisper-AT model
    - blip_model: BLIP-1 generation model
    - blip_processor: BLIP processor
    """

    def __init__(
        self,
        wat_model: Any,
        blip_model: Any,
        blip_processor: Any,
        parse_at_label_fn: Optional[Callable[..., Any]] = None,
        device: str = "cuda",
        audio_sr: int = 16000,
    ):
        self.wat_model = wat_model
        self.blip_model = blip_model
        self.blip_processor = blip_processor
        self.device = device
        self.audio_sr = audio_sr
        self.classifier = AudioClassifier()

        if parse_at_label_fn is not None:
            self.parse_at_label = parse_at_label_fn
        else:
            try:
                from whisper_at import parse_at_label

                self.parse_at_label = parse_at_label
            except Exception:
                self.parse_at_label = None

    def _transcribe_with_whisper_at(self, wav_path: str) -> dict[str, Any]:
        """Notebook-equivalent two-pass Whisper-AT transcription."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.wat_model.transcribe(wav_path, at_time_res=2.0, language=None)

        if result.get("text", "").strip():
            return result

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.wat_model.transcribe(
                wav_path,
                at_time_res=2.0,
                no_speech_threshold=0.3,
                logprob_threshold=None,
                compression_ratio_threshold=None,
                condition_on_previous_text=False,
                language=None,
            )

    def _extract_middle_frame(self, video_path: str, start: float, end: float) -> Optional[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, ((start + end) / 2.0) * 1000.0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def _blip_caption(self, frame: Optional[Image.Image]) -> str:
        if frame is None:
            return ""

        dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        inputs = self.blip_processor(frame, return_tensors="pt").to(self.device, dtype)
        with torch.no_grad():
            ids = self.blip_model.generate(**inputs, max_new_tokens=30)
        return self.blip_processor.decode(ids[0], skip_special_tokens=True)

    def _get_at_label(
        self,
        wat_result: dict[str, Any],
        top_k: int = 2,
        score_threshold: float = 0.0,
    ) -> str:
        if self.parse_at_label is None:
            return ""

        try:
            tags = self.parse_at_label(
                wat_result,
                top_k=top_k,
                p_threshold=score_threshold,
            )
            if not tags:
                return ""
            audio_tags = tags[0].get("audio tags", [])
            positive = [
                (name, score)
                for name, score in audio_tags
                if score > score_threshold
            ]
            if not positive:
                return ""
            return ", ".join(name for name, _ in positive[:top_k])
        except Exception:
            return ""

    def produce(self, video_path: str, start: float, end: float) -> dict[str, Any]:
        """Generate text/trust/source for a single segment."""
        try:
            audio = extract_audio_segment(video_path, start, end, sr=self.audio_sr)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, self.audio_sr)
                wav_path = f.name

            try:
                wat_result = self._transcribe_with_whisper_at(wav_path)
            finally:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)

            transcript = wat_result.get("text", "").strip()
            audio_type, text_trust, _ = self.classifier.classify(audio, wat_result)

            if audio_type == "speech" and transcript:
                combined_text = transcript
                source = "speech"
            elif audio_type == "sound":
                at_label = self._get_at_label(wat_result, top_k=2, score_threshold=0.0)
                frame = self._extract_middle_frame(video_path, start, end)
                caption = self._blip_caption(frame)
                parts = [p for p in [at_label, caption] if p]
                combined_text = ". ".join(parts) if parts else "audio scene"
                source = "whisper_at+blip1"
            else:
                frame = self._extract_middle_frame(video_path, start, end)
                caption = self._blip_caption(frame)
                combined_text = (transcript + " " + caption).strip() or "visual scene"
                source = "blip1" if caption else "fallback"

            return {
                "text": combined_text,
                "trust": round(text_trust, 4),
                "source": source,
                "start": start,
                "end": end,
            }
        except Exception:
            return {
                "text": "visual scene",
                "trust": 0.4,
                "source": "error",
                "start": start,
                "end": end,
            }


def run_preprocessing(
    video_path: str,
    producer: TextProducer,
    window_size: int = 2,
    stride: int = 1,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Run Stage 2+3 text preparation on all segments.
    """
    segments, fps, duration = compute_segments(video_path, window_size, stride)

    if verbose:
        print(f"Video: {os.path.basename(video_path)}")
        print(f"  Duration : {duration:.1f}s  |  FPS: {fps:.2f}")
        print(f"  Segments : {len(segments)} (window={window_size}s, stride={stride}s)")

    iterator = (
        tqdm(enumerate(segments), total=len(segments), desc="Preprocessing")
        if verbose
        else enumerate(segments)
    )

    segment_data: dict[str, dict[str, Any]] = {}
    for i, (start, end) in iterator:
        segment_data[str(i)] = producer.produce(video_path, start, end)

    if verbose:
        sources: dict[str, int] = {}
        for item in segment_data.values():
            src = item["source"]
            sources[src] = sources.get(src, 0) + 1
        print(f"\nDone. Sources: {sources}")

    return segment_data