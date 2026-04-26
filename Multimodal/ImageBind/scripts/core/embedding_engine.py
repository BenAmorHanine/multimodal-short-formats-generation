"""
ImageBind Embedding Extraction Engine
Stage 1: trimodal encoding from precomputed segment_data.

This engine consumes segment_data produced by
Multimodal.Text_Handler.Preprocessing.run_preprocessing and encodes
Vision + Audio + Text embeddings with ImageBind.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from imagebind import data
from imagebind.models.imagebind_model import ModalityType

from shared_utils.video_processing import save_video_segment


class EmbeddingEngine:
    """
    Encodes pre-segmented video into ImageBind trimodal embeddings.

    Expected segment_data format:
        dict[str(index)] -> {
            "text": str,
            "trust": float,
            "source": str,
            "start": float,
            "end": float,
        }
    """

    def __init__(self, model: Any, device: str, whisper_model: Any | None = None):
        """
        Initialize the engine.

        Args:
            model: ImageBind model in eval mode on device.
            device: Device string (e.g. 'cuda:0' or 'cpu').
            whisper_model: Deprecated and ignored. Kept only for compatibility.
        """
        self.model = model
        self.device = device
        self.whisper_model = whisper_model

    def extract_segment_features(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        segment_text: str,
    ) -> dict[str, np.ndarray | float]:
        """
        Encode one segment into ImageBind embeddings.

        Returns:
            dict with keys: vision, audio, text, combined, start, end
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            segment_path = os.path.join(tmpdir, "segment.mp4")
            save_video_segment(video_path, start_time, end_time, segment_path)

            inputs = {
                ModalityType.AUDIO: data.load_and_transform_audio_data(
                    [segment_path], self.device
                )
            }
            with torch.no_grad():
                audio_emb = self.model(inputs)[ModalityType.AUDIO].cpu().numpy()

            inputs = {
                ModalityType.VISION: data.load_and_transform_video_data(
                    [segment_path],
                    self.device,
                    clip_duration=2,
                    clips_per_video=1,
                )
            }
            with torch.no_grad():
                vision_emb = self.model(inputs)[ModalityType.VISION].cpu().numpy()

            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(
                    [segment_text], self.device
                )
            }
            with torch.no_grad():
                text_emb = self.model(inputs)[ModalityType.TEXT].cpu().numpy()

            combined_emb = (vision_emb + audio_emb + text_emb) / 3.0

        return {
            "vision": vision_emb,
            "audio": audio_emb,
            "text": text_emb,
            "combined": combined_emb,
            "start": start_time,
            "end": end_time,
        }

    def extract_video_features(
        self,
        video_path: str,
        segment_data: dict[str, dict[str, Any]],
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Encode all segments of a video using precomputed segment_data.

        Returns:
            dict with keys:
                vision: np.ndarray [N, 1024]
                audio: np.ndarray [N, 1024]
                text: np.ndarray [N, 1024]
                times: np.ndarray [N, 2]
                raw_text: list[str]
                text_trust: np.ndarray [N]
                errors: list[int]
        """
        if not segment_data:
            raise ValueError("segment_data is empty. Run preprocessing first.")

        N = len(segment_data)

        if verbose:
            print(f"Stage 1 - ImageBind encoding: {N} segments")

        vision_embs: list[torch.Tensor] = []
        audio_embs: list[torch.Tensor] = []
        text_embs: list[torch.Tensor] = []
        times_out: list[list[float]] = []
        raw_texts: list[str] = []
        trust_vals: list[float] = []
        errors: list[int] = []

        iterator = tqdm(range(N), desc="ImageBind encoding") if verbose else range(N)

        for i in iterator:
            seg = segment_data[str(i)]
            start = float(seg["start"])
            end = float(seg["end"])
            text = str(seg["text"])
            trust = float(seg.get("trust", 1.0))

            try:
                feats = self.extract_segment_features(video_path, start, end, text)
                vision_embs.append(torch.tensor(feats["vision"]).squeeze(0))
                audio_embs.append(torch.tensor(feats["audio"]).squeeze(0))
                text_embs.append(torch.tensor(feats["text"]).squeeze(0))
            except Exception as e:
                if verbose:
                    print(f"\nSegment {i} ({start:.1f}->{end:.1f}s) failed: {e}")
                errors.append(i)
                vision_embs.append(torch.zeros(1024))
                audio_embs.append(torch.zeros(1024))
                text_embs.append(torch.zeros(1024))

            times_out.append([start, end])
            raw_texts.append(text)
            trust_vals.append(trust)

        V = F.normalize(torch.stack(vision_embs), dim=-1).numpy()
        A = F.normalize(torch.stack(audio_embs), dim=-1).numpy()
        T = F.normalize(torch.stack(text_embs), dim=-1).numpy()

        if verbose:
            print(f"\nEncoded {N - len(errors)}/{N} segments ({len(errors)} errors)")

        return {
            "vision": V,
            "audio": A,
            "text": T,
            "times": np.array(times_out, dtype=np.float32),
            "raw_text": raw_texts,
            "text_trust": np.array(trust_vals, dtype=np.float32),
            "errors": errors,
        }

    def get_embedding_stats(self, results: dict[str, Any]) -> dict[str, float | int]:
        """
        Compute basic norm statistics on embeddings.
        """
        V = results["vision"]
        A = results["audio"]
        T = results["text"]

        return {
            "num_segments": int(len(V)),
            "vision_norm_mean": float(np.linalg.norm(V, axis=-1).mean()),
            "vision_norm_std": float(np.linalg.norm(V, axis=-1).std()),
            "audio_norm_mean": float(np.linalg.norm(A, axis=-1).mean()),
            "audio_norm_std": float(np.linalg.norm(A, axis=-1).std()),
            "text_norm_mean": float(np.linalg.norm(T, axis=-1).mean()),
            "text_norm_std": float(np.linalg.norm(T, axis=-1).std()),
            "error_count": int(len(results.get("errors", []))),
        }


def create_engine(imagebind_model: Any, device: str, whisper_model: Any | None = None) -> EmbeddingEngine:
    """
    Convenience constructor.

    whisper_model is deprecated and ignored, kept for compatibility with
    older notebooks that still pass a third argument.
    """
    return EmbeddingEngine(imagebind_model, device, whisper_model=whisper_model)
