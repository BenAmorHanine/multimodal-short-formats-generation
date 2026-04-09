# ImaginationModel

A trimodal continual pre-training pipeline that builds a vision-audio-language dataset
from VAST-150K annotations and prepares it for fine-tuning an imagination-capable
multimodal model.

## Overview

The pipeline downloads YouTube video clips referenced in VAST-150K, extracts raw frames
and audio waveforms per segment, and stores them as a flat folder structure ready for
training. No GPU is required for data collection.

Each segment produces:
- `frame_00.jpg`, `frame_01.jpg` — raw RGB frames at native resolution, JPEG quality 95
- `audio.npy` — float32 mono waveform at 16kHz, shape `(32000,)` for 2s segments
- `text.txt` — subtitle or VAST caption from the annotation

## Dataset

Source: [VAST-150K](https://github.com/txh-mercury/VAST) — 150,154 video-audio-text
annotations with `clip_span`, `subtitle`, `vast_cap`, `vision_cap`, and `audio_cap` fields.

Target: 15,000 trimodal segments (~89MB per 272 segments → ~5GB for 15K segments).

## Repository Structure

repo/ \
├── data/                        # gitignored — segments live here locally\
├── scripts/   \
│   └── download_dataset.py      # full CLI pipeline        \
├── notebooks/   \
│   └── VAST_150K_init.ipynb     # original exploration notebook   \
├── requirements.txt \
└── README.md      \


## Notes

- ~8–10s per annotation on average (dominated by yt-dlp download speed)
- ~5–10% of URLs are unavailable/private and are silently skipped
- Resume is exact: checkpoint stores annotation index + segment count,
  verified against actual folder count on disk at startup
- Runs fully on CPU — no GPU needed for extraction