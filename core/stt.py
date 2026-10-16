from __future__ import annotations

import logging
import re

import numpy as np
import torch
from faster_whisper import WhisperModel

log = logging.getLogger("jarvis.stt")

HALLUCINATIONS = ("thanks for watching", "subtitles by", "amara.org", "legendas pela comunidade",
                  "obrigado por assistir", "inscreva-se")


class Transcriber:
    def __init__(self, cfg) -> None:
        self.language = cfg.language
        try:
            self.model = WhisperModel(cfg.whisper_model, device=cfg.whisper_device,
                                      compute_type=cfg.whisper_compute)
        except Exception as exc:
            log.warning("Whisper on %s failed (%s); falling back to CPU", cfg.whisper_device, exc)
            self.model = WhisperModel(cfg.whisper_model, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        segments, info = self.model.transcribe(audio, language=self.language, vad_filter=True,
                                               beam_size=5, condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segments).strip()
        plain = re.sub(r"[^\w\s.-]", "", text.lower()).strip()
        if len(plain) < 2 or any(h in plain for h in HALLUCINATIONS):
            return "", info.language
        return text, info.language
