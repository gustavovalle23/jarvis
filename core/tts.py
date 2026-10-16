from __future__ import annotations

import logging
import queue
import re
import threading

import numpy as np
import torch

from .state import Mode
from .voices import AUTO, VoiceProfile, get_voice

log = logging.getLogger("jarvis.tts")


def clean_for_speech(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^\s*([-*•]|\d+\.)\s+", "", text, flags=re.M)
    text = re.sub(r"[*_#`>|~]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_model(cfg):
    from omnivoice import OmniVoice

    return OmniVoice.from_pretrained(cfg.voice_model, device_map=cfg.voice_device, dtype=torch.float16)


def build_voice(model, profile: VoiceProfile, transcriber=None):
    if profile.ref_audio is None:
        raise FileNotFoundError(f"{profile.folder} has no ref.wav")
    text = profile.ref_text
    if not text:
        if transcriber is None:
            raise FileNotFoundError(f"{profile.ref_text_file} is missing")
        text, _ = transcriber.transcribe(str(profile.ref_audio))
        if not text:
            raise RuntimeError(f"No words could be heard in {profile.ref_audio}")
        profile.ref_text_file.write_text(text + "\n", encoding="utf-8")
        log.info('Transcribed %s: "%s"', profile.ref_audio.name, text)
    prompt = model.create_voice_clone_prompt(ref_audio=str(profile.ref_audio), ref_text=text)
    prompt.save(str(profile.prompt_file))
    log.info('Built voice "%s"', profile.name)
    return prompt


def synthesize(model, prompt, text: str) -> np.ndarray:
    kwargs = {"voice_clone_prompt": prompt} if prompt is not None else {}
    with torch.inference_mode():
        audio = model.generate(text=text, **kwargs)[0]
    if hasattr(audio, "cpu"):
        audio = audio.float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    return audio * (0.9 / peak) if peak > 0.9 else audio


class Voice:
    def __init__(self, cfg, transcriber=None, rebuild: bool = False) -> None:
        from omnivoice import VoiceClonePrompt

        self.model = load_model(cfg)
        self.prompt = None
        if cfg.voice.lower() == AUTO:
            log.info("Using OmniVoice's automatic voice")
            return
        profile = get_voice(cfg, cfg.voice)
        if rebuild or profile.needs_build:
            self.prompt = build_voice(self.model, profile, transcriber)
        else:
            self.prompt = VoiceClonePrompt.load(str(profile.prompt_file))
        log.info('Voice "%s" loaded', profile.name)

    def synthesize(self, text: str) -> np.ndarray:
        return synthesize(self.model, self.prompt, text)


class Mouth:
    def __init__(self, voice: Voice, speaker, state) -> None:
        self.voice, self.speaker, self.state = voice, speaker, state
        self._text: queue.Queue = queue.Queue()
        self._audio: queue.Queue = queue.Queue()
        self._gen = 0
        self._pending = 0
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        threading.Thread(target=self._synth_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    def say(self, text: str) -> None:
        text = clean_for_speech(text)
        if not text:
            return
        with self._lock:
            self._pending += 1
            self._idle.clear()
            self._text.put((self._gen, text))

    def stop(self) -> None:
        with self._lock:
            self._gen += 1

    def wait(self) -> None:
        self._idle.wait()

    @property
    def busy(self) -> bool:
        return not self._idle.is_set()

    def _done_one(self) -> None:
        with self._lock:
            self._pending -= 1
            if self._pending <= 0:
                self._pending = 0
                self._idle.set()

    def _synth_loop(self) -> None:
        while True:
            gen, text = self._text.get()
            audio = None
            if gen == self._gen:
                try:
                    audio = self.voice.synthesize(text)
                except Exception:
                    log.exception("Speech synthesis failed for %r", text)
            if audio is not None and gen == self._gen:
                self._audio.put((gen, audio))
            else:
                self._done_one()

    def _play_loop(self) -> None:
        while True:
            gen, audio = self._audio.get()
            if gen == self._gen:
                self.state.update(mode=Mode.SPEAKING, status="Speaking")
                self.speaker.play(audio, keep_going=lambda g=gen: g == self._gen)
            self._done_one()
