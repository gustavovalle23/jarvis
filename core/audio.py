from __future__ import annotations

import logging
import queue
import time
from typing import Callable

import numpy as np
import pyaudio

log = logging.getLogger("jarvis.audio")

_pa: pyaudio.PyAudio | None = None


def portaudio() -> pyaudio.PyAudio:
    global _pa
    if _pa is None:
        _pa = pyaudio.PyAudio()
    return _pa


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    x = samples.astype(np.float32)
    if samples.dtype == np.int16:
        x /= 32768.0
    return float(np.sqrt(np.mean(x * x)))


class Microphone:
    def __init__(self, cfg, on_level: Callable[[float], None] | None = None) -> None:
        self.cfg = cfg
        self.on_level = on_level
        self.chunk_seconds = cfg.chunk / cfg.sample_rate
        self.noise = 0.003
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self.stream = portaudio().open(
            format=pyaudio.paInt16, channels=1, rate=cfg.sample_rate, input=True,
            frames_per_buffer=cfg.chunk, input_device_index=cfg.input_device,
            stream_callback=self._callback,
        )
        self.stream.start_stream()

    def _callback(self, data, frames, time_info, status):
        chunk = np.frombuffer(data, dtype=np.int16).copy()
        level = rms(chunk)
        rate = 0.05 if level < self.noise * 2 else 0.002
        self.noise = max(0.001, self.noise + (level - self.noise) * rate)
        if self.on_level:
            self.on_level(min(1.0, level * 9))
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(chunk)
            except (queue.Empty, queue.Full):
                pass
        return None, pyaudio.paContinue

    def read(self, timeout: float = 0.5) -> np.ndarray:
        return self._queue.get(timeout=timeout)

    def flush(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def record(self, no_speech_timeout: float, should_stop: Callable[[], bool],
               ignore_seconds: float = 0.0) -> np.ndarray | None:
        cfg = self.cfg
        threshold = max(self.noise * cfg.speech_margin, cfg.min_speech_level)
        chunks: list[np.ndarray] = []
        started = False
        silence = elapsed = 0.0
        while not should_stop():
            try:
                chunk = self.read(timeout=0.5)
            except queue.Empty:
                continue
            chunks.append(chunk)
            elapsed += self.chunk_seconds
            if rms(chunk) > threshold and elapsed > ignore_seconds:
                started, silence = True, 0.0
            elif started:
                silence += self.chunk_seconds
            if started and silence >= cfg.silence_seconds:
                break
            if not started and elapsed >= no_speech_timeout:
                return None
            if elapsed >= cfg.max_record_seconds:
                break
        if not started:
            return None
        return np.concatenate(chunks).astype(np.float32) / 32768.0

    def close(self) -> None:
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception:
            pass


class Speaker:
    def __init__(self, cfg, on_level: Callable[[float], None] | None = None) -> None:
        self.source_rate = cfg.output_rate
        self.on_level = on_level
        self.stream = None
        for rate in (cfg.output_rate, 48000, 44100):
            try:
                self.stream = portaudio().open(format=pyaudio.paFloat32, channels=1, rate=rate,
                                               output=True, output_device_index=cfg.output_device)
                self.rate = rate
                break
            except Exception as exc:
                log.debug("Output at %d Hz failed: %s", rate, exc)
        if self.stream is None:
            raise RuntimeError("Could not open the speakers")
        self.block = self.rate // 50

    def _resample(self, audio: np.ndarray, rate: int) -> np.ndarray:
        if rate == self.rate:
            return audio
        n = int(len(audio) * self.rate / rate)
        return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)

    def play(self, audio: np.ndarray, rate: int | None = None,
             keep_going: Callable[[], bool] = lambda: True) -> bool:
        audio = self._resample(np.asarray(audio, dtype=np.float32), rate or self.source_rate)
        try:
            for i in range(0, len(audio), self.block):
                if not keep_going():
                    return False
                block = audio[i:i + self.block]
                if self.on_level:
                    self.on_level(min(1.0, rms(block) * 5))
                self.stream.write(block.tobytes())
            return True
        finally:
            if self.on_level:
                self.on_level(0.0)

    def chime(self, rising: bool = True) -> np.ndarray:
        notes = (880, 1320) if rising else (1320, 880)
        n = int(self.rate * 0.07)
        t = np.arange(n) / self.rate
        env = np.sin(np.pi * t / t[-1]) ** 2
        tone = np.concatenate([np.sin(2 * np.pi * f * t) * env for f in notes])
        return (tone * 0.18).astype(np.float32)

    def close(self) -> None:
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception:
            pass
