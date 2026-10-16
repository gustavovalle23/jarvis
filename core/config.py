from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PERSONA = """You are J.A.R.V.I.S., a voice assistant running on the user's Windows PC, modeled on Tony Stark's AI: calm, precise, quietly witty and unfailingly polite. Address the user as "sir" now and then, not in every reply.

Everything you write is converted to speech, so:
- Reply in the same language the user spoke.
- Keep it short: one to three sentences unless the user asks for detail.
- Plain spoken text only. No markdown, lists, headings, code blocks, emojis or URLs.
- Write numbers, times and units the way they are said aloud.
- Before using a tool, say one short sentence such as "One moment, sir."
- Never say your own name; the wake word is "Hey Jarvis" and hearing it from you would interrupt you.
"""


@dataclass
class Config:
    wake_model: str = "hey_jarvis"
    wake_threshold: float = 0.5
    barge_in: bool = True
    barge_in_threshold: float = 0.65

    sample_rate: int = 16000
    chunk: int = 1280
    input_device: int | None = None

    silence_seconds: float = 0.9
    max_record_seconds: float = 15.0
    no_speech_timeout: float = 5.0
    follow_up_seconds: float = 4.0
    speech_margin: float = 3.0
    min_speech_level: float = 0.012

    whisper_model: str = "small"
    whisper_device: str = "cuda"
    whisper_compute: str = "float16"
    language: str | None = None

    claude_model: str = "sonnet"
    claude_tools: list[str] = field(default_factory=lambda: ["WebSearch", "WebFetch"])
    claude_workdir: Path = ROOT / "workspace"
    claude_timeout: float = 180.0
    persona: str = PERSONA

    voice: str = "jarvis"
    voices_dir: Path = ROOT / "voices"
    voice_sample_text: str = "Good evening, sir. All systems are online and ready."
    voice_model: str = "k2-fsa/OmniVoice"
    voice_device: str = "cuda:0"
    output_rate: int = 24000
    output_device: int | None = None
    chimes: bool = True

    window_size: tuple[int, int] = (1000, 860)
    fps: int = 60
    greeting: str = "All systems online. Good to see you, sir."

    @classmethod
    def load(cls, env_file: Path = ROOT / ".env") -> Config:
        cfg = cls()
        env = {**read_env(env_file), **os.environ}
        if env.get("JARVIS_VOICE", "").strip():
            cfg.voice = env["JARVIS_VOICE"].strip()
        return cfg


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.split(" #", 1)[0].strip().strip("\"'")
    return values
