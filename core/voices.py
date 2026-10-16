from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

AUTO = "auto"
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")


@dataclass(frozen=True)
class VoiceProfile:
    name: str
    folder: Path

    @property
    def ref_audio(self) -> Path | None:
        for ext in AUDIO_EXTENSIONS:
            path = self.folder / f"ref{ext}"
            if path.exists():
                return path
        return None

    @property
    def ref_text_file(self) -> Path:
        return self.folder / "ref.txt"

    @property
    def ref_text(self) -> str:
        if not self.ref_text_file.exists():
            return ""
        return " ".join(self.ref_text_file.read_text(encoding="utf-8").split())

    @property
    def prompt_file(self) -> Path:
        return self.folder / "voice.pt"

    @property
    def sample_file(self) -> Path:
        return self.folder / "sample.wav"

    @property
    def needs_build(self) -> bool:
        if not self.prompt_file.exists():
            return True
        built = self.prompt_file.stat().st_mtime
        sources = [p for p in (self.ref_audio, self.ref_text_file) if p is not None and p.exists()]
        return any(p.stat().st_mtime > built for p in sources)

    def status(self) -> str:
        if self.ref_audio is None:
            return "no ref.wav yet"
        if not self.ref_text:
            return "not built (ref.txt will be transcribed automatically)"
        if self.needs_build:
            return "not built, or ref changed since the last build"
        return "ready"


def list_voices(cfg) -> list[VoiceProfile]:
    cfg.voices_dir.mkdir(parents=True, exist_ok=True)
    return [VoiceProfile(p.name, p) for p in sorted(cfg.voices_dir.iterdir()) if p.is_dir()]


def get_voice(cfg, name: str) -> VoiceProfile:
    folder = cfg.voices_dir / name
    if not folder.is_dir():
        available = ", ".join(v.name for v in list_voices(cfg)) or "none"
        raise FileNotFoundError(f'Voice "{name}" not found in {cfg.voices_dir} (available: {available}, or "auto")')
    return VoiceProfile(name, folder)
