from __future__ import annotations

from enum import Enum


class Mode(str, Enum):
    BOOT = "booting"
    IDLE = "standby"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


class State:
    def __init__(self) -> None:
        self.mode = Mode.BOOT
        self.status = "Initializing systems"
        self.user_text = ""
        self.reply_text = ""
        self.mic_level = 0.0
        self.out_level = 0.0
        self.running = True

    def update(self, **values) -> None:
        for key, value in values.items():
            setattr(self, key, value)
