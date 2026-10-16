from __future__ import annotations

import numpy as np
from openwakeword.model import Model


class WakeWord:
    def __init__(self, cfg) -> None:
        self.threshold = cfg.wake_threshold
        self.model = Model(wakeword_models=[cfg.wake_model], inference_framework="onnx")

    def score(self, chunk: np.ndarray) -> float:
        return max(self.model.predict(chunk).values(), default=0.0)

    def detected(self, chunk: np.ndarray, threshold: float | None = None) -> bool:
        return self.score(chunk) >= (self.threshold if threshold is None else threshold)

    def reset(self) -> None:
        if hasattr(self.model, "reset"):
            self.model.reset()
