import numpy as np
import pyaudio
from openwakeword.model import Model

model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
mic = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=16000,
                             input=True, frames_per_buffer=1280)
print("Listening... say 'Hey Jarvis' (Ctrl+C to stop)")
while True:
    audio = np.frombuffer(mic.read(1280), dtype=np.int16)
    scores = model.predict(audio)
    for name, score in scores.items():
        if score > 0.5:
            print(f"Wake word detected! ({score:.2f})")