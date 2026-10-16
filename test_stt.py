import torch  # load first: faster-whisper uses the NVIDIA libraries that come with PyTorch
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cuda", compute_type="float16")
segments, info = model.transcribe("ref.wav", vad_filter=True)
print("Language:", info.language)
for s in segments:
    print(s.text)
