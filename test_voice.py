from omnivoice import OmniVoice
import soundfile as sf
import torch

model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)

# Save the voice once so you can reuse it later
prompt = model.create_voice_clone_prompt(
    ref_audio="ref.wav",
    ref_text="Type exactly what is said in ref.wav here.",
)
prompt.save("jarvis_voice.pt")

audio = model.generate(text="Good evening, sir. All systems are online.", voice_clone_prompt=prompt)
sf.write("out.wav", audio[0], 24000)
print("Saved out.wav")