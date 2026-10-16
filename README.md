# J.A.R.V.I.S.

A voice assistant that runs on your own PC, with a holographic orb that reacts while it listens, thinks and talks.

Say **"Hey Jarvis"**, ask anything, and it answers out loud in a cloned voice. The only thing that leaves your machine is the question sent to Claude, which runs on your existing Claude subscription. It uses no API key and costs nothing extra.

## How it works

```
 microphone (always on)
      │
      ▼
 openWakeWord      waits for "Hey Jarvis"                      local, CPU
      │
      ▼
 recorder          records until you stop talking              local
      │
      ▼
 faster-whisper    turns your speech into text                 local, GPU
      │
      ▼
 Claude Code       thinks of the answer, can search the web    your Claude subscription
      │            (streams the reply sentence by sentence)
      ▼
 OmniVoice         speaks each sentence in the cloned voice    local, GPU
      │
      ▼
 speakers  +  orb window showing every step live
```

| Part | What it does | Size |
|---|---|---|
| **openWakeWord** | Listens all the time for one phrase ("Hey Jarvis") and ignores everything else. Tiny, runs on the CPU. | ~50 MB |
| **faster-whisper** | Speech-to-text. Understands Portuguese, English and many other languages, detected automatically. | ~0.5 GB (`small`) |
| **Claude Code** | The brain. One Claude Code process stays open for the whole session, so there's no startup delay per question and it remembers the conversation. | already installed |
| **OmniVoice** | Text-to-speech with voice cloning from a 3–10 second clip. While one sentence plays, the next one is being generated. | ~3.3 GB |
| **pygame orb** | The 3D particle sphere and on-screen display. Its color and motion follow the state and the audio level. | ~10 MB |

### Orb states

| State | Color | Meaning |
|---|---|---|
| Standby | Blue, slow breathing | Waiting for "Hey Jarvis" |
| Listening | Teal, follows your voice | Recording your question |
| Thinking | Violet, fast spin | Transcribing and waiting for Claude |
| Speaking | Cyan, pulses with the voice | Answering |
| Error | Red | Something failed; the status line under the orb says what |

### Conversation flow

1. Say **"Hey Jarvis"** (or press Space). You hear a short beep.
2. Ask your question. A pause of 0.9 seconds ends it.
3. Jarvis starts speaking as soon as Claude's first sentence arrives.
4. For 4 seconds after the answer, you can ask a follow-up without the wake word.
5. Say **"Hey Jarvis"** at any point while it's thinking or talking to cut it off and ask something new.

## Requirements

- Windows 10/11 or Linux with an **NVIDIA GPU** (6 GB+ of VRAM; tested on an RTX 5060 with 8 GB). It also runs on a Mac, but slowly.
- **Python 3.12**
- **Claude Code**, logged in with a Claude Pro or Max subscription
- About **10 GB** of free disk space
- A microphone and speakers (headphones work best)

## Installation

### 1. Install the prerequisites

```powershell
winget install Python.Python.3.12
```

Install [Claude Code](https://claude.com/claude-code) and run `claude` once to log in.

On Linux, also install PortAudio for the microphone: `sudo apt install portaudio19-dev python3.12-venv`.

### 2. Run the installer

Open **Git Bash** (on Windows) or a terminal (on Linux) in the Jarvis folder:

```bash
cd /d/Jarvis
bash install.sh
```

The installer:
1. Creates `.venv`.
2. Installs PyTorch with CUDA 12.8 and everything in `requirements.txt`.
3. Checks that the GPU is detected.
4. Downloads all models (about 4 GB) into `models/`.
5. Checks that Claude Code is available.

It's safe to run again; finished steps are skipped.

<details>
<summary>Manual install (PowerShell, without the script)</summary>

```powershell
cd D:\Jarvis
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import openwakeword; openwakeword.utils.download_models()"
```
</details>

### 3. Clone a voice

See [Voices](#voices) below. In short: save a clip as `voices/jarvis/ref.wav` and run `python jarvis.py --build-voice jarvis`.

## Voices

You can have as many voices as you like. Each one is a folder inside `voices/`, and `.env` decides which one Jarvis uses.

```
voices/
  jarvis/
    ref.wav      your 3–10 second clip                   (you add this)
    ref.txt      the exact words spoken in ref.wav       (written for you, fix it if wrong)
    voice.pt     the built voice                         (created by --build-voice)
    sample.wav   a test sentence in this voice           (created by --build-voice)
  narrator/
    ref.wav
    ...
```

### Add a voice

1. Create a folder named after the voice, for example `voices/narrator/`.
2. Record **3–10 seconds** of clean speech, with no music or background noise, and save it as `ref.wav` in that folder. `.flac`, `.mp3` and `.ogg` also work, but `.wav` is best.
3. Build it:
   ```bash
   python jarvis.py --build-voice narrator     # one voice
   python jarvis.py --build-voice              # every voice in voices/
   ```
   If there's no `ref.txt`, Whisper writes down what the clip says and saves it there. **Open `ref.txt` and check it word for word.** If it's wrong, the voice will say strange words. Fix it and build again.
4. Listen to `voices/narrator/sample.wav`.

### Choose a voice

Edit `.env`:

```ini
JARVIS_VOICE=narrator
```

Then restart Jarvis. You can also try a voice for one run only with `python jarvis.py --voice narrator`, or use `JARVIS_VOICE=auto` for OmniVoice's built-in voice.

```bash
python jarvis.py --list-voices
```

```
 * jarvis             ready
   narrator           not built, or ref changed since the last build
   auto               OmniVoice's built-in voice
```

If you change `ref.wav` or `ref.txt`, the voice is rebuilt automatically the next time it's used.

Tips:
- The voice keeps the accent of its clip. A Portuguese clip gives English answers a Portuguese accent, and the other way round. For the most natural results, record a clip in the language Jarvis will mostly speak.
- Calm, steady speech gives a calmer voice. The clip's tone, speed and microphone quality all carry over.
- Only clone voices you have permission to use.

## Running it

| How | Command |
|---|---|
| Double-click (Windows) | `start_jarvis.bat` |
| Git Bash / Linux | `.venv/Scripts/python.exe jarvis.py` (Linux: `.venv/bin/python jarvis.py`) |
| PowerShell | `.\.venv\Scripts\Activate.ps1` then `python jarvis.py` |
| Orb only, no models | `python jarvis.py --demo` |
| List voices | `python jarvis.py --list-voices` |
| Build a voice (or all) | `python jarvis.py --build-voice [NAME]` |
| Use a voice for this run only | `python jarvis.py --voice NAME` |
| Debug logging | `python jarvis.py --verbose` |

The first start takes about 15–20 seconds to load the models, then Jarvis greets you.

### Controls

| Input | Action |
|---|---|
| "Hey Jarvis" | Start talking; while it's thinking or answering, it stops and listens again |
| Space or clicking the orb | Same as "Hey Jarvis" |
| Esc | Stop the current answer and go back to standby |
| F11 | Toggle fullscreen |
| Closing the window | Quit |

## Configuration

`.env` (a copy of `.env.example`, not committed to git) holds the per-machine choice:

| Variable | Default | Meaning |
|---|---|---|
| `JARVIS_VOICE` | `jarvis` | The folder in `voices/` to speak with, or `auto` |

Every other setting is in `core/config.py`.

| Setting | Default | Change it when… |
|---|---|---|
| `wake_threshold` | `0.5` | Lower it (around 0.35) if it misses you; raise it if it triggers by itself |
| `barge_in_threshold` | `0.65` | The wake-word threshold while Jarvis is talking. Lower it if it won't stop when you call it; raise it if it interrupts itself |
| `whisper_model` | `small` | Use `medium` or `large-v3-turbo` for better accuracy, especially in Portuguese |
| `language` | `None` (auto) | Set `"pt"` or `"en"` to always use one language |
| `claude_model` | `sonnet` | `haiku` is the fastest; `opus` is the smartest |
| `claude_tools` | `WebSearch`, `WebFetch` | Tools Jarvis may use without asking |
| `persona` | JARVIS style | Its personality and speaking rules |
| `silence_seconds` | `0.9` | Raise it if Jarvis cuts you off mid-sentence |
| `follow_up_seconds` | `4.0` | `0` turns follow-ups off |
| `greeting` | "All systems online…" | Set to `""` for a silent start |

Facts Jarvis should always know, such as your name, city or preferences, go in `workspace/CLAUDE.md`. Claude Code reads that file at the start of every session.

## Project layout

```
jarvis.py            entry point, --demo, and the voice commands
start_jarvis.bat     Windows launcher
install.sh           installer
requirements.txt     Python packages
.env.example         template for .env (selected voice)
.gitignore           keeps models, voices, logs and secrets out of git
voices/              one folder per cloned voice (ref.wav, ref.txt, voice.pt, sample.wav)
core/
  config.py          all settings, the persona, and .env loading
  voices.py          finds voices and decides when one needs building
  assistant.py       the loop: wake → listen → transcribe → think → speak, plus interruptions
  audio.py           microphone (adaptive noise floor, recording) and speaker (playback, chimes)
  wake.py            openWakeWord wrapper
  stt.py             faster-whisper wrapper, with a CPU fallback and a filter for invented phrases
  brain.py           Claude Code process: stream-json, sentence streaming, interrupt
  tts.py             OmniVoice voice, and the "mouth" that queues, plays and stops sentences
  orb.py             the orb and on-screen display (pygame)
  state.py           state shared between the assistant and the window
workspace/CLAUDE.md  what Jarvis knows about you
models/              downloaded models (HF_HOME)
jarvis.log           log of every run
```

### Under the hood

- **Threads:** the window runs on the main thread. The assistant loop runs in a background thread, and they share one `State` object. Voice generation and playback each have their own thread.
- **Wake word and recording:** the microphone stays open and feeds 80 ms chunks into a queue. The wake-word detector and the recorder take turns reading it. A speech threshold that adapts to background noise decides when you start and stop talking.
- **Talking to Claude:** Claude Code runs as `claude -p --input-format stream-json --output-format stream-json --include-partial-messages`. Each question is one JSON line in, and the reply streams back word by word. Each complete sentence is spoken straight away.
- **Interrupting:** "Hey Jarvis", Space or Esc does two things. It sends Claude Code an `interrupt` control request, which stops the answer and any running tool while keeping the conversation memory. It also marks every queued sentence as out of date, so the voice stops within 20 ms.

## Troubleshooting

| Problem | Fix |
|---|---|
| Nothing happens on "Hey Jarvis" | Press Space to check that the rest works. Then check the Windows input device (Settings → System → Sound → Input) and lower `wake_threshold` |
| The voice says strange words before every answer | That voice's `ref.txt` doesn't match its `ref.wav`. Fix it and run `--build-voice NAME` |
| `Voice "x" not found` | `JARVIS_VOICE` in `.env` doesn't match a folder in `voices/`. Run `--list-voices` |
| `CUDA available: False` | Update the NVIDIA driver. RTX 50-series cards need the `cu128` PyTorch build |
| A `cublas` / `cudnn` DLL error | `pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"` |
| "Claude Code ('claude') was not found" | Install Claude Code and make sure `claude` works in a new terminal |
| It cuts you off mid-sentence | Raise `silence_seconds` |
| It interrupts itself while talking | Use headphones, or raise `barge_in_threshold` |
| Anything else | Check `jarvis.log` and run with `--verbose` |
