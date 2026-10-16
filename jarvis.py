from __future__ import annotations

import argparse
import logging
import math
import os
import threading
import time

from core.config import ROOT, Config
from core.state import Mode, State

os.environ.setdefault("HF_HOME", str(ROOT / "models"))


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / "jarvis.log", encoding="utf-8")],
    )


def demo(state: State) -> None:
    script = [
        (Mode.BOOT, "Initializing systems", "", ""),
        (Mode.IDLE, 'Say "Hey Jarvis" or press SPACE', "", ""),
        (Mode.LISTENING, "Listening", "What's the weather like tomorrow?", ""),
        (Mode.THINKING, "Thinking", "What's the weather like tomorrow?", ""),
        (Mode.SPEAKING, "Speaking", "What's the weather like tomorrow?",
         "Tomorrow looks clear, sir, with a high of twenty six degrees. I'd leave the umbrella at home."),
        (Mode.ERROR, "Demo error state", "", ""),
    ]
    while state.running:
        for mode, status, heard, reply in script:
            state.update(mode=mode, status=status, user_text=heard, reply_text=reply)
            end = time.time() + 4
            while time.time() < end and state.running:
                t = time.time()
                fake = abs(math.sin(t * 7) * math.sin(t * 2.3)) * 0.8
                state.update(mic_level=fake if mode == Mode.LISTENING else 0.0,
                             out_level=fake if mode == Mode.SPEAKING else 0.0)
                time.sleep(0.03)


def list_voices(cfg: Config) -> None:
    from core.voices import list_voices as profiles

    voices = profiles(cfg)
    print(f"Voices in {cfg.voices_dir}   (* = selected)\n")
    for v in voices:
        print(f" {'*' if v.name == cfg.voice else ' '} {v.name:<18} {v.status()}")
    print(f" {'*' if cfg.voice == 'auto' else ' '} {'auto':<18} OmniVoice's built-in voice")
    if not voices:
        print(f"\nNo voices yet. Create {cfg.voices_dir / 'myvoice'} with a ref.wav inside, then run --build-voice.")


def build_voices(cfg: Config, name: str | None) -> None:
    import soundfile as sf

    from core.tts import build_voice, load_model, synthesize
    from core.voices import get_voice, list_voices as profiles

    targets = []
    for v in [get_voice(cfg, name)] if name else profiles(cfg):
        if v.ref_audio is None:
            print(f"  {v.name}: skipped, no ref.wav")
        else:
            targets.append(v)
    if not targets:
        print(f"Nothing to build. Put a ref.wav in {cfg.voices_dir / '<name>'} first.")
        return
    model, transcriber = load_model(cfg), None
    for v in targets:
        if not v.ref_text and transcriber is None:
            from core.stt import Transcriber
            transcriber = Transcriber(cfg)
        prompt = build_voice(model, v, transcriber)
        sf.write(v.sample_file, synthesize(model, prompt, cfg.voice_sample_text), cfg.output_rate)
        print(f'\n  {v.name}: built\n    words:  "{v.ref_text}"\n    listen: {v.sample_file}')
    print("\nIf the words are wrong, fix ref.txt and build again. Pick a voice with JARVIS_VOICE in .env.")


def main() -> None:
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. voice assistant")
    parser.add_argument("--demo", action="store_true", help="show the orb only, no models")
    parser.add_argument("--voice", metavar="NAME", help="use this voice instead of JARVIS_VOICE from .env")
    parser.add_argument("--list-voices", action="store_true", help="show every voice in voices/ and its status")
    parser.add_argument("--build-voice", nargs="?", const="*", metavar="NAME",
                        help="build one voice (or all of them) and save a sample.wav to listen to")
    parser.add_argument("--rebuild-voice", action="store_true", help="rebuild the selected voice on startup")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    cfg, state = Config.load(), State()
    if args.voice:
        cfg.voice = args.voice

    if args.list_voices:
        list_voices(cfg)
        return
    if args.build_voice:
        build_voices(cfg, None if args.build_voice == "*" else args.build_voice)
        return

    from core.orb import OrbUI

    if args.demo:
        threading.Thread(target=demo, args=(state,), daemon=True).start()
        OrbUI(cfg, state).run()
        return

    from core.assistant import Assistant
    assistant = Assistant(cfg, state, rebuild_voice=args.rebuild_voice)
    threading.Thread(target=assistant.run, name="assistant", daemon=True).start()
    try:
        OrbUI(cfg, state, on_talk=assistant.request_talk, on_interrupt=assistant.interrupt).run()
    finally:
        assistant.shutdown()


if __name__ == "__main__":
    main()
