from __future__ import annotations

import logging
import queue
import threading
import time

from .audio import Microphone, Speaker
from .brain import ClaudeBrain
from .state import Mode, State

log = logging.getLogger("jarvis")


class Assistant:
    def __init__(self, cfg, state: State, rebuild_voice: bool = False) -> None:
        self.cfg, self.state = cfg, state
        self.rebuild_voice = rebuild_voice
        self._talk = threading.Event()
        self._responding = False
        self._cancelled = False
        self._barged = False
        self.mic = self.speaker = self.wake = self.stt = self.brain = self.mouth = None

    def request_talk(self) -> None:
        if self._responding:
            self._stop_response(listen_after=True)
        else:
            self._talk.set()

    def interrupt(self) -> None:
        self._stop_response(listen_after=False)

    def _stop_response(self, listen_after: bool) -> None:
        if listen_after:
            self._barged = True
        self._cancelled = True
        if self.mouth:
            self.mouth.stop()
        if self.brain:
            self.brain.cancel()

    def run(self) -> None:
        try:
            self._boot()
        except Exception as exc:
            log.exception("Boot failed")
            self.state.update(mode=Mode.ERROR, status=f"Boot failed: {exc}")
            return
        if self.cfg.greeting:
            self.mouth.say(self.cfg.greeting)
            self.mouth.wait()
        while self.state.running:
            try:
                if self._wait_for_wake():
                    self._conversation()
            except Exception as exc:
                log.exception("Turn failed")
                self.state.update(mode=Mode.ERROR, status=str(exc)[:120])
                self.mouth.say("Apologies, sir. Something went wrong on my end.")
                self.mouth.wait()

    def _boot(self) -> None:
        cfg, st = self.cfg, self.state
        st.update(status="Connecting to Claude")
        self.brain = ClaudeBrain(cfg)
        self.brain.start()

        st.update(status="Calibrating audio interfaces")
        self.speaker = Speaker(cfg, on_level=lambda v: setattr(st, "out_level", v))
        self.mic = Microphone(cfg, on_level=lambda v: setattr(st, "mic_level", v))

        st.update(status="Loading wake word engine")
        from .wake import WakeWord
        self.wake = WakeWord(cfg)

        st.update(status="Loading speech recognition")
        from .stt import Transcriber
        self.stt = Transcriber(cfg)

        st.update(status=f'Loading voice "{cfg.voice}"')
        from .tts import Mouth, Voice
        self.mouth = Mouth(Voice(cfg, transcriber=self.stt, rebuild=self.rebuild_voice), self.speaker, st)
        log.info("All systems online")

    def shutdown(self) -> None:
        self.state.running = False
        for part in (self.brain, self.mic, self.speaker):
            if part:
                part.close()

    def _wait_for_wake(self) -> bool:
        self.state.update(mode=Mode.IDLE, status='Say "Hey Jarvis" or press SPACE')
        self.mic.flush()
        self.wake.reset()
        self._talk.clear()
        while self.state.running:
            if self._talk.is_set():
                return True
            try:
                chunk = self.mic.read(timeout=0.2)
            except queue.Empty:
                continue
            if self.wake.detected(chunk):
                log.info("Wake word detected")
                self.wake.reset()
                return True
        return False

    def _conversation(self) -> None:
        timeout, chime = self.cfg.no_speech_timeout, True
        while self.state.running:
            heard = self._listen(timeout, chime)
            if not heard:
                break
            self._respond(heard)
            if self._barged:
                timeout, chime = self.cfg.no_speech_timeout, True
                continue
            if self._cancelled or self.cfg.follow_up_seconds <= 0:
                break
            timeout, chime = self.cfg.follow_up_seconds, False

    def _listen(self, timeout: float, chime: bool) -> str | None:
        st = self.state
        st.update(mode=Mode.LISTENING, status="Listening" if chime else "Listening for a follow-up")
        ignore = 0.0
        self.mic.flush()
        if chime and self.cfg.chimes:
            tone = self.speaker.chime(rising=True)
            threading.Thread(target=self.speaker.play, args=(tone, self.speaker.rate), daemon=True).start()
            ignore = len(tone) / self.speaker.rate + 0.1

        audio = self.mic.record(timeout, should_stop=lambda: not st.running, ignore_seconds=ignore)
        if audio is None:
            return None
        st.update(mode=Mode.THINKING, status="Transcribing")
        text, language = self.stt.transcribe(audio)
        log.info("Heard (%s): %s", language, text)
        if text:
            st.update(user_text=text, reply_text="")
        return text or None

    def _respond(self, text: str) -> None:
        st = self.state
        st.update(mode=Mode.THINKING, status="Thinking")
        self._cancelled = self._barged = False
        self._responding = True
        spoken: list[str] = []

        def on_sentence(sentence: str) -> None:
            if self._cancelled:
                return
            spoken.append(sentence)
            st.update(reply_text=" ".join(spoken))
            self.mouth.say(sentence)

        done = threading.Event()
        watcher = None
        if self.cfg.barge_in:
            watcher = threading.Thread(target=self._watch_for_wake, args=(done,), daemon=True)
            watcher.start()
        try:
            reply = self.brain.ask(text, on_sentence)
            log.info("Reply: %s", reply if not self._cancelled else "(interrupted)")
        except Exception as exc:
            log.exception("Brain error")
            if not self._cancelled:
                self.mouth.say("I'm afraid I can't reach my brain right now, sir.")
                st.update(mode=Mode.ERROR, status=f"Claude error: {exc}"[:120])
        try:
            while self.mouth.busy and not self._cancelled and st.running:
                time.sleep(0.03)
        finally:
            done.set()
            if watcher:
                watcher.join(timeout=1)
            self._responding = False
        if self._cancelled:
            st.update(reply_text=" ".join(spoken) + " …", status="Interrupted")

    def _watch_for_wake(self, done: threading.Event) -> None:
        self.mic.flush()
        self.wake.reset()
        while not done.is_set() and self.state.running:
            try:
                chunk = self.mic.read(timeout=0.1)
            except queue.Empty:
                continue
            threshold = self.cfg.barge_in_threshold if self.mouth.busy else self.cfg.wake_threshold
            if self.wake.detected(chunk, threshold):
                log.info("Wake word during answer: interrupting")
                self.wake.reset()
                self._stop_response(listen_after=True)
                return
