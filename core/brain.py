from __future__ import annotations

import json
import logging
import queue
import re
import shutil
import subprocess
import threading
import time
from typing import Callable

log = logging.getLogger("jarvis.brain")

SENTENCE_END = re.compile(r"(?<=[.!?…:;])\s+|\n+")


class SentenceSplitter:
    def __init__(self, emit: Callable[[str], None]) -> None:
        self.emit = emit
        self.buffer = ""

    def feed(self, text: str) -> None:
        self.buffer += text
        parts = SENTENCE_END.split(self.buffer)
        for sentence in parts[:-1]:
            if sentence.strip():
                self.emit(sentence.strip())
        self.buffer = parts[-1]

    def flush(self) -> None:
        if self.buffer.strip():
            self.emit(self.buffer.strip())
        self.buffer = ""


class ClaudeBrain:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.exe = shutil.which("claude")
        if not self.exe:
            raise RuntimeError("Claude Code ('claude') was not found on PATH")
        self.proc: subprocess.Popen | None = None
        self.session_id: str | None = None
        self._lines: queue.Queue[dict | None] = queue.Queue()
        self._busy = False
        self._cancelled = False
        self._write_lock = threading.Lock()

    def start(self) -> None:
        cfg = self.cfg
        cfg.claude_workdir.mkdir(parents=True, exist_ok=True)
        cmd = [self.exe, "-p",
               "--input-format", "stream-json", "--output-format", "stream-json",
               "--verbose", "--include-partial-messages",
               "--model", cfg.claude_model,
               "--append-system-prompt", cfg.persona]
        if cfg.claude_tools:
            cmd += ["--allowedTools", ",".join(cfg.claude_tools)]
        if self.session_id:
            cmd += ["--resume", self.session_id]
        self._lines = queue.Queue()
        self.proc = subprocess.Popen(
            cmd, cwd=cfg.claude_workdir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        threading.Thread(target=self._read_stdout, args=(self.proc, self._lines), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(self.proc,), daemon=True).start()
        log.info("Claude Code started (model=%s)", cfg.claude_model)

    @staticmethod
    def _read_stdout(proc: subprocess.Popen, lines: queue.Queue) -> None:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                lines.put(json.loads(line))
            except json.JSONDecodeError:
                log.debug("non-JSON from claude: %s", line[:200])
        lines.put(None)

    @staticmethod
    def _read_stderr(proc: subprocess.Popen) -> None:
        for line in proc.stderr:
            if line.strip():
                log.warning("claude: %s", line.strip())

    def _alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def close(self) -> None:
        if self._alive():
            self.proc.terminate()

    def _send(self, payload: dict) -> None:
        with self._write_lock:
            self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()

    def cancel(self) -> None:
        if not self._busy or self._cancelled or not self._alive():
            return
        self._cancelled = True
        try:
            self._send({"type": "control_request", "request_id": f"interrupt-{time.monotonic_ns()}",
                        "request": {"subtype": "interrupt"}})
            log.info("Interrupted Claude")
        except OSError:
            self.proc.kill()

    def ask(self, text: str, on_sentence: Callable[[str], None]) -> str:
        if not self._alive():
            self.start()
        self._cancelled = False
        self._busy = True
        try:
            self._send({"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": text}]}})
            return self._read_reply(on_sentence)
        finally:
            self._busy = False

    def _read_reply(self, on_sentence: Callable[[str], None]) -> str:
        splitter = SentenceSplitter(on_sentence)
        full: list[str] = []
        streamed = False
        while True:
            try:
                msg = self._lines.get(timeout=self.cfg.claude_timeout)
            except queue.Empty:
                self.proc.kill()
                raise TimeoutError("Claude took too long to answer")
            if msg is None:
                if self._cancelled:
                    return ""
                raise RuntimeError("Claude Code exited unexpectedly")
            if self._cancelled and msg.get("type") != "result":
                continue
            self.session_id = msg.get("session_id", self.session_id)
            kind = msg.get("type")

            if kind == "stream_event":
                event = msg.get("event", {})
                if event.get("type") == "content_block_delta" and event["delta"].get("type") == "text_delta":
                    streamed = True
                    full.append(event["delta"]["text"])
                    splitter.feed(event["delta"]["text"])
                elif event.get("type") == "content_block_stop":
                    splitter.flush()
            elif kind == "assistant" and not streamed:
                for block in msg.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        full.append(block["text"])
                        splitter.feed(block["text"] + "\n")
            elif kind == "result":
                if self._cancelled:
                    return ""
                splitter.flush()
                if msg.get("is_error"):
                    raise RuntimeError(msg.get("result") or "Claude returned an error")
                return msg.get("result") or "".join(full)
