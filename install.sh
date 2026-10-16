#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
export HF_HOME="$ROOT/models"

TORCH_VERSION="2.8.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
WHISPER_MODEL="small"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) OS=windows; VENV_PY=".venv/Scripts/python.exe" ;;
  Darwin)               OS=macos;   VENV_PY=".venv/bin/python" ;;
  *)                    OS=linux;   VENV_PY=".venv/bin/python" ;;
esac

WIN_PY="${LOCALAPPDATA:-}/Programs/Python/Python312/python.exe"

step "Finding Python 3.12"
if [ -x "$VENV_PY" ]; then
  PY=("$VENV_PY")
elif [ "$OS" = windows ] && command -v py >/dev/null 2>&1 && py -3.12 --version >/dev/null 2>&1; then
  PY=(py -3.12)
elif [ "$OS" = windows ] && [ -x "$WIN_PY" ]; then
  PY=("$WIN_PY")
elif command -v python3.12 >/dev/null 2>&1; then
  PY=(python3.12)
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
  PY=(python3)
else
  case "$OS" in
    windows) fail "Python 3.12 not found. Install it with: winget install Python.Python.3.12" ;;
    macos)   fail "Python 3.12 not found. Install it with: brew install python@3.12" ;;
    linux)   fail "Python 3.12 not found. Install python3.12 and python3.12-venv with your package manager." ;;
  esac
fi
echo "Using: $("${PY[@]}" --version)"

if [ "$OS" = linux ] && ! { ldconfig -p 2>/dev/null | grep -q libportaudio; }; then
  echo "Note: PyAudio needs PortAudio. On Ubuntu/Debian: sudo apt install portaudio19-dev"
fi

step "Creating the virtual environment (.venv)"
[ -x "$VENV_PY" ] || "${PY[@]}" -m venv .venv
"$VENV_PY" -m pip install --upgrade pip --quiet

step "Installing PyTorch $TORCH_VERSION"
if "$VENV_PY" -c "import torch, sys; sys.exit(not torch.__version__.startswith('$TORCH_VERSION'))" 2>/dev/null; then
  echo "Already installed."
elif [ "$OS" = macos ]; then
  "$VENV_PY" -m pip install "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION"
else
  "$VENV_PY" -m pip install "torch==$TORCH_VERSION+cu128" "torchaudio==$TORCH_VERSION+cu128" --extra-index-url "$TORCH_INDEX"
fi

step "Installing Jarvis's packages"
"$VENV_PY" -m pip install -r requirements.txt

step "Checking the GPU"
"$VENV_PY" -c "import torch; print('CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no NVIDIA GPU found, things will be slow')"

step "Downloading models into $HF_HOME (about 4 GB the first time)"
"$VENV_PY" -c "import openwakeword; openwakeword.utils.download_models()"
"$VENV_PY" -c "from huggingface_hub import snapshot_download as d; d('Systran/faster-whisper-$WHISPER_MODEL'); d('k2-fsa/OmniVoice'); print('Models ready.')"

step "Checking Claude Code"
if command -v claude >/dev/null 2>&1; then
  echo "Found: $(claude --version 2>/dev/null || echo claude)"
else
  echo "Claude Code not found. Install it from https://claude.com/claude-code and run 'claude' once to log in."
fi

mkdir -p workspace voices
[ -f .env ] || cp .env.example .env

step "Done"
cat <<EOF
Next steps:
  1. Record 3-10 seconds of the voice you want and save it as voices/jarvis/ref.wav
  2. Build it:          $VENV_PY jarvis.py --build-voice jarvis
     (the words are transcribed into voices/jarvis/ref.txt; fix them if wrong and build again)
  3. Listen to:         voices/jarvis/sample.wav
  4. Run Jarvis:        $VENV_PY jarvis.py
     (on Windows you can also double-click start_jarvis.bat)
  Choose another voice any time with JARVIS_VOICE in .env
EOF
