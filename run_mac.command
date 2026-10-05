#!/bin/bash
# Double-click in Finder to launch the Hansmeyer subdivision explorer.
# First run creates .venv and installs requirements; later runs start immediately.
cd "$(dirname "$0")" || exit 1

find_python() {
  for c in python3.14 python3.13 python3.12 python3.11 python3.10 python3 \
           /opt/homebrew/bin/python3 /usr/local/bin/python3; do
    if command -v "$c" >/dev/null 2>&1 && \
       "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}

if [ ! -x .venv/bin/python ]; then
  PY=$(find_python) || {
    echo "Python 3.10+ not found. Install it from https://www.python.org/downloads/ (or: brew install python) and run this again."
    read -r -p "Press Enter to close..."; exit 1; }
  echo "Creating virtual environment with $PY ..."
  "$PY" -m venv .venv || { read -r -p "venv creation failed. Press Enter..."; exit 1; }
fi

if ! cmp -s requirements.txt .venv/requirements.stamp; then
  echo "Installing dependencies (first run only)..."
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r requirements.txt || { read -r -p "Install failed. Press Enter..."; exit 1; }
  cp requirements.txt .venv/requirements.stamp
fi

.venv/bin/python app.py || read -r -p "The app exited with an error (see above). Press Enter to close..."
