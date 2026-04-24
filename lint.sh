#!/bin/bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src"
echo "PYTHONPATH: $PYTHONPATH" >&2

if [ -x ".venv/bin/python" ] && .venv/bin/python -m flake8 --version >/dev/null 2>&1; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

"$PYTHON_BIN" -c "import sys; print('sys.path:', sys.path[:3])" >&2
"$PYTHON_BIN" -m flake8 . --config=.flake8 --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=.git,.venv,htmrl_env,.pytest_cache,notebooks,reports -v
"$PYTHON_BIN" -m flake8 . --config=.flake8 --count --show-source --max-complexity=10 --statistics --exclude=.git,.venv,htmrl_env,.pytest_cache,notebooks,reports
