#!/bin/bash
export PYTHONPATH=$(pwd)/src
echo "PYTHONPATH: $PYTHONPATH" >&2
.venv/bin/python -c "import sys; print('sys.path:', sys.path[:3])" >&2
.venv/bin/python -c "from psu_capstone.encoder_layer.rdse import RDSEParameters; print('Import works')" >&2
.venv/bin/python -m flake8 . --config=.flake8 --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=.git,.venv,htmrl_env,.pytest_cache,notebooks,reports -v
.venv/bin/python -m flake8 . --config=.flake8 --count --show-source --max-complexity=10 --statistics --exclude=.git,.venv,htmrl_env,.pytest_cache,notebooks,reports
