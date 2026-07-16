#!/bin/bash
cd "$(dirname "$0")"
if [ -d .venv ]; then source .venv/bin/activate; fi
python3 -m pip install -q -r requirements.txt
python3 replay_predict.py --all || exit 1
python3 web_app.py
