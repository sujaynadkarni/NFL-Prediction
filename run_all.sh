#!/bin/zsh
# Weekly update: refresh current-season data → retrain → simulate → rebuild page
cd "$(dirname "$0")"
.venv/bin/python fetch_data.py --current && \
.venv/bin/python nfl_model.py && \
.venv/bin/python simulate.py && \
.venv/bin/python generate_html.py
