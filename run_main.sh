#!/bin/bash
# Run this if it doesn't work: chmod +x run_main.sh
# This makes this file an executable file
cd "$(dirname "$0")"
source ./venv/bin/activate
python main.py
