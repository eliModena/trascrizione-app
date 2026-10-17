#!/bin/bash
cd "$(dirname "$0")"
echo "Avvio di BE Video Intercom Transcriptor..."
pip3 install streamlit faster-whisper sounddevice numpy pillow > /dev/null 2>&1
streamlit run app.py
