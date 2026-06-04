#!/bin/bash

# Exit on any error
set -e

# Must run as root to write to /etc/systemd/system/
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (using sudo)"
  exit 1
fi

# Detect absolute workspace path
WORKSPACE_DIR=$(pwd)
# We need the user who ran sudo
ACTUAL_USER=${SUDO_USER:-$USER}
# Detect uv python path
PYTHON_PATH="$WORKSPACE_DIR/.venv/bin/python"

if [ ! -f "$PYTHON_PATH" ]; then
  echo "Virtualenv not found at $WORKSPACE_DIR/.venv. Please run 'uv sync' or initialize environment first."
  exit 1
fi

# Check if main.py exists
if [ ! -f "$WORKSPACE_DIR/main.py" ]; then
  echo "main.py not found in current directory. Please run this script from the project root."
  exit 1
fi

SERVICE_FILE="/etc/systemd/system/melotts.service"

echo "Creating systemd service file at $SERVICE_FILE..."
cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=MeloTTS API & Streaming Service
After=network.target

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$WORKSPACE_DIR
ExecStart=$PYTHON_PATH main.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=HF_ENDPOINT=https://hf-mirror.com
# Environment=TTS_OUTPUT_DIR=$WORKSPACE_DIR/outputs # Uncomment to set a global output dir

[Install]
WantedBy=multi-user.target
EOF

echo "Reloading systemd daemon..."
systemctl daemon-reload

echo "Enabling melotts.service..."
systemctl enable melotts.service

echo "Starting melotts.service..."
systemctl start melotts.service

echo "Checking melotts.service status..."
systemctl status melotts.service --no-pager -l

echo "=================================================="
echo "MeloTTS service successfully installed!"
echo "API is running at http://localhost:8100"
echo "You can check logs using: journalctl -u melotts.service -f"
echo "=================================================="
