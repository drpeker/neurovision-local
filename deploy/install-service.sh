#!/bin/bash
# NeuroVision local as a user service; starts at boot without a login. Stops the Gemini one (Conflicts=).
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p ~/.config/systemd/user
cp deploy/neurovision-local.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now neurovision-local

# Oturum acilmadan baslamasi icin "linger" gerekir
if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
    loginctl enable-linger "$USER" 2>/dev/null \
        || echo "Acilista baslamasi icin bir kez calistir: sudo loginctl enable-linger $USER"
fi
systemctl --user --no-pager status neurovision-local | head -5
