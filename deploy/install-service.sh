#!/bin/bash
# NeuroVision local (German) as a user service; starts at boot without a login. Only one robot
# program can own the microphone: the English and Gemini services are stopped (Conflicts=) and
# taken out of autostart.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p ~/.config/systemd/user
cp deploy/neurovision-local-deutsch.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user disable neurovision-local neurovision 2>/dev/null || true
systemctl --user enable --now neurovision-local-deutsch

# Oturum acilmadan baslamasi icin "linger" gerekir
if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
    loginctl enable-linger "$USER" 2>/dev/null \
        || echo "Acilista baslamasi icin bir kez calistir: sudo loginctl enable-linger $USER"
fi
systemctl --user --no-pager status neurovision-local-deutsch | head -5
