#!/bin/bash
# NeuroVision'i kullanici servisi olarak kurar; Pi acilinca oturum acmadan baslar.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p ~/.config/systemd/user
cp deploy/neurovision.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now neurovision

# Oturum acilmadan baslamasi icin "linger" gerekir
if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
    loginctl enable-linger "$USER" 2>/dev/null \
        || echo "Acilista baslamasi icin bir kez calistir: sudo loginctl enable-linger $USER"
fi
systemctl --user --no-pager status neurovision | head -5
