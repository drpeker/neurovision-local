#!/bin/bash
# A Pi 4 as the NeuroVision robot, with the Zero 2W's hardware and pins: one time, with sudo.
#   sudo bash deploy/setup-pi4.sh
# Gives the Pi 4 the same system setup as the Zero (checked on the Zero on 2026-10-28):
# I2S sound card, SPI screen, camera, poweroff permission, boot chime/screen, Wi-Fi portal, linger.
# Before: ~/.asoundrc = deploy/asoundrc and the user service (bash deploy/install-service.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }

bash deploy/setup-zero.sh

# The Zero's config.txt also has dtparam=i2s=on (googlevoicehat-soundcard enables I2S itself)
CFG=/boot/firmware/config.txt
grep -q "^dtparam=i2s=on" "$CFG" || sed -i "s/^#dtparam=i2s=on/dtparam=i2s=on/" "$CFG"
grep -q "^dtparam=i2s=on" "$CFG" || echo "dtparam=i2s=on" >> "$CFG"

echo "== Wi-Fi setup portal (does nothing while a known network is in range)"
sh deploy/wifi/install-wifi.sh

echo
echo "Pi 4 ready as the robot. Now: sudo reboot"
