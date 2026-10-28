#!/bin/sh
# Robotun "sistemi kapat" komutuyla Pi'yi kapatabilmesi icin: yalnizca poweroff ve reboot,
# sifresiz. Bir kez calistir: sudo sh deploy/allow-poweroff.sh
set -e
USER_NAME="${SUDO_USER:-$(id -un)}"
FILE=/etc/sudoers.d/neurovision-poweroff
printf '%s ALL=(root) NOPASSWD: /usr/bin/systemctl poweroff, /usr/bin/systemctl reboot\n' "$USER_NAME" > "$FILE.tmp"
chmod 440 "$FILE.tmp"
visudo -cf "$FILE.tmp"
mv "$FILE.tmp" "$FILE"
echo "tamam: $FILE"
