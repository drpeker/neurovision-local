#!/bin/bash
# Yeni bir Pi'ye (Zero 2W dahil) NeuroVision sistem ayarlari: tek seferlik, sudo ile.
#   sudo bash deploy/setup-zero.sh
# Proje, .venv, .env ve ~/.asoundrc kullanici olarak onceden kopyalanmis olmali.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$(id -u)" -eq 0 ] || { echo "sudo ile calistir"; exit 1; }
U="${SUDO_USER:?sudo ile calistir}"

echo "== 1/5 PortAudio (sounddevice icin)"
apt-get install -y libportaudio2

echo "== 2/5 Donanim: ses karti, SPI ekran, kamera (config.txt; yedek alinir)"
# Onceden elle eklenmis ses karti satiri varsa kaldir: hw-setup kendi blogunda ekliyor
sed -i '/^# >>> neurovision-hw/,/^# <<< neurovision-hw/!{/^dtoverlay=googlevoicehat-soundcard$/d}' /boot/firmware/config.txt
bash deploy/hw-setup.sh

echo "== 3/5 Sesle kapatma izni (yalnizca poweroff/reboot)"
SUDO_USER="$U" sh deploy/allow-poweroff.sh

echo "== 4/5 Acilis sesi ve ekrani"
sh deploy/boot/install-boot.sh

echo "== 5/5 Oturum acmadan kullanici servisleri (linger)"
loginctl enable-linger "$U"

echo
echo "Tamam. Simdi: sudo reboot"
echo "Donanim takilinca robotu baslat: bash deploy/install-service.sh"
