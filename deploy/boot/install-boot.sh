#!/bin/sh
# Acilis anonsunu sistem servisi olarak kur: sudo sh deploy/boot/install-boot.sh
set -e
cp "$(dirname "$0")/neurovision-boot.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable neurovision-boot.service
echo "tamam: bir sonraki acilista calisir"
