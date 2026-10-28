#!/bin/sh
# Wi-Fi kurulum servisini kur: sudo sh deploy/wifi/install-wifi.sh
set -e
dir="$(dirname "$0")"
mkdir -p /etc/NetworkManager/dnsmasq-shared.d
cp "$dir/captive-dns.conf" /etc/NetworkManager/dnsmasq-shared.d/neurovision-captive.conf
cp "$dir/neurovision-wifi.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable neurovision-wifi.service
echo "tamam: bir sonraki acilista calisir (kayitli aga baglanirsa hicbir sey yapmaz)"
