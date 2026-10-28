#!/bin/bash
# NeuroVision donanim ayari: I2S ses karti (INMP441 mikrofon + MAX98357A amfi), SPI ekran, CSI kamera.
# Kullanim: sudo bash deploy/hw-setup.sh [kamera]   kamera: ov5647 | imx219 | imx708 | auto (varsayilan)
#
# googlevoicehat-soundcard: mikrofon ve amfi ayni I2S hattinda tek kart (BCLK GPIO18, LRCLK GPIO19,
# mikrofon GPIO20, amfi GPIO21, amfi SD GPIO16). Analog jak kapanir; GPIO12/13 donanim PWM'i
# motorlara kalir.
set -euo pipefail
CFG=/boot/firmware/config.txt
CAM=${1:-auto}
[ "$(id -u)" -eq 0 ] || { echo "sudo ile calistir"; exit 1; }

cp -a "$CFG" "$CFG.before-nv-$(date +%Y%m%d-%H%M%S)"

# Analog jak kapali: ses I2S amfiden, PWM kanallari motorlara
sed -i "s/^dtparam=audio=on/dtparam=audio=off/" "$CFG"

# Kamera: model belliyse otomatik algilamayi kapatip overlay ile zorla
sed -i "/^dtoverlay=\(ov5647\|imx219\|imx708\|imx477\)/d" "$CFG"
if [ "$CAM" = auto ]; then
    sed -i "s/^camera_auto_detect=0/camera_auto_detect=1/" "$CFG"
else
    sed -i "s/^camera_auto_detect=1/camera_auto_detect=0/" "$CFG"
fi

# [all] bolumune NeuroVision blogu (tekrar calistirilinca yenilenir)
sed -i "/^# >>> neurovision-hw/,/^# <<< neurovision-hw/d" "$CFG"
{
    echo "# >>> neurovision-hw"
    echo "[all]"
    echo "dtparam=spi=on"
    echo "dtoverlay=googlevoicehat-soundcard"
    [ "$CAM" = auto ] || echo "dtoverlay=$CAM"
    echo "# <<< neurovision-hw"
} >> "$CFG"

echo "---- yeni config.txt (yorumsuz) ----"
grep -vE "^\s*#|^\s*$" "$CFG"
echo "------------------------------------"
echo "Tamam. Simdi: sudo reboot"
