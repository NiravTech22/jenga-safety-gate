#!/usr/bin/env bash
# Lock camera image settings before every recording session (they reset on replug/reboot).
#   ./scripts/tools/lock_cameras.sh                     # each camera by its stable by-id name
#   ./scripts/tools/lock_cameras.sh "" "" 330           # same, wrist exposure 330
#   ./scripts/tools/lock_cameras.sh /dev/video2 /dev/video0 330   # explicit devices
# /dev/videoN numbers swap on replug; the by-id links follow the physical camera.
# Values from Nirav's cameras, 2026-11-21. Use the same lighting every session:
# the overhead camera has no exposure control, so its auto-exposure follows the light.
set -euo pipefail
OVERHEAD=${1:-/dev/v4l/by-id/usb-Jieli_Technology_USB_PHY_2.0-video-index0}
WRIST=${2:-/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0}
WRIST_EXPOSURE=${3:-330}     # 100 us units: 330 = 33 ms (166 was too dark in Nirav's room)
for dev in "$OVERHEAD" "$WRIST"; do
  [ -e "$dev" ] || { echo "camera $dev not found (unplugged?); see: ls -l /dev/v4l/by-id/" >&2; exit 1; }
done

# Overhead: only white balance is adjustable (0-255 scale); 51 gave correct colour.
v4l2-ctl -d "$OVERHEAD" -c white_balance_temperature=51

# Wrist: manual white balance (paper looked white at 4130 K), manual exposure, and no
# frame-rate drop in dim light (exposure_dynamic_framerate=1 gave 19.9 fps instead of 30).
v4l2-ctl -d "$WRIST" -c white_balance_automatic=0
v4l2-ctl -d "$WRIST" -c white_balance_temperature=4130
v4l2-ctl -d "$WRIST" -c auto_exposure=1
v4l2-ctl -d "$WRIST" -c exposure_time_absolute="$WRIST_EXPOSURE"
v4l2-ctl -d "$WRIST" -c exposure_dynamic_framerate=0

echo "overhead $OVERHEAD:"; v4l2-ctl -d "$OVERHEAD" -C white_balance_temperature
echo "wrist $WRIST:"
v4l2-ctl -d "$WRIST" -C white_balance_automatic -C white_balance_temperature \
         -C auto_exposure -C exposure_time_absolute -C exposure_dynamic_framerate
