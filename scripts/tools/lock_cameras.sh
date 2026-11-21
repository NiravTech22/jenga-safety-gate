#!/usr/bin/env bash
# Lock camera image settings before every recording session (they reset on replug/reboot).
#   ./scripts/tools/lock_cameras.sh                     # overhead /dev/video2, wrist /dev/video0
#   ./scripts/tools/lock_cameras.sh /dev/video2 /dev/video0 [wrist_exposure]
# Values from Nirav's cameras, 2026-11-21. Use the same lighting every session:
# the overhead camera has no exposure control, so its auto-exposure follows the light.
set -euo pipefail
OVERHEAD=${1:-/dev/video2}
WRIST=${2:-/dev/video0}
WRIST_EXPOSURE=${3:-166}     # 100 us units: 166 = 16.6 ms, under the 33 ms of one 30 fps frame

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
