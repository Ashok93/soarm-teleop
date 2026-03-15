#!/usr/bin/env bash
set -euo pipefail

if [[ "${ENABLE_VNC:-0}" == "1" ]]; then
  export DISPLAY="${DISPLAY:-:1}"
  Xvfb "${DISPLAY}" -screen 0 "${VNC_RESOLUTION:-1920x1080x24}" &
  x11vnc -display "${DISPLAY}" -forever -shared -rfbport "${VNC_PORT:-5900}" -nopw &
fi

exec "$@"
