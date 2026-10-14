#!/bin/bash
# display from env vars, everything else hard-coded for this environment
if [ "$XDG_SESSION_TYPE" = wayland ] || [ -n "$WAYLAND_DISPLAY" ]; then d=wayland
elif [ "$XDG_SESSION_TYPE" = x11 ] || [ -n "$DISPLAY" ]; then d=x11
else d=headless; fi
echo "DISPLAY_SERVER=$d"
echo "INIT_SYSTEM=unknown"
echo "CONTAINER=docker"
echo "VIRTUALIZATION=kvm"
exit 0
