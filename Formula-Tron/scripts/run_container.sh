#!/bin/bash
# set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
if [ "$(basename "$PARENT_DIR")" = "src" ]; then
    WS_ROOT="$(dirname "$PARENT_DIR")"
else
    WS_ROOT="$SCRIPT_DIR"
fi
cd "$WS_ROOT"

# Jetson: preload TLS-heavy libs to avoid "cannot allocate memory in static TLS block"
# (PyQt5/OpenGL).
if [ "$(uname -m)" = "aarch64" ]; then
    for lib in \
        /usr/lib/aarch64-linux-gnu/libgomp.so.1 \
        /lib/aarch64-linux-gnu/libgomp.so.1 \
        /usr/lib/aarch64-linux-gnu/libGLdispatch.so.0 \
        /lib/aarch64-linux-gnu/libGLdispatch.so.0; do
        if [ -f "$lib" ]; then
            case ":${LD_PRELOAD}:" in
                *":$lib:"*) ;;
                *) export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$lib" ;;
            esac
        fi
    done
fi

python3 "$SCRIPT_DIR/fix_line_endings.py"
chmod +x "$SCRIPT_DIR"/*.sh
source /opt/ros/foxy/setup.bash
[ -f "/home/f1tenth1/f1tenth_ws/install/setup.bash" ] && source /home/f1tenth1/f1tenth_ws/install/setup.bash

# Ensure user-installed Python packages are visible
# to ROS-launched nodes in this shell.
PY_USER_SITE=$(python3 -c "import site; print(site.getusersitepackages())" 2>/dev/null)
if [ -n "$PY_USER_SITE" ] && [ -d "$PY_USER_SITE" ]; then
    export PYTHONPATH="$PY_USER_SITE${PYTHONPATH:+:$PYTHONPATH}"
fi

# Run preflight check (auto-installs missing dependencies)
# --- AUTOMATIC PREFLIGHT CHECK (Comment out the block below to disable) ---
PREFLIGHT="$(dirname $SCRIPT_DIR)/preflight_check.sh"
if [ -f "$PREFLIGHT" ]; then
    bash "$PREFLIGHT"
    if [ $? -ne 0 ]; then
        echo "Preflight failed - could not auto-fix all issues. See above."
        exit 1
    fi
fi
# -------------------------------------------------------------------------

# Keep runtime clean: stale source cv_bridge overlays can shadow apt packages
# and break realsense2_camera startup. Can be bypassed for developers.
if [ "${FORMULA_TRON_KEEP_SOURCE_CV_BRIDGE:-0}" != "1" ]; then
    rm -rf "$WS_ROOT/build/cv_bridge" "$WS_ROOT/install/cv_bridge"
fi

rm -rf build/formula_tron install/formula_tron
colcon build --packages-select formula_tron
source "$WS_ROOT/install/setup.bash"
ros2 node list 2>/dev/null | grep -q "vesc" || { ros2 launch formula_tron drivers.launch.py > /tmp/drivers.log 2>&1 & sleep 3; }
ros2 topic list 2>/dev/null | grep -qE "(/camera/camera/color/image_raw|/camera/color/image_raw)" || { pkill -f realsense2_camera 2>/dev/null; sleep 1; ros2 launch realsense2_camera rs_launch.py > /tmp/camera.log 2>&1 & sleep 5; }
ros2 launch formula_tron bringup.launch.py
