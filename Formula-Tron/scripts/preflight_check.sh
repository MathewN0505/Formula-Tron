#!/bin/bash
# Formula-Tron Pre-Flight Check
# Auto-detects AND auto-fixes missing dependencies.
# Run standalone or as part of your deployment process.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FORMULA_TRON_DIR="$SCRIPT_DIR/formula_tron"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
if [[ "$(basename "$PARENT_DIR")" == "src" ]]; then
    WS_ROOT="$(dirname "$PARENT_DIR")"
else
    WS_ROOT="$SCRIPT_DIR"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

FAILED=0
FIXED=0

echo ""
echo -e "${BOLD}â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—${NC}"
echo -e "${BOLD}â•‘       FORMULA-TRON PRE-FLIGHT CHECK              â•‘${NC}"
echo -e "${BOLD}â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•${NC}"
echo ""
echo -e "  SCRIPT_DIR:  $SCRIPT_DIR"
echo -e "  WS_ROOT:     $WS_ROOT"
echo ""

PY_USER_SITE=$(python3 -c "import site; print(site.getusersitepackages())" 2>/dev/null)
if [ -n "$PY_USER_SITE" ] && [ -d "$PY_USER_SITE" ]; then
    export PYTHONPATH="$PY_USER_SITE${PYTHONPATH:+:$PYTHONPATH}"
fi


chmod +x "$FORMULA_TRON_DIR"/*.sh 2>/dev/null

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. ROS 2 Environment
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo -e "${CYAN}[1/7] ROS 2 Environment${NC}"

echo -n "  ROS_DISTRO... "
if [ -z "$ROS_DISTRO" ]; then
    if [ -f /opt/ros/foxy/setup.bash ]; then
        echo -e "${YELLOW}not set, sourcing foxy${NC}"
        source /opt/ros/foxy/setup.bash
        FIXED=$((FIXED+1))
    else
        echo -e "${RED}FAIL${NC} (ROS 2 Foxy not installed at /opt/ros/foxy)"
        FAILED=1
    fi
else
    if [ "$ROS_DISTRO" = "foxy" ]; then
        echo -e "${GREEN}OK${NC} (Foxy)"
    else
        echo -e "${YELLOW}WARN${NC} (Expected foxy, got $ROS_DISTRO)"
    fi
fi

echo -n "  ros2 command... "
if command -v ros2 &> /dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC} (ros2 not in PATH)"
    FAILED=1
fi

echo -n "  colcon command... "
if command -v colcon &> /dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}MISSING - installing...${NC}"
    pip3 install --quiet colcon-common-extensions 2>/dev/null
    if command -v colcon &> /dev/null; then
        echo -e "  colcon ${GREEN}FIXED${NC}"
        FIXED=$((FIXED+1))
    else
        echo -e "  colcon ${RED}INSTALL FAILED${NC}"
        FAILED=1
    fi
fi

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. ROS Apt Packages (the ones apt autoremove kills)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo ""
echo -e "${CYAN}[2/7] ROS Apt Packages${NC}"

# Refresh ROS GPG key if expired (common failure after 2025)
echo -n "  ROS GPG key... "
if ! apt-get update -o Dir::Etc::sourcelist="/etc/apt/sources.list.d/ros2.list" \
     -o Dir::Etc::sourceparts="-" -o APT::Get::List-Cleanup="0" 2>&1 | grep -q "EXPKEYSIG\|NO_PUBKEY"; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}EXPIRED - refreshing...${NC}"
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo apt-key add - 2>/dev/null \
      || curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo tee /usr/share/keyrings/ros-archive-keyring.gpg >/dev/null
    echo -e "  ROS GPG key ${GREEN}FIXED${NC}"
    FIXED=$((FIXED+1))
fi

APT_NEEDED=""
ROS_PKGS=(
    # Core F1TENTH Dependencies
    ros-foxy-rclpy
    ros-foxy-std-msgs
    ros-foxy-sensor-msgs
    ros-foxy-ackermann-msgs
    ros-foxy-cv-bridge
    ros-foxy-joy
    ros-foxy-joy-teleop
    ros-foxy-tf2-ros
    ros-foxy-launch-ros
    ros-foxy-ros2launch
    ros-foxy-ament-index-python
    ros-foxy-image-transport
    ros-foxy-diagnostic-updater
    
    # Improved Camera & Streaming Support
    ros-foxy-image-transport-plugins
    ros-foxy-compressed-image-transport
    ros-foxy-realsense2-camera
    ros-foxy-realsense2-description
    
    # Improved VESC & Serial Support
    ros-foxy-serial-driver
    # ros-foxy-transport-drivers  # Not available in Foxy repos; build from source if needed
    ros-foxy-teleop-twist-joy
    
    # F1TENTH Standard Hardware (LIDAR + VESC C++ Driver)
    ros-foxy-urg-node
    ros-foxy-urg-c
    ros-foxy-laser-proc
    ros-foxy-vesc-driver
    ros-foxy-vesc-msgs
    # ros-foxy-ackermann-mux  # Not in Foxy repos; use f1tenth/ackermann_mux from source
    
    # Navigation & SLAM (Standard F1TENTH Stack)
    ros-foxy-slam-toolbox
    ros-foxy-navigation2
    ros-foxy-nav2-bringup
    ros-foxy-robot-localization
    # ros-foxy-openslam-gmapping  # Not ported to ROS 2 Foxy; use slam-toolbox instead
    
    # Math & Transforms
    ros-foxy-tf-transformations
    ros-foxy-angles

    # Debugging & Visualization Tools
    ros-foxy-rqt
    ros-foxy-rqt-common-plugins
    ros-foxy-rqt-image-view
    ros-foxy-rviz2
    
    # System Utilities (Essential for hardware interfacing)
    python3-serial
    python3-pip
    mysql-common
    libmysqlclient-dev
    build-essential
    usbutils
    i2c-tools
    wireless-tools
    net-tools
    
    # CRITICAL: System OpenCV (Prevents mismatch with cv_bridge)
    python3-opencv
    # CRITICAL: System Numpy/Scipy (Prevents ABI mismatch with ROS binaries)
    python3-numpy
    python3-scipy
)

for pkg in "${ROS_PKGS[@]}"; do
    echo -n "  $pkg... "
    # Check if installed (handles both ROS packages and system packages via dpkg)
    if dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}MISSING${NC}"
        APT_NEEDED="$APT_NEEDED $pkg"
    fi
done

if [ -n "$APT_NEEDED" ]; then
    echo ""
    echo -e "${YELLOW}  Installing missing ROS packages...${NC}"
    echo "  (Will ask for sudo password)"
    sudo apt-get update
    sudo apt-get install -y $APT_NEEDED
    # Verify
    for pkg in $APT_NEEDED; do
        if dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
            echo -e "  $pkg ${GREEN}FIXED${NC}"
            FIXED=$((FIXED+1))
        else
            echo -e "  $pkg ${RED}INSTALL FAILED${NC}"
            FAILED=1
        fi
    done
fi

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Python Packages (critical)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo ""
echo "----------------------------------------------------------------"
echo -e "${CYAN}[3/7] Checks: Conflicting OpenCV (The 'Mismatch' Fixer)${NC}"

# SANITIZER: Remove pip opencv if present, because it crashes ros-foxy-cv-bridge
if pip3 list 2>/dev/null | grep -qE "opencv-python|opencv-contrib-python"; then
    echo -e "${RED}  CONFLICT DETECTED: pip-installed OpenCV found.${NC}"
    echo "  This causes the 'mismatch' crash with ROS cv_bridge."
    echo "  Removing conflicting packages..."
    sudo pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
    echo -e "${GREEN}  Cleaned.${NC}"
else
    echo -e "${GREEN}  OK (No conflicting pip OpenCV found)${NC}"
fi

# SANITIZER: Remove stale source cv_bridge overlays that can shadow apt packages
# and break camera startup (especially after old source-build attempts).
echo -n "  cv_bridge overlay hygiene... "
if [ "${FORMULA_TRON_KEEP_SOURCE_CV_BRIDGE:-0}" = "1" ]; then
    echo -e "${YELLOW}SKIPPED${NC} (FORMULA_TRON_KEEP_SOURCE_CV_BRIDGE=1)"
else
    REMOVED_CV_BRIDGE_OVERLAY=0
    if [ -d "$WS_ROOT/build/cv_bridge" ]; then
        rm -rf "$WS_ROOT/build/cv_bridge"
        REMOVED_CV_BRIDGE_OVERLAY=1
    fi
    if [ -d "$WS_ROOT/install/cv_bridge" ]; then
        rm -rf "$WS_ROOT/install/cv_bridge"
        REMOVED_CV_BRIDGE_OVERLAY=1
    fi
    if [ "$REMOVED_CV_BRIDGE_OVERLAY" -eq 1 ]; then
        echo -e "${YELLOW}FIXED${NC} (removed stale cv_bridge overlay)"
        FIXED=$((FIXED+1))
    else
        echo -e "${GREEN}OK${NC}"
    fi
fi

# cv_bridge: use system apt package (import cv2 before cv_bridge on Jetson)
echo -n "  cv_bridge... "
if python3 -c "import cv2; from cv_bridge import CvBridge" 2>/dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC} (cv_bridge not working)"
    echo -e "  ${YELLOW}Try: sudo apt install ros-foxy-cv-bridge${NC}"
    FAILED=1
fi

echo ""
echo -e "${CYAN}[3/7] Python Packages (Critical)${NC}"

# GUARD: Pin numpy to safe range (<1.24 for system scipy on Focal)
echo -n "  numpy version guard... "
NUMPY_VER=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null)
if [ -n "$NUMPY_VER" ]; then
    NUMPY_MAJOR=$(echo "$NUMPY_VER" | cut -d. -f1)
    NUMPY_MINOR=$(echo "$NUMPY_VER" | cut -d. -f2)
    if [ "$NUMPY_MAJOR" -eq 1 ] && [ "$NUMPY_MINOR" -ge 20 ] && [ "$NUMPY_MINOR" -lt 24 ]; then
        echo -e "${GREEN}OK${NC} ($NUMPY_VER)"
    else
        echo -e "${YELLOW}$NUMPY_VER out of safe range â€” installing 1.23.5...${NC}"
        pip3 install --user numpy==1.23.5 2>/dev/null
        FIXED=$((FIXED+1))
    fi
else
    echo -e "${YELLOW}not found â€” installing 1.23.5...${NC}"
    pip3 install --user numpy==1.23.5 2>/dev/null
    FIXED=$((FIXED+1))
fi

# module_name:pip_package_name
CRITICAL_PY=(
    # Note: cv2 is now handled via apt (python3-opencv) above to avoid mismatch
    # Note: numpy/scipy are now handled via apt (python3-numpy/scipy)
    "PyQt5:PyQt5"
    "pyqtgraph:pyqtgraph"
    # Power User Libraries
    "numba:numba"
    "transformations:transformations"
    "websockets:websockets"
)

for entry in "${CRITICAL_PY[@]}"; do
    mod="${entry%%:*}"
    pkg="${entry##*:}"
    echo -n "  $mod... "
    if python3 -c "import $mod" 2>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}MISSING - installing $pkg...${NC}"
        pip3 install --quiet "$pkg" 2>/dev/null
        if python3 -c "import $mod" 2>/dev/null; then
            echo -e "  $mod ${GREEN}FIXED${NC}"
            FIXED=$((FIXED+1))
        else
            echo -e "  $mod ${RED}INSTALL FAILED${NC}"
            FAILED=1
        fi
    fi
done

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. Permissions & Files
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo ""
echo -e "${CYAN}[5/7] Permissions & Files${NC}"


# Fix all shell scripts
chmod +x "$FORMULA_TRON_DIR"/*.sh 2>/dev/null
chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6. Python Import Smoke Test
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo ""
echo -e "${CYAN}[6/7] Formula-Tron Import Check${NC}"

echo -n "  Core modules... "

# Ensure ROS + workspace overlay are sourced for import checks
if [ -f /opt/ros/foxy/setup.bash ]; then
    source /opt/ros/foxy/setup.bash
fi
if [ -f "$WS_ROOT/install/setup.bash" ]; then
    source "$WS_ROOT/install/setup.bash"
fi

cd "$FORMULA_TRON_DIR" 2>/dev/null || true
IMPORT_ERROR=$(python3 -c "
from formula_tron.vision_controller import VisionController
from formula_tron.control_gui import ControlGUI
from formula_tron.utils.track_detection import TrackDetector
from formula_tron.utils.safety import WatchdogTimer
from formula_tron import config
" 2>&1)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC} (import errors)"
    echo "$IMPORT_ERROR"
    FAILED=1
fi


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 7. Hardware (informational only, never blocks)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo ""
echo -e "${CYAN}[7/7] Hardware Status (informational)${NC}"

if command -v ros2 &> /dev/null; then
    echo -n "  VESC driver... "
    if ros2 node list 2>/dev/null | grep -q "vesc"; then
        echo -e "${GREEN}RUNNING${NC}"
    else
        echo -e "${YELLOW}NOT RUNNING${NC} (bringup launch will start it)"
    fi

    echo -n "  Camera... "
    if ros2 topic list 2>/dev/null | grep -qE "(/camera/camera/color/image_raw|/camera/color/image_raw)"; then
        echo -e "${GREEN}RUNNING${NC}"
    else
        echo -e "${YELLOW}NOT RUNNING${NC} (bringup launch will start it)"
    fi
else
    echo -e "  ${YELLOW}Skipping hardware checks (ros2 not available)${NC}"
fi

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Summary
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo ""
echo -e "${BOLD}â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€${NC}"
if [ $FIXED -gt 0 ]; then
    echo -e "${GREEN}  Auto-fixed $FIXED issue(s)${NC}"
fi
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  ALL CHECKS PASSED - Ready to run!${NC}"
    echo -e "${BOLD}â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}${BOLD}  SOME CHECKS FAILED - See errors above${NC}"
    echo -e "${BOLD}â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€${NC}"
    echo ""
    echo -e "  ${YELLOW}If many ROS packages missing, try:${NC}"
    echo -e "    sudo apt install ros-foxy-desktop"
    echo ""
    exit 1
fi

