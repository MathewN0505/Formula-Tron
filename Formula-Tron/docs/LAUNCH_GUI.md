# Launching the Control GUI (WSL + ROS 2 Foxy)

> This guide assumes WSL2 with Ubuntu 20.04 and ROS 2 Foxy installed.

## Quick Launch

```bash
wsl -d Ubuntu-20.04 -- bash -c "\
  source /opt/ros/foxy/setup.bash && \
  cd /mnt/c/<path-to>/Formula-Tron && \
  colcon build --packages-select formula_tron 2>&1 && \
  source install/setup.bash && \
  export DISPLAY=:0 && \
  ros2 run formula_tron control_gui"
```

Replace `<path-to>` with the actual path to your `Formula-Tron` directory on the Windows side (e.g., `Users/you/Desktop/Formula-Tron`).

## What It Does

1. Sources the ROS 2 Foxy environment
2. Builds the `formula_tron` ROS 2 package via `colcon`
3. Sources the built workspace
4. Sets `DISPLAY=:0` for X11 forwarding (WSLg handles this automatically on Windows 11)
5. Launches the `control_gui` ROS 2 node

## Prerequisites

- **WSL2** with **Ubuntu 20.04** distro
- **ROS 2 Foxy** installed at `/opt/ros/foxy`
- **PyQt5** installed in the WSL Python environment
- **WSLg** or an X server (e.g., VcXsrv) for GUI display

## Full Bringup (vision + GUI in this repo)

After sourcing your workspace, vision controller + GUI:

```bash
source /opt/ros/foxy/setup.bash
cd /path/to/your/ws
source install/setup.bash
ros2 launch formula_tron bringup.launch.py
```

**Note:** Some docs or older notes refer to `RUN.sh` for full car bringup (drivers, camera, etc.). That script is **not** part of this repository clone; use `ros2 launch` as above, or maintain `RUN.sh` only on the vehicle if your team uses it.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No module named 'rclpy'` | Ensure `source /opt/ros/foxy/setup.bash` runs before Python |
| GUI doesn't appear | Check `echo $DISPLAY` is set (should be `:0`) |
| `colcon build` fails | Check Python dependencies: `pip3 install pyqtgraph casadi` |
| `ros2 topic list` timeout | Normal if no car is connected — GUI still works |
| `RUN.sh` not found | Expected in-repo — use `ros2 launch formula_tron bringup.launch.py` after build, or your team's on-car script |
