# Formula-Tron

**Autonomous F1TENTH Racing Using Computer Vision and Optimal Control**

McMaster Mechatronics Capstone 2025–2026 · Group 6

---

Formula-Tron is a real-time vision-based autonomous racing system built for the [F1TENTH](https://f1tenth.org/) 1/10th-scale platform. It uses an onboard Intel RealSense camera to detect track boundaries via HSV filtering and polynomial fitting, then drives the car using either classical PD control or nonlinear Model Predictive Control (MPC). The system is built on ROS 2 and ships with a full-featured PyQt5 control GUI for live tuning, telemetry, and manual override.

## System Architecture

```
Camera (RealSense D435i)
        │
        ▼
┌──────────────────┐
│   Perception     │  HSV filtering → histogram peak detection
│   Pipeline       │  polynomial lookahead → waypoint extraction
│                  │  AprilTag detection → lap timing
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Controller     │  Poly LookAhead (PD)  ─── classical steering
│   Registry       │  MPC (CasADi/IPOPT)   ─── optimal trajectory
│                  │  Legacy (histogram)    ─── fallback mode
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Safety Layer   │  Watchdog timer (2s timeout)
│                  │  Obstacle detection (depth ROI)
│                  │  Rate limiting (3.2 rad/s, 2.0 m/s²)
│                  │  Steering/speed clamping
└────────┬─────────┘
         │
         ▼
    VESC Motor Controller → Drive
```

## Control Modes

| Mode | Description |
|------|-------------|
| **Poly LookAhead** | Bird's-eye-view polynomial fitting with PD steering control. Default mode. |
| **MPC** | Nonlinear Model Predictive Control using CasADi/IPOPT. Plans optimal trajectories over a receding horizon with speed-adaptive lookahead. |
| **Legacy** | Histogram-based lane detection with PD control. Lightweight fallback mode. |
| **Expo** | On-stand demonstration mode with choreographed motor/servo routines and battery monitoring. |

## Controls

### Keyboard (GUI)

| Key | Action |
|-----|--------|
| W / S | Forward / Reverse |
| A / D | Steer Left / Right |
| SPACE | Emergency Stop |

### Joystick

| Control | Action |
|---------|--------|
| **LB (Left Bumper)** | **Hold for manual override** (instantly disables autonomous) |
| Left Stick Y | Throttle |
| Right Stick X | Steering |

## Build & Run

### On the car (Jetson Orin Nano)

```bash
# Copy to workspace
cp -r formula_tron/ ~/your_ws/src/

# Build
cd ~/your_ws
colcon build --packages-select formula_tron
source install/setup.bash

# Launch
ros2 launch formula_tron bringup.launch.py
```

### GUI only (no car)

The GUI can run standalone for layout testing and tuning. Without camera hardware, topic warnings will appear but the interface is fully functional.

```bash
ros2 run formula_tron control_gui
```

## Tuning

All parameters are adjustable live through the GUI:

- **Kp / Kd** — PD steering gains
- **HSV Thresholds** — Track color detection (auto-calibration available)
- **Auto Speed** — Target autonomous speed
- **MPC Horizon / Weights** — Trajectory optimization parameters (when in MPC mode)

Tuning presets can be saved/loaded as JSON files.

## Safety

The system includes multiple layers of safety:

- **Watchdog Timer** — Automatically stops the car if no control commands are received within 2 seconds
- **Obstacle Detection** — Depth-based ROI filtering detects obstacles and triggers emergency braking
- **Rate Limiting** — Steering rate capped at 3.2 rad/s, acceleration at 2.0 m/s²
- **Joystick Override** — Physical deadman switch (LB) instantly overrides autonomous mode

## Testing

The project includes a comprehensive test suite covering perception, control, safety, and configuration:

```bash
python -m pytest formula_tron/test/unit/ -v
```

251 unit tests across 11 test modules.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Platform | F1TENTH (1/10 scale RC car) |
| Compute | NVIDIA Jetson Orin Nano |
| Camera | Intel RealSense D435i |
| Middleware | ROS 2 Foxy |
| Control GUI | PyQt5 + pyqtgraph |
| MPC Solver | CasADi + IPOPT |
| Vision | OpenCV (HSV filtering, BEV transform, polynomial fitting) |
| Language | Python 3 |

## Project Structure

```
Formula-Tron/
├── formula_tron/
│   ├── formula_tron/
│   │   ├── vision_controller.py   # Main ROS 2 node — perception + control loop
│   │   ├── control_gui.py         # PyQt5 control interface
│   │   ├── config.py              # Global configuration
│   │   └── utils/
│   │       ├── track_detection.py     # HSV + histogram lane detection
│   │       ├── lane_detection.py      # BEV polynomial fitting
│   │       ├── mpc_controller.py      # CasADi MPC solver
│   │       ├── pd_controller.py       # Classical PD controller
│   │       ├── safety.py             # Watchdog + obstacle detection
│   │       ├── lap_timer.py          # AprilTag-based lap timing
│   │       ├── telemetry.py          # Live telemetry recording
│   │       └── vehicle_model.py      # Bicycle dynamics model
│   ├── test/                      # 251 unit tests
│   ├── launch/                    # ROS 2 launch files
│   └── setup.py
├── scripts/                       # Deployment scripts
├── docs/                          # Additional documentation
└── README.md
```

## License

MIT
