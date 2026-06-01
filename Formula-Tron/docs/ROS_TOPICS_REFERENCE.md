```
/ackermann_cmd
/ackermann_cmd_mux/input/navigation
/camera/color/camera_info
/camera/color/image_raw             <-- Used by Vision Controller (CAMERA_TOPIC)
/camera/color/metadata
/camera/depth/camera_info
/camera/depth/image_rect_raw
/camera/depth/metadata
/camera/extrinsics/depth_to_color
/camera/imu                         <-- Available for Lap Timer (sub2)
/commands/motor/brake
/commands/motor/current
/commands/motor/duty_cycle
/commands/motor/position
/commands/motor/speed               <-- Used by Vision Controller (MOTOR_TOPIC)
/commands/servo/position            <-- Used by Vision Controller (SERVO_TOPIC)
/dev/null
/diagnostics
/joy
/joy/set_feedback
/laser_status
/odom
/parameter_events
/rosout
/scan
/sensors/core
/sensors/imu                        <-- Used by Lap Timer (sub1)
/sensors/imu/raw
/sensors/servo_position_command
/teleop
/tf
/tf_static
```

## Configuration Check
Matches found in `config.py`:
- `CAMERA_TOPIC`: `/camera/color/image_raw` (MATCH)
- `MOTOR_TOPIC`: `/commands/motor/speed` (MATCH)
- `SERVO_TOPIC`: `/commands/servo/position` (MATCH)

