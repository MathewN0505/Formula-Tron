from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    
    vision_node = Node(
        package='formula_tron',
        executable='vision_controller',
        name='vision_controller',
        output='screen',
        parameters=[{
            'kp': 0.85,
            'kd': 0.20,
            'base_speed': 1.5,
            'turn_slowdown': 0.35,
            'track_width': 450,
            'steering_bias': 0.0,
            'hsv_h_min': 35,
            'hsv_h_max': 90,
            'hsv_s_min': 50,
            'hsv_v_min': 50,
        }]
    )
    
    gui_node = Node(
        package='formula_tron',
        executable='control_gui',
        name='control_gui',
        output='screen'
    )
    
    gui_delayed = TimerAction(
        period=1.0,
        actions=[gui_node]
    )
    
    return LaunchDescription([
        vision_node,
        gui_delayed
    ])
