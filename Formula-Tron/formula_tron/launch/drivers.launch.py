from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    f1tenth_share = get_package_share_directory('f1tenth_stack')
    
    joy_config = os.path.join(f1tenth_share, 'config', 'joy_teleop.yaml')
    vesc_config = os.path.join(f1tenth_share, 'config', 'vesc.yaml')

    ld = LaunchDescription()

    # Formula-Tron publishes direct VESC commands on /commands/motor/speed and
    # /commands/servo/position. Do not launch ackermann_to_vesc / ackermann_mux here,
    # otherwise two pipelines can fight over the same actuator topics.
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy',
        parameters=[joy_config]
    )

    vesc_driver_node = Node(
        package='vesc_driver',
        executable='vesc_driver_node',
        name='vesc_driver_node',
        parameters=[vesc_config]
    )

    vesc_to_odom_node = Node(
        package='vesc_ackermann',
        executable='vesc_to_odom_node',
        name='vesc_to_odom_node',
        parameters=[vesc_config]
    )

    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_baselink_to_laser',
        arguments=['0.27', '0.0', '0.11', '0.0', '0.0', '0.0', 'base_link', 'laser']
    )

    ld.add_action(joy_node)
    ld.add_action(vesc_driver_node)
    ld.add_action(vesc_to_odom_node)
    ld.add_action(static_tf_node)

    return ld
