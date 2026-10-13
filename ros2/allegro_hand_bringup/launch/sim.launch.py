"""Template launch: robot_state_publisher + controller_manager with the MuJoCo ros2_control plugin.
TODO: fill in the xacro path and the mujoco_ros2_control node once the workspace exists."""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": "TODO: xacro allegro_hand_fixed_base.urdf.xacro"}]),
        Node(package="controller_manager", executable="spawner", arguments=["joint_state_broadcaster"]),
        Node(package="controller_manager", executable="spawner", arguments=["hand_position_controller"]),
    ])
