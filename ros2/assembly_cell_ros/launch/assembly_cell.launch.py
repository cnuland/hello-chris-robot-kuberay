"""Launch the multi-part assembly cell graph: cell_sim + part_inspector + supervisor + student_controller."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration as C
from launch_ros.actions import Node


def generate_launch_description():
    args = [DeclareLaunchArgument(n, default_value=v) for n, v in (
        ("seed", "1"), ("parts", "bracket,cap,lbracket"), ("checkpoint", ""), ("dgemma_url", ""), ("record_dir", "/data"))]
    common = {"parts": C("parts")}
    sim = Node(package="assembly_cell_ros", executable="cell_sim", name="cell_sim", output="screen",
               parameters=[{**common, "seed": C("seed"), "record_dir": C("record_dir")}])
    return LaunchDescription(args + [
        sim,
        # the episode ends when the cell finishes: take the whole graph down so the Job can upload its results
        RegisterEventHandler(OnProcessExit(target_action=sim, on_exit=[EmitEvent(event=Shutdown(reason="episode done"))])),
        Node(package="assembly_cell_ros", executable="student_controller", name="student_controller", output="screen",
             parameters=[{**common, "checkpoint": C("checkpoint")}]),
        Node(package="assembly_cell_ros", executable="part_inspector", name="part_inspector", output="screen",
             parameters=[{**common, "dgemma_url": C("dgemma_url")}]),
        Node(package="assembly_cell_ros", executable="supervisor", name="supervisor", output="screen",
             parameters=[{"dgemma_url": C("dgemma_url")}]),
    ])
