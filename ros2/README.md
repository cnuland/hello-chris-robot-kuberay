# ROS 2 interface layer

ROS 2 (Humble or Jazzy) is the integration surface customers recognise; planning stays in MuJoCo.

* `allegro_hand_bringup/` - launch files and `ros2_control` config for the Allegro Hand on a fixed base, with the
  `mujoco_ros2_control` plugin for simulation and the vendor driver for hardware.
* The student policy publishes joint targets to the `JointGroupPositionController` at 20 Hz; the same topic is used
  in sim and on the real hand.
* Gazebo is wired only for integration tests of the ROS graph; it is not the planning simulator (no exact snapshot).

TODO when the project starts: colcon workspace, MJCF/URDF pairing (Menagerie `wonik_allegro` + URDF from the vendor
package), controller yaml, a `policy_bridge` node that wraps `app/decision_client.py`.
